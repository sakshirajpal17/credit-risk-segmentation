"""
Script 05: Load all processed data into ClickHouse.

Assumes ClickHouse is running locally (Docker or native install).
Quick start: docker-compose up -d  (see docker-compose.yml)

Tables created:
  credit_bureau.bureau_pulls       — raw pull records, partitioned by month
  credit_bureau.applicants         — deduplicated applicant master
  credit_bureau.pull_applicant_map — pull_id → applicant_id with confidence
  credit_bureau.applicant_features — 12 engineered features per applicant
  credit_bureau.segments           — risk cohort, activation score, blind-spot flag

Run this AFTER scripts 01–04 have produced their output CSVs.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    APPLICANTS_CSV, BUREAU_PULLS_CSV, CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST, CLICKHOUSE_PASSWORD, CLICKHOUSE_PORT,
    CLICKHOUSE_USER, DEDUPED_PULLS_CSV, FEATURES_CSV, SEGMENTS_CSV,
)

try:
    import clickhouse_connect
except ImportError:
    print("clickhouse-connect not installed. Run: pip install clickhouse-connect")
    sys.exit(1)


def get_client():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
    )


DDL = f"""
CREATE DATABASE IF NOT EXISTS {CLICKHOUSE_DATABASE};

CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DATABASE}.bureau_pulls
(
    pull_id               String,
    pull_date             Date,
    lender                LowCardinality(String),
    bureau                LowCardinality(String),
    loan_type             LowCardinality(String),
    pan                   String,
    mobile                String,
    dob                   String,
    name                  String,
    credit_score          Int16,
    num_tradelines        Int16,
    num_enquiries         Int16,
    num_delinquencies     Int16,
    credit_limit          Int64,
    credit_used           Int64,
    oldest_account_months Int16,
    ingested_at           DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(pull_date)
ORDER BY (pan, pull_date)
TTL pull_date + INTERVAL 7 YEAR
SETTINGS index_granularity = 8192;

CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DATABASE}.applicants
(
    applicant_id     String,
    canonical_pan    String,
    canonical_mobile String,
    canonical_name   String,
    pull_count       Int16,
    first_pull_date  String,
    last_pull_date   String,
    lender_count     Int16,
    bureau_count     Int16
)
ENGINE = MergeTree()
ORDER BY applicant_id;

CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DATABASE}.pull_applicant_map
(
    pull_id          String,
    applicant_id     String,
    match_confidence Float32
)
ENGINE = MergeTree()
ORDER BY (applicant_id, pull_id);

CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DATABASE}.applicant_features
(
    applicant_id          String,
    latest_score          Int16,
    earliest_score        Int16,
    score_delta           Int16,
    score_volatility      Float32,
    pull_count            Int16,
    pull_freq_30d         Int16,
    pull_freq_90d         Int16,
    days_since_first_pull Int16,
    days_since_last_pull  Int16,
    lender_diversity      Int16,
    multi_lender_flag     UInt8,
    thin_file_flag        UInt8,
    avg_delinquencies     Float32,
    avg_enquiries         Float32,
    avg_utilization       Float32,
    first_pull_date       String,
    last_pull_date        String
)
ENGINE = MergeTree()
ORDER BY applicant_id;

CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DATABASE}.segments
(
    applicant_id      String,
    latest_score      Int16,
    score_delta       Int16,
    score_volatility  Float32,
    pull_count        Int16,
    pull_freq_30d     Int16,
    pull_freq_90d     Int16,
    lender_diversity  Int16,
    multi_lender_flag UInt8,
    thin_file_flag    UInt8,
    risk_cohort       LowCardinality(String),
    activation_score  Float32,
    blind_spot_flag   UInt8
)
ENGINE = MergeTree()
ORDER BY (risk_cohort, applicant_id);
"""


def create_schema(client):
    print("Creating schema …")
    for stmt in DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            client.command(stmt)
    # Truncate tables so re-runs don't duplicate data
    for tbl in ["bureau_pulls", "applicants", "pull_applicant_map", "applicant_features", "segments"]:
        client.command(f"TRUNCATE TABLE IF EXISTS {CLICKHOUSE_DATABASE}.{tbl}")
    print("  Schema ready.")


def _coerce_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Cast any column that should be String but pandas inferred as numeric (e.g. mobile numbers)."""
    import datetime as _dt
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            # Skip columns that already contain date/datetime objects (leave those for ClickHouse Date type)
            sample = df[col].dropna()
            if len(sample) > 0 and isinstance(sample.iloc[0], (_dt.date, _dt.datetime)):
                continue
            df[col] = df[col].astype(str).replace("nan", "")
        elif df[col].dtype in ("int64", "float64") and any(
            kw in col for kw in ("mobile", "pan", "dob", "name", "id_dedup")
        ):
            df[col] = df[col].astype(str)
    return df


def load_table(client, table: str, df: pd.DataFrame, batch_size: int = 10_000):
    df = _coerce_strings(df)
    total = len(df)
    print(f"  Loading {total:,} rows into {table} …")
    for start in range(0, total, batch_size):
        chunk = df.iloc[start : start + batch_size]
        client.insert_df(f"{CLICKHOUSE_DATABASE}.{table}", chunk)
    print(f"    ✓ {total:,} rows loaded")


def main():
    client = get_client()
    create_schema(client)

    # Raw pulls
    print("\nLoading bureau_pulls …")
    bp = pd.read_csv(BUREAU_PULLS_CSV, dtype={"mobile": str, "pan": str, "dob": str, "name": str})
    bp["pull_date"] = pd.to_datetime(bp["pull_date"]).dt.date
    load_table(client, "bureau_pulls", bp[[
        "pull_id", "pull_date", "lender", "bureau", "loan_type",
        "pan", "mobile", "dob", "name", "credit_score",
        "num_tradelines", "num_enquiries", "num_delinquencies",
        "credit_limit", "credit_used", "oldest_account_months",
    ]])

    # Applicant master
    print("\nLoading applicants …")
    apps = pd.read_csv(APPLICANTS_CSV)
    load_table(client, "applicants", apps)

    # Pull-applicant map
    print("\nLoading pull_applicant_map …")
    dedup = pd.read_csv(DEDUPED_PULLS_CSV)[["pull_id", "applicant_id_dedup", "match_confidence"]]
    dedup = dedup.rename(columns={"applicant_id_dedup": "applicant_id"})
    load_table(client, "pull_applicant_map", dedup)

    # Features
    print("\nLoading applicant_features …")
    feats = pd.read_csv(FEATURES_CSV)
    feats = feats.rename(columns={"applicant_id": "applicant_id"})
    load_table(client, "applicant_features", feats)

    # Segments
    print("\nLoading segments …")
    segs = pd.read_csv(SEGMENTS_CSV)
    seg_cols = [
        "applicant_id", "latest_score", "score_delta", "score_volatility",
        "pull_count", "pull_freq_30d", "pull_freq_90d", "lender_diversity",
        "multi_lender_flag", "thin_file_flag", "risk_cohort",
        "activation_score", "blind_spot_flag",
    ]
    load_table(client, "segments", segs[seg_cols])

    # Quick verification
    print("\nRow counts:")
    for tbl in ["bureau_pulls", "applicants", "pull_applicant_map", "applicant_features", "segments"]:
        n = client.command(f"SELECT count() FROM {CLICKHOUSE_DATABASE}.{tbl}")
        print(f"  {tbl}: {n:,}")


if __name__ == "__main__":
    main()
