"""
Script 03: Engineer 12 bureau-derived features per deduplicated applicant.

Features (all computed from pull history relative to REFERENCE_DATE = 2025-01-01):
  1.  latest_score          — most recent credit score
  2.  earliest_score        — first recorded score
  3.  score_delta           — latest − earliest (signed; positive = improving)
  4.  score_volatility      — std dev across all pulls (instability signal)
  5.  pull_count            — total bureau pulls (inquiry pressure)
  6.  pull_freq_30d         — pulls in last 30 days (active-shopping signal)
  7.  pull_freq_90d         — pulls in last 90 days (rate-shopping window)
  8.  days_since_first_pull — bureau relationship age
  9.  days_since_last_pull  — recency of credit activity
  10. lender_diversity      — distinct lenders who pulled (demand-side signal)
  11. multi_lender_flag     — binary: 3+ distinct lenders (strong activation signal)
  12. thin_file_flag        — binary: avg tradelines < 3 across pulls

Output: data/processed/features.csv
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEDUPED_PULLS_CSV, FEATURES_CSV, REFERENCE_DATE

REF = date.fromisoformat(REFERENCE_DATE)


def compute_features(group: pd.DataFrame) -> pd.Series:
    """Compute all 12 features for one applicant's pull history."""
    group = group.sort_values("pull_date")

    scores     = group["credit_score"].tolist()
    pull_dates = pd.to_datetime(group["pull_date"]).dt.date.tolist()

    latest_score    = scores[-1]
    earliest_score  = scores[0]
    score_delta     = latest_score - earliest_score
    score_volatility = round(float(np.std(scores)), 2) if len(scores) > 1 else 0.0

    pull_count = len(group)

    days_since = [(REF - d).days for d in pull_dates]
    days_since_first_pull = max(days_since)
    days_since_last_pull  = min(days_since)

    pull_freq_30d = int(sum(d <= 30 for d in days_since))
    pull_freq_90d = int(sum(d <= 90 for d in days_since))

    lender_diversity  = int(group["lender"].nunique())
    multi_lender_flag = int(lender_diversity >= 3)

    avg_tradelines = group["num_tradelines"].mean()
    thin_file_flag = int(avg_tradelines < 3)

    # Extra context columns (not in the 12 core features, but useful downstream)
    avg_delinquencies   = round(group["num_delinquencies"].mean(), 2)
    avg_num_enquiries   = round(group["num_enquiries"].mean(), 2)
    avg_utilization     = round(
        (group["credit_used"] / group["credit_limit"].replace(0, np.nan)).mean(), 4
    )

    return pd.Series(
        {
            # ── Core 12 features ──────────────────────────────────────────
            "latest_score":           latest_score,
            "earliest_score":         earliest_score,
            "score_delta":            score_delta,
            "score_volatility":       score_volatility,
            "pull_count":             pull_count,
            "pull_freq_30d":          pull_freq_30d,
            "pull_freq_90d":          pull_freq_90d,
            "days_since_first_pull":  days_since_first_pull,
            "days_since_last_pull":   days_since_last_pull,
            "lender_diversity":       lender_diversity,
            "multi_lender_flag":      multi_lender_flag,
            "thin_file_flag":         thin_file_flag,
            # ── Supporting context ────────────────────────────────────────
            "avg_delinquencies":      avg_delinquencies,
            "avg_enquiries":          avg_num_enquiries,
            "avg_utilization":        avg_utilization,
            "first_pull_date":        pull_dates[0].isoformat(),
            "last_pull_date":         pull_dates[-1].isoformat(),
        }
    )


def main():
    print(f"Loading {DEDUPED_PULLS_CSV} …")
    df = pd.read_csv(DEDUPED_PULLS_CSV)
    n_applicants = df["applicant_id_dedup"].nunique()
    print(f"  {len(df):,} pulls  |  {n_applicants:,} deduplicated applicants")

    print("\nEngineering features …")
    tqdm.pandas(desc="Applicants")
    features = (
        df.groupby("applicant_id_dedup", group_keys=False)
        .progress_apply(compute_features)
        .reset_index()
        .rename(columns={"applicant_id_dedup": "applicant_id"})
    )

    features.to_csv(FEATURES_CSV, index=False)
    print(f"\n✓  Feature matrix → {FEATURES_CSV}  shape: {features.shape}")

    # Quick sanity checks
    print("\nFeature summary:")
    core = [
        "latest_score", "score_delta", "score_volatility", "pull_count",
        "pull_freq_90d", "lender_diversity",
    ]
    print(features[core].describe().round(2).to_string())

    thin_pct = features["thin_file_flag"].mean() * 100
    ml_pct   = features["multi_lender_flag"].mean() * 100
    print(f"\n  Thin-file applicants:   {thin_pct:.1f}%")
    print(f"  Multi-lender flag:      {ml_pct:.1f}%")


if __name__ == "__main__":
    main()
