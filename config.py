"""
Central config: all paths, constants, and tunable parameters.
Never hardcode these in scripts — import from here.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_OUTPUTS   = PROJECT_ROOT / "data" / "outputs"

for _d in [DATA_RAW, DATA_PROCESSED, DATA_OUTPUTS]:
    _d.mkdir(parents=True, exist_ok=True)

# File paths
BUREAU_PULLS_CSV  = DATA_RAW       / "bureau_pulls.csv"
DEDUPED_PULLS_CSV = DATA_PROCESSED / "deduped_pulls.csv"
APPLICANTS_CSV    = DATA_PROCESSED / "applicants.csv"
FEATURES_CSV      = DATA_PROCESSED / "features.csv"
SEGMENTS_CSV      = DATA_OUTPUTS   / "segments.csv"

# Simulation parameters
N_APPLICANTS  = 17_000
TARGET_PULLS  = 50_000
RANDOM_SEED   = 42
PULL_START    = "2023-01-01"
PULL_END      = "2025-01-01"
REFERENCE_DATE = "2025-01-01"  # anchor for recency features

# Deduplication
DEDUP_THRESHOLD     = 0.80   # composite similarity score to declare a match
MAX_BLOCK_SIZE      = 500    # skip pathological blocks

# ClickHouse
CLICKHOUSE_HOST     = "localhost"
CLICKHOUSE_PORT     = 8123
CLICKHOUSE_USER     = "default"
CLICKHOUSE_PASSWORD = ""
CLICKHOUSE_DATABASE = "credit_bureau"

# Domain constants
BUREAUS = ["CIBIL", "Experian", "Equifax", "CRIF"]

LENDERS = [
    "HDFC Bank", "ICICI Bank", "SBI", "Axis Bank", "Kotak Mahindra Bank",
    "Yes Bank", "IndusInd Bank", "Bajaj Finance", "Tata Capital",
    "HDB Financial Services", "Aditya Birla Finance", "L&T Finance",
    "Muthoot Finance", "Manappuram Finance", "IDFC First Bank",
]

LOAN_TYPES = [
    "personal_loan", "home_loan", "credit_card",
    "auto_loan", "business_loan", "consumer_durable",
]

# Risk cohort labels (used in reports and dashboard)
COHORT_LABELS = {
    "A_PrimeStable":  "Prime Stable  (≥750)",
    "B_PrimeActive":  "Prime Active  (700–749)",
    "C_NearPrime":    "Near Prime    (600–699)",
    "D_Subprime":     "Subprime      (<600)",
    "E_ThinFile":     "Thin File     (<3 tradelines)",
}
