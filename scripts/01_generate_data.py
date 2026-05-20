"""
Script 01: Simulate 50K bureau pull records for ~17K unique applicants.

Generates realistic CIBIL/Experian/Equifax/CRIF pull data with:
  - Indian PAN (AAAAA9999A format), DOB, mobile, name
  - Noise injected per pull: OCR errors, format variations, digit swaps
  - 1–8 pulls per applicant reflecting real loan-shopping behaviour
  - Score drift across pulls (±15 pts/pull) with directional trend
  - Thin-file cohort (~28% of applicants with <3 tradelines)

Ground-truth applicant_id is embedded for pipeline validation (not used by dedup).
"""

import random
import sys
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    BUREAUS, BUREAU_PULLS_CSV, LENDERS, LOAN_TYPES,
    N_APPLICANTS, PULL_END, PULL_START, RANDOM_SEED, TARGET_PULLS,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Indian name corpus
# ---------------------------------------------------------------------------
FIRST_NAMES = [
    "Aarav", "Aditi", "Ajay", "Akash", "Amit", "Amita", "Anjali", "Ankit",
    "Anuja", "Arjun", "Arun", "Ashish", "Deepak", "Deepika", "Dinesh",
    "Divya", "Gaurav", "Geeta", "Harsh", "Hemant", "Ishaan", "Jyoti",
    "Karan", "Kavita", "Kunal", "Lata", "Madhuri", "Manoj", "Meena",
    "Mohit", "Nalini", "Neha", "Nikhil", "Nitin", "Pallavi", "Pankaj",
    "Pooja", "Priya", "Rahul", "Raj", "Rakesh", "Ravi", "Rekha", "Rita",
    "Rohit", "Ritu", "Sanjay", "Sangeeta", "Saurabh", "Seema", "Shweta",
    "Smita", "Sneha", "Sonia", "Sunil", "Suresh", "Tarun", "Usha",
    "Varun", "Vijay", "Vikram", "Vishal", "Vivek", "Yash", "Zara",
]
LAST_NAMES = [
    "Agarwal", "Aggarwal", "Anand", "Bhandari", "Bhatnagar", "Bose",
    "Chauhan", "Chatterjee", "Chopra", "Choudhary", "Das", "Dubey",
    "Goyal", "Gupta", "Iyer", "Jain", "Joshi", "Kaur", "Kapoor",
    "Khanna", "Kumar", "Malhotra", "Mathur", "Mehta", "Menon", "Mishra",
    "Mukherjee", "Nair", "Pandey", "Patel", "Pillai", "Rao", "Rastogi",
    "Reddy", "Saxena", "Sehgal", "Shah", "Sharma", "Shetty", "Shukla",
    "Singh", "Srivastava", "Tiwari", "Trivedi", "Verma", "Walia", "Yadav",
]


# ---------------------------------------------------------------------------
# Identifier generators
# ---------------------------------------------------------------------------

def _random_pan() -> str:
    """AAAAA9999A — personal PAN (4th char always 'P')."""
    alpha = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # common chars; exclude I, O to reduce confusion
    prefix = "".join(random.choices(alpha, k=3))
    surname_initial = random.choice(alpha)
    seq = f"{random.randint(1, 9999):04d}"
    check = random.choice(alpha)
    return f"{prefix}P{surname_initial}{seq}{check}"


def _random_mobile() -> str:
    """10-digit Indian mobile starting with 6–9."""
    first = str(random.choice([6, 7, 8, 9]))
    rest = "".join([str(random.randint(0, 9)) for _ in range(9)])
    return first + rest


def _random_dob(min_age: int = 21, max_age: int = 65) -> date:
    today = date.today()
    lo = today - timedelta(days=max_age * 365)
    hi = today - timedelta(days=min_age * 365)
    return lo + timedelta(days=random.randint(0, (hi - lo).days))


def _random_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def _cibil_score(base: int, trend: int, pull_index: int) -> int:
    """Simulate score at a given pull. Trend is -1, 0, or +1."""
    drift = trend * random.randint(0, 15) * pull_index
    noise = int(np.random.normal(0, 8))
    return max(300, min(900, base + drift + noise))


# ---------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------

def _corrupt_pan(pan: str) -> str:
    """85% pass-through; 15% realistic OCR / data-entry errors."""
    if random.random() > 0.15:
        return pan
    pan = list(pan)
    err = random.choices(
        ["transpose", "ocr_O0", "ocr_I1", "case_lower"],
        weights=[40, 25, 25, 10],
    )[0]
    if err == "transpose":
        # Swap two adjacent digits in positions 5-8
        i = random.randint(5, 7)
        pan[i], pan[i + 1] = pan[i + 1], pan[i]
    elif err == "ocr_O0":
        for i in range(5, 9):
            if pan[i] == "0":
                pan[i] = "O"
    elif err == "ocr_I1":
        if pan[9] == "I":
            pan[9] = "1"
        elif pan[9] == "1":
            pan[9] = "I"
    elif err == "case_lower":
        i = random.randint(0, 4)
        pan[i] = pan[i].lower()
    return "".join(pan)


def _corrupt_mobile(mobile: str) -> str:
    """Inject common mobile format variants."""
    r = random.random()
    if r < 0.55:
        return mobile           # bare 10 digits (most common)
    elif r < 0.82:
        return f"+91{mobile}"   # international prefix
    elif r < 0.92:
        return f"0{mobile}"     # legacy trunk prefix
    else:
        # Single digit transcription error
        i = random.randint(1, 9)
        m = list(mobile)
        m[i] = str((int(m[i]) + random.randint(1, 3)) % 10)
        return "".join(m)


def _corrupt_dob(dob: date) -> str:
    """Return DOB as a string in one of several real-world formats."""
    if random.random() < 0.10:
        dob = dob + timedelta(days=random.choice([-1, 1]))  # ±1 day entry error
    r = random.random()
    if r < 0.60:
        return dob.strftime("%d%m%Y")       # DDMMYYYY  (CIBIL/Experian standard)
    elif r < 0.78:
        return dob.strftime("%d/%m/%Y")
    elif r < 0.90:
        return dob.strftime("%d-%m-%Y")
    else:
        return dob.strftime("%Y-%m-%d")     # ISO — occasionally seen in REST payloads


def _corrupt_name(name: str) -> str:
    """Name variations: abbreviations, order swaps, minor spellings."""
    r = random.random()
    if r < 0.72:
        return name
    parts = name.split()
    if r < 0.86:
        return f"{parts[0][0]}. {parts[-1]}"   # "A. Sharma"
    else:
        return f"{parts[-1]}, {parts[0]}"       # "Sharma, Amit"


# ---------------------------------------------------------------------------
# Pull-count assignment
# ---------------------------------------------------------------------------

_PULL_WEIGHTS = [25, 25, 20, 14, 8, 4, 3, 1]   # for counts 1..8


def _assign_pull_counts(n: int, target: int) -> np.ndarray:
    """Randomly assign pull counts so the total is close to `target`."""
    counts = np.random.choice(
        range(1, len(_PULL_WEIGHTS) + 1),
        size=n,
        p=np.array(_PULL_WEIGHTS) / sum(_PULL_WEIGHTS),
    )
    # Scale up if needed (add extras to heavy-pull applicants)
    diff = target - counts.sum()
    if diff > 0:
        heavy = np.where(counts >= 3)[0]
        idx = np.random.choice(heavy, size=min(diff, len(heavy)), replace=False)
        counts[idx] += 1
    return counts


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_applicants(n: int) -> list[dict]:
    applicants = []
    for _ in tqdm(range(n), desc="Generating applicants"):
        base_score = max(300, min(900, int(np.random.normal(665, 95))))
        score_trend = random.choices([-1, 0, 0, 1, 1], weights=[15, 25, 10, 30, 20])[0]

        # Thin-file: ~28% of applicants have <3 tradelines (new-to-credit)
        if random.random() < 0.28:
            num_tradelines = random.randint(0, 2)
            base_score = max(300, min(700, base_score))   # thin files skew lower
        else:
            num_tradelines = max(3, int(np.random.poisson(5.5)))

        delinquencies = 0
        if base_score < 650:
            delinquencies = random.choices([0, 1, 2], weights=[55, 30, 15])[0]

        credit_limit = int(
            random.uniform(0.8, 1.2)
            * (
                random.randint(200_000, 5_000_000) if base_score >= 750
                else random.randint(50_000, 1_000_000) if base_score >= 650
                else random.randint(15_000, 300_000)
            )
        )
        utilization = np.clip(
            np.random.normal(
                0.25 if base_score >= 750 else 0.50 if base_score >= 650 else 0.72,
                0.12,
            ),
            0.01, 0.99,
        )

        applicants.append(
            {
                "applicant_id": str(uuid.uuid4()),
                "true_pan": _random_pan(),
                "true_mobile": _random_mobile(),
                "true_dob": _random_dob(),
                "true_name": _random_name(),
                "base_score": base_score,
                "score_trend": score_trend,
                "num_tradelines": num_tradelines,
                "num_delinquencies": delinquencies,
                "credit_limit": credit_limit,
                "utilization": float(utilization),
                "oldest_account_months": max(0, int(np.random.exponential(30))),
            }
        )
    return applicants


def generate_pulls(applicants: list[dict], pull_counts: np.ndarray) -> list[dict]:
    start_dt = date.fromisoformat(PULL_START)
    end_dt   = date.fromisoformat(PULL_END)
    span     = (end_dt - start_dt).days

    pulls = []
    for app, n_pulls in tqdm(zip(applicants, pull_counts), total=len(applicants), desc="Generating pulls"):
        # All pulls for this applicant cluster in a 10–90 day shopping window
        window = random.randint(10, 90)
        base_offset = random.randint(0, span - window)
        base_date = start_dt + timedelta(days=base_offset)

        pull_dates = sorted(
            base_date + timedelta(days=random.randint(0, window))
            for _ in range(n_pulls)
        )

        lenders = random.sample(LENDERS, min(n_pulls, len(LENDERS)))
        if len(lenders) < n_pulls:
            lenders += random.choices(LENDERS, k=n_pulls - len(lenders))

        score = app["base_score"]
        for j, (pdate, lender) in enumerate(zip(pull_dates, lenders)):
            score = _cibil_score(score, app["score_trend"], j)
            util  = float(np.clip(app["utilization"] + np.random.normal(0, 0.07), 0.01, 0.99))

            pulls.append(
                {
                    "pull_id":              str(uuid.uuid4()),
                    "applicant_id":         app["applicant_id"],   # ground truth
                    "pull_date":            pdate.isoformat(),
                    "lender":               lender,
                    "bureau":               random.choice(BUREAUS),
                    "loan_type":            random.choice(LOAN_TYPES),
                    # Noisy identifiers
                    "pan":                  _corrupt_pan(app["true_pan"]),
                    "mobile":               _corrupt_mobile(app["true_mobile"]),
                    "dob":                  _corrupt_dob(app["true_dob"]),
                    "name":                 _corrupt_name(app["true_name"]),
                    # Credit metrics (vary slightly per pull)
                    "credit_score":         score,
                    "num_tradelines":       max(0, app["num_tradelines"] + random.randint(-1, 1)),
                    "num_enquiries":        random.randint(1, 15),
                    "num_delinquencies":    app["num_delinquencies"],
                    "credit_limit":         int(app["credit_limit"] * random.uniform(0.92, 1.08)),
                    "credit_used":          int(app["credit_limit"] * util),
                    "oldest_account_months": max(0, app["oldest_account_months"] + j),
                }
            )
    return pulls


def main():
    print(f"Generating {N_APPLICANTS:,} applicants …")
    applicants  = generate_applicants(N_APPLICANTS)
    pull_counts = _assign_pull_counts(N_APPLICANTS, TARGET_PULLS)

    print(f"\nGenerating pulls (target ≈ {TARGET_PULLS:,}) …")
    pulls = generate_pulls(applicants, pull_counts)

    df = pd.DataFrame(pulls)
    df.to_csv(BUREAU_PULLS_CSV, index=False)

    actual = len(df)
    unique_apps = df["applicant_id"].nunique()
    print(f"\n✓  Saved {actual:,} pulls for {unique_apps:,} applicants → {BUREAU_PULLS_CSV}")
    print(f"   Avg pulls/applicant: {actual/unique_apps:.2f}")
    print(f"   Score range: {df['credit_score'].min()}–{df['credit_score'].max()}")
    print(f"   Score mean:  {df['credit_score'].mean():.0f}")
    print(f"   Bureaus:     {df['bureau'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
