"""
Script 04: Segment applicants into 5 risk cohorts and score activation likelihood.

Cohorts (applied in priority order — E overrides score-based cohorts):
  A_PrimeStable  — score ≥ 750, pull_count ≤ 3, low delinquency
  B_PrimeActive  — score ≥ 700 with multi-lender activity (rate shopping)
  C_NearPrime    — 600 ≤ score < 700
  D_Subprime     — score < 600
  E_ThinFile     — avg tradelines < 3 (new-to-credit / thin bureau file)

Activation score (0–1): proxy for likelihood to respond to a credit product offer.
  = 0.30 × normalised(positive score delta)
  + 0.25 × normalised(recency — days since last pull, inverted)
  + 0.25 × normalised(pull activity in 90-day window)
  + 0.20 × normalised(lender diversity)

Policy blind-spot flag: applicants who would typically fail standard underwriting
(score < 650 or thin file) but carry positive activation signals.
Target: ≈ 18% of the scored population.

Output: data/outputs/segments.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import COHORT_LABELS, FEATURES_CSV, SEGMENTS_CSV


# ---------------------------------------------------------------------------
# Cohort assignment
# ---------------------------------------------------------------------------

def assign_cohort(row: pd.Series) -> str:
    if row["thin_file_flag"]:
        return "E_ThinFile"
    if row["latest_score"] >= 750:
        return "A_PrimeStable"
    if row["latest_score"] >= 700:
        return "B_PrimeActive" if row["multi_lender_flag"] else "A_PrimeStable"
    if row["latest_score"] >= 600:
        return "C_NearPrime"
    return "D_Subprime"


# ---------------------------------------------------------------------------
# Activation score
# ---------------------------------------------------------------------------

def compute_activation_scores(df: pd.DataFrame) -> pd.Series:
    """
    Higher activation = more likely to respond to a credit product offer.
    All four components are independently min-max normalised before weighting.
    """
    scaler = MinMaxScaler()

    # Component 1: score improvement (only positive deltas rewarded)
    positive_delta = df["score_delta"].clip(lower=0)

    # Component 2: recency (lower days_since_last_pull → higher score)
    recency_inv = 1.0 / (1.0 + df["days_since_last_pull"])

    # Component 3: pull activity in 90-day window (capped at 8 to avoid penalising)
    activity = df["pull_freq_90d"].clip(upper=8)

    # Component 4: lender diversity (capped at 7)
    diversity = df["lender_diversity"].clip(upper=7)

    components = np.column_stack([positive_delta, recency_inv, activity, diversity])
    normed     = scaler.fit_transform(components)

    weights = np.array([0.30, 0.25, 0.25, 0.20])
    raw     = normed @ weights
    # Clip to [0, 1] and round
    return pd.Series(np.round(np.clip(raw, 0, 1), 4), index=df.index)


# ---------------------------------------------------------------------------
# Policy blind-spot detection
# ---------------------------------------------------------------------------

def flag_blind_spots(df: pd.DataFrame) -> pd.Series:
    """
    Flag applicants who fail standard policy thresholds but show positive signals.
    Target: ~18% of portfolio. Three qualifying conditions (any one sufficient):

      C1. Thin file + any score improvement + repeat inquiry activity
          → New-to-credit but bureau is building; banks are testing the waters

      C2. Subprime (< 650) + meaningful score recovery + multi-lender demand
          → Score trajectory reversing; 2+ lenders already interested

      C3. Near-prime (600–699) + multi-lender demand + non-negative trend
          → Multiple banks willing to lend; standard policy decline leaves revenue
    """
    c1 = (
        (df["thin_file_flag"] == 1)
        & (df["score_delta"] > 12)        # material improvement
        & (df["pull_count"] >= 2)          # at least one repeat inquiry
    )
    c2 = (
        (df["latest_score"] < 650)
        & (df["score_delta"] > 25)
        & (df["lender_diversity"] >= 2)
        & (df["avg_delinquencies"] < 1)
    )
    c3 = (
        (df["latest_score"] >= 600)
        & (df["latest_score"] < 700)
        & (df["lender_diversity"] >= 2)
        & (df["score_delta"] >= 10)
    )
    return (c1 | c2 | c3).astype(int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading {FEATURES_CSV} …")
    df = pd.read_csv(FEATURES_CSV)
    print(f"  {len(df):,} applicants loaded")

    print("\nAssigning risk cohorts …")
    df["risk_cohort"] = df.apply(assign_cohort, axis=1)

    print("Computing activation scores …")
    df["activation_score"] = compute_activation_scores(df)

    print("Flagging policy blind spots …")
    df["blind_spot_flag"] = flag_blind_spots(df)

    df.to_csv(SEGMENTS_CSV, index=False)
    print(f"\n✓  Segments → {SEGMENTS_CSV}  shape: {df.shape}")

    # ── Summary report ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RISK COHORT DISTRIBUTION")
    print("=" * 60)
    cohort_summary = (
        df.groupby("risk_cohort")
        .agg(
            count=("applicant_id", "count"),
            avg_score=("latest_score", "mean"),
            avg_activation=("activation_score", "mean"),
            blind_spots=("blind_spot_flag", "sum"),
        )
        .assign(pct=lambda x: x["count"] / x["count"].sum() * 100)
        .round(2)
        .sort_values("count", ascending=False)
    )
    cohort_summary.index = cohort_summary.index.map(
        lambda c: COHORT_LABELS.get(c, c)
    )
    print(cohort_summary.to_string())

    total        = len(df)
    blind_count  = df["blind_spot_flag"].sum()
    blind_pct    = blind_count / total * 100
    print(f"\n  Total applicants:  {total:,}")
    print(f"  Blind-spot flags:  {blind_count:,}  ({blind_pct:.1f}% of portfolio)")
    print(f"  Avg activation:    {df['activation_score'].mean():.3f}")
    print(
        f"  Blind-spot avg activation: "
        f"{df.loc[df['blind_spot_flag']==1,'activation_score'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
