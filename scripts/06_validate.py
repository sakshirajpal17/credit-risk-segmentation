"""
Script 06: Validate deduplication precision and recall against ground truth.

Since the data is synthetic, every pull carries the true applicant_id.
We compare our predicted clusters (applicant_id_dedup) to those ground-truth groups.

Metrics:
  Precision = TP / (TP + FP)
    A "true positive" pair = two pulls predicted in the same cluster AND sharing a true applicant_id.
    A "false positive" pair = two pulls in the same predicted cluster with DIFFERENT true applicant_ids.

  Recall = TP / (TP + FN)
    A "false negative" = same true applicant whose pulls were split into >1 predicted cluster.

  F1 = harmonic mean of precision and recall

Also reports:
  - Score-band distribution before/after dedup
  - Pull-count distribution
  - Blind-spot cohort breakdown
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEDUPED_PULLS_CSV, FEATURES_CSV, SEGMENTS_CSV


def compute_pair_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Enumerate all within-cluster pairs and count TP / FP / FN.
    Optimised: work at the cluster level, not pair level.
    """
    true_pos = false_pos = false_neg = 0

    # ── Precision: look inside each predicted cluster ─────────────────────
    for _, grp in tqdm(
        df.groupby("applicant_id_dedup"),
        desc="Precision pass",
        total=df["applicant_id_dedup"].nunique(),
    ):
        if len(grp) < 2:
            continue
        true_ids = grp["applicant_id"].values
        for i, j in combinations(range(len(true_ids)), 2):
            if true_ids[i] == true_ids[j]:
                true_pos += 1
            else:
                false_pos += 1

    # ── Recall: look inside each true applicant group ─────────────────────
    for _, grp in tqdm(
        df.groupby("applicant_id"),
        desc="Recall pass",
        total=df["applicant_id"].nunique(),
    ):
        if len(grp) < 2:
            continue
        pred_ids = grp["applicant_id_dedup"].values
        for i, j in combinations(range(len(pred_ids)), 2):
            if pred_ids[i] != pred_ids[j]:
                false_neg += 1

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall    = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def score_band(score: int) -> str:
    if score >= 750: return "750–900 Prime"
    if score >= 700: return "700–749 Preferred"
    if score >= 650: return "650–699 Standard"
    if score >= 600: return "600–649 Sub-Prime"
    return            "300–599 Deep Sub-Prime"


def main():
    print(f"Loading {DEDUPED_PULLS_CSV} …")
    df = pd.read_csv(DEDUPED_PULLS_CSV)
    print(f"  {len(df):,} pulls | true applicants: {df['applicant_id'].nunique():,} | "
          f"predicted: {df['applicant_id_dedup'].nunique():,}")

    # ── Deduplication quality ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DEDUPLICATION QUALITY METRICS")
    print("=" * 60)
    precision, recall, f1 = compute_pair_metrics(df)
    print(f"  Match Precision : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Link Recall     : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  F1 Score        : {f1:.4f}  ({f1*100:.2f}%)")

    # How many true applicants were completely un-split?
    split = (
        df.groupby("applicant_id")["applicant_id_dedup"]
        .nunique()
        .gt(1)
        .sum()
    )
    print(f"\n  Split applicants (recall failures): {split:,}  "
          f"({split/df['applicant_id'].nunique()*100:.1f}%)")

    merged = (
        df.groupby("applicant_id_dedup")["applicant_id"]
        .nunique()
        .gt(1)
        .sum()
    )
    print(f"  Merged clusters  (precision failures): {merged:,}")

    # ── Score band distribution ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SCORE-BAND DISTRIBUTION (latest pull per applicant)")
    print("=" * 60)
    latest = (
        df.sort_values("pull_date")
        .groupby("applicant_id_dedup")["credit_score"]
        .last()
        .reset_index()
    )
    latest["band"] = latest["credit_score"].apply(score_band)
    band_tbl = (
        latest["band"]
        .value_counts()
        .rename_axis("Band")
        .reset_index(name="Count")
        .assign(Pct=lambda x: (x["Count"] / x["Count"].sum() * 100).round(1))
    )
    print(band_tbl.sort_values("Band").to_string(index=False))

    # ── Segments summary ──────────────────────────────────────────────────
    if SEGMENTS_CSV.exists():
        print("\n" + "=" * 60)
        print("SEGMENTATION SUMMARY")
        print("=" * 60)
        segs = pd.read_csv(SEGMENTS_CSV)
        cohort_tbl = (
            segs.groupby("risk_cohort")
            .agg(
                count=("applicant_id", "count"),
                avg_score=("latest_score", "mean"),
                avg_activation=("activation_score", "mean"),
                blind_spots=("blind_spot_flag", "sum"),
            )
            .assign(pct=lambda x: (x["count"] / x["count"].sum() * 100).round(1))
        )
        print(cohort_tbl.round(2).to_string())

        n_blind = segs["blind_spot_flag"].sum()
        print(f"\n  Total blind-spot flags: {n_blind:,}  "
              f"({n_blind/len(segs)*100:.1f}% of portfolio)")

    # ── Pull-count histogram ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PULL-COUNT DISTRIBUTION (predicted applicants)")
    print("=" * 60)
    pc = (
        df.groupby("applicant_id_dedup")
        .size()
        .value_counts()
        .sort_index()
        .reset_index()
        .rename(columns={"applicant_id_dedup": "pulls_per_applicant", "count": "applicants"})
    )
    pc["pct"] = (pc["applicants"] / pc["applicants"].sum() * 100).round(1)
    print(pc.to_string(index=False))


if __name__ == "__main__":
    main()
