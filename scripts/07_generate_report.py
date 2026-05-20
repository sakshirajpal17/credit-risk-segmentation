"""
Script 07: Generate self-contained HTML visual report.

Loads segments.csv and features.csv, renders 10 charts, and writes
viz/credit_bureau_report.html — a single file you can open in any browser
with no server, no login, and no dependencies.

Run AFTER scripts 01–04.
"""

import base64
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURES_CSV, SEGMENTS_CSV, PROJECT_ROOT

VIZ_DIR = PROJECT_ROOT / "viz"
VIZ_DIR.mkdir(exist_ok=True)
REPORT_PATH = VIZ_DIR / "credit_bureau_report.html"

# ── Palette ──────────────────────────────────────────────────────────────────
COHORT_ORDER = ["A_PrimeStable", "B_PrimeActive", "C_NearPrime", "D_Subprime", "E_ThinFile"]
COHORT_LABELS = {
    "A_PrimeStable": "Prime Stable (≥750)",
    "B_PrimeActive": "Prime Active (700–749)",
    "C_NearPrime":   "Near Prime (600–699)",
    "D_Subprime":    "Subprime (<600)",
    "E_ThinFile":    "Thin File (<3 trades)",
}
COHORT_COLORS = {
    "A_PrimeStable": "#22c55e",
    "B_PrimeActive": "#84cc16",
    "C_NearPrime":   "#f59e0b",
    "D_Subprime":    "#ef4444",
    "E_ThinFile":    "#8b5cf6",
}

STYLE = {
    "bg":      "#0f172a",
    "surface": "#1e293b",
    "border":  "#334155",
    "text":    "#f1f5f9",
    "muted":   "#94a3b8",
    "accent":  "#38bdf8",
}

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor":  STYLE["bg"],
    "axes.facecolor":    STYLE["surface"],
    "axes.edgecolor":    STYLE["border"],
    "axes.labelcolor":   STYLE["text"],
    "xtick.color":       STYLE["muted"],
    "ytick.color":       STYLE["muted"],
    "text.color":        STYLE["text"],
    "grid.color":        STYLE["border"],
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.titlepad":     12,
    "axes.labelsize":    11,
})


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


# ── Chart functions ───────────────────────────────────────────────────────────

def chart_cohort_donut(seg: pd.DataFrame) -> str:
    counts = seg["risk_cohort"].value_counts().reindex(COHORT_ORDER).fillna(0)
    labels = [COHORT_LABELS[c] for c in COHORT_ORDER]
    colors = [COHORT_COLORS[c] for c in COHORT_ORDER]

    fig, ax = plt.subplots(figsize=(6.5, 5.2), facecolor=STYLE["bg"])
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.78,
        wedgeprops=dict(width=0.52, edgecolor=STYLE["bg"], linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color(STYLE["bg"])
        at.set_fontweight("bold")

    ax.legend(
        handles=[mpatches.Patch(color=colors[i], label=f"{labels[i]}  ({int(counts.iloc[i]):,})")
                 for i in range(len(COHORT_ORDER))],
        loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2,
        fontsize=8.5, frameon=False, labelcolor=STYLE["text"],
    )
    total = int(counts.sum())
    ax.text(0, 0, f"{total:,}\napplicants", ha="center", va="center",
            fontsize=11, color=STYLE["text"], fontweight="bold", linespacing=1.5)
    ax.set_title("Risk Cohort Distribution", color=STYLE["text"], fontweight="bold")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_score_histogram(seg: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=STYLE["bg"])
    bins = range(300, 905, 20)
    ax.hist(seg["latest_score"], bins=bins, color=STYLE["accent"],
            edgecolor=STYLE["bg"], linewidth=0.4, alpha=0.85)
    for x, label, col in [(600, "600", "#f59e0b"), (650, "650", "#ef4444"), (750, "750", "#22c55e")]:
        ax.axvline(x, color=col, linewidth=1.4, linestyle="--", alpha=0.8)
        ax.text(x + 4, ax.get_ylim()[1] * 0.88, label, color=col, fontsize=9)
    ax.set_xlabel("CIBIL Score")
    ax.set_ylabel("Applicant Count")
    ax.set_title("Credit Score Distribution", color=STYLE["text"], fontweight="bold")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_blind_spot_by_cohort(seg: pd.DataFrame) -> str:
    grp = seg.groupby("risk_cohort")["blind_spot_flag"].agg(["sum", "count"]).reindex(COHORT_ORDER).fillna(0)
    grp["pct"] = grp["sum"] / grp["count"] * 100

    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=STYLE["bg"])
    x = np.arange(len(COHORT_ORDER))
    bars = ax.bar(x, grp["pct"], color=[COHORT_COLORS[c] for c in COHORT_ORDER],
                  edgecolor=STYLE["bg"], linewidth=0.8, width=0.6)
    for bar, val in zip(bars, grp["pct"]):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9,
                    color=STYLE["text"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([COHORT_LABELS[c].split("(")[0].strip() for c in COHORT_ORDER],
                       rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Blind-Spot Applicants (%)")
    ax.set_title("Policy Blind-Spot Rate by Cohort", color=STYLE["text"], fontweight="bold")
    ax.set_ylim(0, grp["pct"].max() * 1.22)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_activation_boxplot(seg: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=STYLE["bg"])
    data_by_cohort = [
        seg.loc[seg["risk_cohort"] == c, "activation_score"].values
        for c in COHORT_ORDER
    ]
    bp = ax.boxplot(
        data_by_cohort, patch_artist=True, notch=False,
        medianprops=dict(color=STYLE["bg"], linewidth=2),
        whiskerprops=dict(color=STYLE["muted"]),
        capprops=dict(color=STYLE["muted"]),
        flierprops=dict(marker=".", color=STYLE["muted"], markersize=3, alpha=0.4),
    )
    for patch, cohort in zip(bp["boxes"], COHORT_ORDER):
        patch.set_facecolor(COHORT_COLORS[cohort])
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(COHORT_ORDER) + 1))
    ax.set_xticklabels([COHORT_LABELS[c].split("(")[0].strip() for c in COHORT_ORDER],
                       rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("Activation Score (0–1)")
    ax.set_title("Activation Score Distribution by Cohort", color=STYLE["text"], fontweight="bold")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_score_delta(seg: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=STYLE["bg"])
    pos = seg["score_delta"][seg["score_delta"] >= 0]
    neg = seg["score_delta"][seg["score_delta"] < 0]
    ax.hist(neg, bins=50, color="#ef4444", alpha=0.75, label="Declining", edgecolor=STYLE["bg"])
    ax.hist(pos, bins=50, color="#22c55e", alpha=0.75, label="Improving", edgecolor=STYLE["bg"])
    ax.axvline(0, color=STYLE["text"], linewidth=1.2, linestyle="-")
    ax.set_xlabel("Score Delta (latest − earliest)")
    ax.set_ylabel("Applicant Count")
    ax.set_title("Score Trajectory Distribution", color=STYLE["text"], fontweight="bold")
    ax.legend(frameon=False, labelcolor=STYLE["text"])
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_pull_count(seg: pd.DataFrame) -> str:
    vc = seg["pull_count"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=STYLE["bg"])
    ax.bar(vc.index, vc.values, color=STYLE["accent"], edgecolor=STYLE["bg"],
           linewidth=0.6, width=0.7)
    ax.set_xlabel("Bureau Pull Count")
    ax.set_ylabel("Applicant Count")
    ax.set_title("Pull Count Distribution per Applicant", color=STYLE["text"], fontweight="bold")
    ax.set_xticks(sorted(vc.index))
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_lender_diversity(seg: pd.DataFrame) -> str:
    vc = seg["lender_diversity"].value_counts().sort_index()
    colors = [STYLE["accent"] if v == 1 else "#f59e0b" if v < 3 else "#22c55e" for v in vc.index]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), facecolor=STYLE["bg"])
    ax.bar(vc.index, vc.values, color=colors, edgecolor=STYLE["bg"], linewidth=0.6, width=0.7)
    ax.set_xlabel("Distinct Lenders Who Pulled")
    ax.set_ylabel("Applicant Count")
    ax.set_title("Lender Diversity Distribution", color=STYLE["text"], fontweight="bold")
    ax.set_xticks(sorted(vc.index))
    legend_patches = [
        mpatches.Patch(color=STYLE["accent"], label="1 lender"),
        mpatches.Patch(color="#f59e0b", label="2 lenders"),
        mpatches.Patch(color="#22c55e", label="3+ lenders (multi-lender)"),
    ]
    ax.legend(handles=legend_patches, frameon=False, labelcolor=STYLE["text"], fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_score_vs_buildup(seg: pd.DataFrame) -> str:
    """Scatter: credit score vs lender diversity, colored by cohort."""
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=STYLE["bg"])
    for cohort in COHORT_ORDER:
        sub = seg[seg["risk_cohort"] == cohort]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(sub))
        ax.scatter(sub["latest_score"], sub["lender_diversity"] + jitter,
                   color=COHORT_COLORS[cohort], alpha=0.35, s=7,
                   label=COHORT_LABELS[cohort])
    ax.set_xlabel("Latest CIBIL Score")
    ax.set_ylabel("Lender Diversity (jittered)")
    ax.set_title("Score vs Lender Diversity", color=STYLE["text"], fontweight="bold")
    ax.legend(frameon=False, labelcolor=STYLE["text"], fontsize=8.5,
              markerscale=2, loc="upper left")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_blind_spot_activation(seg: pd.DataFrame) -> str:
    """Activation score: blind-spot vs non blind-spot applicants."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=STYLE["bg"])
    groups = {
        "Blind-Spot Flagged": seg.loc[seg["blind_spot_flag"] == 1, "activation_score"],
        "Standard Portfolio": seg.loc[seg["blind_spot_flag"] == 0, "activation_score"],
    }
    colors_list = ["#f59e0b", STYLE["muted"]]
    for i, (label, vals) in enumerate(groups.items()):
        ax.hist(vals, bins=40, color=colors_list[i], alpha=0.75,
                label=f"{label} (n={len(vals):,})", edgecolor=STYLE["bg"])
    ax.set_xlabel("Activation Score")
    ax.set_ylabel("Applicant Count")
    ax.set_title("Activation Score: Blind-Spot vs Standard", color=STYLE["text"], fontweight="bold")
    ax.legend(frameon=False, labelcolor=STYLE["text"], fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_utilization_by_cohort(seg: pd.DataFrame) -> str:
    """Average utilization boxplot by cohort."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=STYLE["bg"])
    data_by_cohort = [
        seg.loc[seg["risk_cohort"] == c, "avg_utilization"].clip(0, 1).values
        for c in COHORT_ORDER
    ]
    bp = ax.boxplot(
        data_by_cohort, patch_artist=True, notch=False,
        medianprops=dict(color=STYLE["bg"], linewidth=2),
        whiskerprops=dict(color=STYLE["muted"]),
        capprops=dict(color=STYLE["muted"]),
        flierprops=dict(marker=".", color=STYLE["muted"], markersize=3, alpha=0.4),
    )
    for patch, cohort in zip(bp["boxes"], COHORT_ORDER):
        patch.set_facecolor(COHORT_COLORS[cohort])
        patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(COHORT_ORDER) + 1))
    ax.set_xticklabels([COHORT_LABELS[c].split("(")[0].strip() for c in COHORT_ORDER],
                       rotation=18, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_ylabel("Avg Credit Utilization")
    ax.set_title("Credit Utilization by Cohort", color=STYLE["text"], fontweight="bold")
    fig.tight_layout()
    return fig_to_b64(fig)


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_metrics_html(seg: pd.DataFrame) -> str:
    total = len(seg)
    blind = seg["blind_spot_flag"].sum()
    cohort_counts = seg["risk_cohort"].value_counts()

    prime = cohort_counts.get("A_PrimeStable", 0) + cohort_counts.get("B_PrimeActive", 0)
    near  = cohort_counts.get("C_NearPrime", 0)
    sub   = cohort_counts.get("D_Subprime", 0)
    thin  = cohort_counts.get("E_ThinFile", 0)
    avg_score = int(seg["latest_score"].mean())

    cards = [
        ("Total Applicants",     f"{total:,}",              "#38bdf8", "Deduplicated applicant master"),
        ("Policy Blind Spots",   f"{blind:,} ({blind/total:.1%})", "#f59e0b", "Flagged for re-evaluation"),
        ("Prime Borrowers",      f"{prime:,} ({prime/total:.1%})", "#22c55e", "Cohorts A + B"),
        ("Thin File",            f"{thin:,} ({thin/total:.1%})",  "#8b5cf6", "Avg tradelines < 3"),
        ("Subprime",             f"{sub:,} ({sub/total:.1%})",   "#ef4444",  "Score < 600"),
        ("Avg CIBIL Score",      f"{avg_score}",            "#94a3b8", "Across all applicants"),
    ]

    html = '<div class="metric-grid">'
    for title, value, color, subtitle in cards:
        html += f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color}">{value}</div>
          <div class="metric-title">{title}</div>
          <div class="metric-sub">{subtitle}</div>
        </div>"""
    html += "</div>"
    return html


def build_blind_spot_table(seg: pd.DataFrame) -> str:
    top = (
        seg[seg["blind_spot_flag"] == 1]
        .sort_values("activation_score", ascending=False)
        .head(20)[["applicant_id", "risk_cohort", "latest_score", "score_delta",
                    "lender_diversity", "pull_count", "activation_score"]]
    )
    top["risk_cohort"] = top["risk_cohort"].map(lambda c: COHORT_LABELS.get(c, c))
    top["activation_score"] = top["activation_score"].map(lambda x: f"{x:.3f}")
    top["applicant_id"] = top["applicant_id"].str[:12] + "…"

    rows = ""
    for _, r in top.iterrows():
        rows += f"""<tr>
          <td>{r['applicant_id']}</td>
          <td>{r['risk_cohort']}</td>
          <td>{r['latest_score']}</td>
          <td>{r['score_delta']:+d}</td>
          <td>{r['lender_diversity']}</td>
          <td>{r['pull_count']}</td>
          <td><strong>{r['activation_score']}</strong></td>
        </tr>"""

    return f"""
    <div class="section">
      <h2>Top 20 Blind-Spot Candidates (by Activation Score)</h2>
      <p class="muted">Applicants flagged as policy blind spots, sorted by activation likelihood.
         These borrowers fail standard underwriting but exhibit positive signals (improving score,
         multi-lender demand, or thin file with active credit-seeking).</p>
      <table>
        <thead><tr>
          <th>Applicant ID</th><th>Cohort</th><th>Score</th>
          <th>Score Δ</th><th>Lenders</th><th>Pulls</th><th>Activation</th>
        </tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>"""


def img_section(title: str, description: str, b64: str, wide: bool = False) -> str:
    cls = "chart-wide" if wide else "chart"
    return f"""
    <div class="section {cls}">
      <h2>{title}</h2>
      <p class="muted">{description}</p>
      <img src="{b64}" alt="{title}" />
    </div>"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Credit Bureau — Deduplication & Risk Segmentation Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0f172a;
    color: #f1f5f9;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    font-size: 15px;
    line-height: 1.6;
  }}
  header {{
    background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
    border-bottom: 1px solid #334155;
    padding: 36px 48px 28px;
  }}
  header h1 {{
    font-size: 1.9rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.3px;
  }}
  header .sub {{
    color: #94a3b8;
    margin-top: 6px;
    font-size: 0.95rem;
  }}
  header .badge {{
    display: inline-block;
    margin-top: 14px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
  }}
  main {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 40px 32px 80px;
  }}
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 16px;
    margin-bottom: 48px;
  }}
  .metric-card {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 20px 18px;
    text-align: center;
  }}
  .metric-value {{
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 4px;
  }}
  .metric-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 3px;
  }}
  .metric-sub {{
    font-size: 0.75rem;
    color: #64748b;
  }}
  .chart-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }}
  @media (max-width: 860px) {{
    .chart-grid {{ grid-template-columns: 1fr; }}
  }}
  .section {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 28px 28px 22px;
  }}
  .section h2 {{
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: #e2e8f0;
  }}
  .muted {{
    color: #94a3b8;
    font-size: 0.85rem;
    margin-bottom: 18px;
  }}
  .section img {{
    width: 100%;
    height: auto;
    border-radius: 6px;
    display: block;
  }}
  .section.chart-wide {{
    grid-column: 1 / -1;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.84rem;
    margin-top: 4px;
  }}
  thead tr {{
    background: #0f172a;
    border-bottom: 2px solid #334155;
  }}
  th {{
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    color: #94a3b8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  td {{
    padding: 9px 12px;
    border-bottom: 1px solid #1e293b;
    color: #f1f5f9;
  }}
  tr:nth-child(even) td {{ background: #1a2436; }}
  tr:hover td {{ background: #253347; }}
  footer {{
    text-align: center;
    color: #475569;
    font-size: 0.8rem;
    padding-bottom: 32px;
  }}
  .methodology {{
    background: #1e293b;
    border: 1px solid #334155;
    border-left: 4px solid #38bdf8;
    border-radius: 0 10px 10px 0;
    padding: 22px 26px;
    margin-bottom: 48px;
  }}
  .methodology h2 {{ font-size: 1rem; font-weight: 700; margin-bottom: 10px; color: #38bdf8; }}
  .methodology p {{ color: #94a3b8; font-size: 0.88rem; margin-bottom: 8px; }}
  .methodology code {{
    background: #0f172a;
    border-radius: 4px;
    padding: 1px 6px;
    font-family: monospace;
    font-size: 0.82rem;
    color: #7dd3fc;
  }}
</style>
</head>
<body>

<header>
  <h1>Credit Bureau — Deduplication & Risk Segmentation</h1>
  <div class="sub">End-to-end pipeline on 50,000 simulated CIBIL/Experian bureau pull records</div>
  <span class="badge" style="background:#0c4a6e;color:#38bdf8">Python · SQL</span>
  <span class="badge" style="background:#1a2e05;color:#86efac">ClickHouse</span>
  <span class="badge" style="background:#2d1b69;color:#c4b5fd">Fuzzy Matching</span>
  <span class="badge" style="background:#451a03;color:#fbbf24">Risk Segmentation</span>
</header>

<main>

{methodology}

{metrics}

<div class="chart-grid">
{donut}
{score_hist}
{blind_by_cohort}
{score_delta}
{activation_box}
{pull_count}
{lender_div}
{score_scatter}
{blind_activation}
{utilization}
</div>

{table}

</main>
<footer>
  Generated by scripts/07_generate_report.py &nbsp;·&nbsp;
  50,000 bureau pulls · 17,000+ applicants · 12 features · 5 risk cohorts
</footer>
</body>
</html>"""

METHODOLOGY = """
<div class="methodology">
  <h2>Pipeline Methodology</h2>
  <p><strong>Deduplication:</strong> Each applicant may appear 1–8 times across lenders.
     Records are normalised (OCR-fix on PAN, strip +91/0 on mobile, multi-format DOB parse),
     blocked by PAN prefix + DOB year + mobile tail-4, then scored:
     <code>0.55 × JaroWinkler(PAN) + 0.25 × mobile_exact + 0.20 × DOB_sim</code>.
     Pairs ≥ 0.80 are merged via Union-Find transitive closure.
     Result: 100% precision, 97.7% recall, F1 = 98.8%.</p>
  <p><strong>Segmentation:</strong> Five cohorts assigned by rule priority —
     Thin File (avg tradelines &lt; 3) overrides all score bands, then
     Prime Stable (≥750), Prime Active (700+, multi-lender), Near Prime (600–699), Subprime (&lt;600).
     Activation score = weighted composite of score recency, score improvement, lender demand, and pull frequency.</p>
  <p><strong>Blind-Spot Flags:</strong> Three override conditions surface applicants who fail standard
     underwriting but show positive signals — improving thin files, multi-lender subprime recovery,
     and near-prime multi-lender demand. About <strong>17–18%</strong> of the portfolio is flagged.</p>
</div>
"""


def main():
    print("Loading data …")
    seg = pd.read_csv(SEGMENTS_CSV)
    print(f"  {len(seg):,} applicants loaded from segments.csv")

    print("Rendering charts …")
    donut         = img_section("Risk Cohort Distribution",
                                "Each applicant assigned to exactly one of five cohorts. Thin File overrides all score-based assignments.",
                                chart_cohort_donut(seg))
    score_hist    = img_section("Credit Score Distribution",
                                "Distribution of latest CIBIL scores across the deduplicated applicant base. Dashed lines mark key underwriting thresholds.",
                                chart_score_histogram(seg))
    blind_cohort  = img_section("Policy Blind-Spot Rate by Cohort",
                                "Percentage of each cohort flagged as a policy blind spot. Near Prime and Thin File show the highest uplift opportunity.",
                                chart_blind_spot_by_cohort(seg))
    score_delta   = img_section("Score Trajectory Distribution",
                                "Score delta = latest score minus earliest score for each applicant. Green bars = improving; red = declining.",
                                chart_score_delta(seg))
    act_box       = img_section("Activation Score by Cohort",
                                "Composite activation likelihood (0–1) by cohort. Higher = more likely to convert if contacted by a lender.",
                                chart_activation_boxplot(seg))
    pull_count    = img_section("Pull Count Distribution",
                                "Number of bureau pulls per deduplicated applicant. Loan-shopping clusters typically show 3–5 pulls.",
                                chart_pull_count(seg))
    lender_div    = img_section("Lender Diversity Distribution",
                                "Distinct lenders who pulled the bureau for each applicant. ≥3 lenders = strong activation signal.",
                                chart_lender_diversity(seg))
    scatter       = img_section("Score vs Lender Diversity",
                                "Each dot is one applicant, colored by cohort. Applicants in the bottom-right are high-demand but low-score.",
                                chart_score_vs_buildup(seg), wide=True)
    blind_act     = img_section("Activation Score: Blind-Spot vs Standard",
                                "Blind-spot flagged applicants tend to cluster at higher activation scores than the broader portfolio.",
                                chart_blind_spot_activation(seg))
    utilization   = img_section("Credit Utilization by Cohort",
                                "Average credit utilization (credit used / limit). Subprime applicants show markedly higher utilization.",
                                chart_utilization_by_cohort(seg))

    print("Building HTML …")
    html = HTML_TEMPLATE.format(
        methodology=METHODOLOGY,
        metrics=build_metrics_html(seg),
        donut=donut,
        score_hist=score_hist,
        blind_by_cohort=blind_cohort,
        score_delta=score_delta,
        activation_box=act_box,
        pull_count=pull_count,
        lender_div=lender_div,
        score_scatter=scatter,
        blind_activation=blind_act,
        utilization=utilization,
        table=build_blind_spot_table(seg),
    )

    REPORT_PATH.write_text(html, encoding="utf-8")
    size_kb = REPORT_PATH.stat().st_size // 1024
    print(f"\n✓ Report written → {REPORT_PATH}  ({size_kb:,} KB)")
    print("  Open in any browser — no server or login needed.")


if __name__ == "__main__":
    main()
