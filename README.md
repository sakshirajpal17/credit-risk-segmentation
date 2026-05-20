# Credit Risk Segmentation

End-to-end data pipeline that deduplicates **50,000 credit bureau pulls** into **17,000 unique applicants**, engineers **12 risk features**, and segments borrowers into **5 cohorts** — surfacing **3,040 high-value applicants** that standard policy would have wrongly declined.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-ClickHouse-FFCC01?logo=clickhouse&logoColor=black)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Metabase](https://img.shields.io/badge/Metabase-509EE3?logo=metabase&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

---

## Headline Results

| Metric | Value |
| :--- | :--- |
| Dedup match precision | **100%** (zero false merges) |
| Dedup link recall | **97.7%** |
| F1 score | **98.8%** |
| Applicants resolved from 50K pulls | **17,547** |
| Policy blind spots surfaced | **3,040** (17.3% of portfolio) |
| Dashboard query latency (ClickHouse) | **sub-second** on 50K rows |

---

## The Problem

A single borrower applying to 5 lenders generates 5 bureau pulls, each with the PAN, mobile, or DOB recorded slightly differently — OCR errors, format variants, digit swaps. Exact-match logic treats them as 5 different people, inflates portfolio size, and hides the real signal: *this person is loan-shopping*.

This pipeline solves that, then turns the cleaned applicant view into actionable risk intelligence.

---

## What's Inside

```
bureau_pulls.csv (50K rows, 4 bureaus, 17K real people)
        │
        ▼
  Deduplication        →  Union-Find over fuzzy-matched PAN/DOB/mobile
                          → 17,547 applicants, 98.8% F1
        │
        ▼
  Feature engineering  →  12 features per applicant
                          (score trends, pull velocity, lender diversity)
        │
        ▼
  Risk segmentation    →  5 cohorts + activation score (0–1)
                          + policy blind-spot flag
        │
        ▼
  ClickHouse + Metabase →  10 dashboard cards, sub-second queries
```

---

## Tech Stack

- **Python 3.11** — pandas, numpy, jellyfish (fuzzy matching), python-dateutil
- **SQL / ClickHouse** — columnar storage, `LowCardinality` types, materialised views
- **Metabase** — interactive dashboard with 10 analytical cards
- **Docker Compose** — one-command local stack

---

## Quick Start

```bash
pip install -r requirements.txt

python scripts/01_generate_data.py        # 50K synthetic bureau pulls
python scripts/02_deduplicate.py          # fuzzy dedup → 17K applicants
python scripts/03_feature_engineering.py  # 12 features per applicant
python scripts/04_risk_segmentation.py    # 5 cohorts + blind-spot flagging
python scripts/06_validate.py             # precision/recall report

# Optional — ClickHouse + Metabase dashboard
docker-compose up -d
python scripts/05_clickhouse_setup.py
# → http://localhost:3000, paste queries from dashboard/metabase_queries.sql
```

---

## Deduplication Approach

**Normalise** PAN (OCR fixes O→0, I→1), mobile (strip `+91`/`0`), DOB (multi-format parse).

**Block** to cut 2.5B comparisons down to ~150K — primary on PAN-prefix-6, fallbacks on mobile-last-4 + birth year.

**Score** each candidate pair:
```
composite = 0.55 × JaroWinkler(PAN)
          + 0.25 × mobile_exact
          + 0.20 × DOB_similarity
```

**Threshold ≥ 0.80** → Union-Find transitive closure → one `applicant_id` per cluster.

> Validated against ground truth in [scripts/06_validate.py](scripts/06_validate.py): **100% precision, 97.7% recall, 98.8% F1**.

---

## Risk Cohorts

| Cohort | Criteria | Applicants | Blind Spots |
| :--- | :--- | ---: | ---: |
| **A — Prime Stable** | Score ≥ 750, ≤3 pulls | 3,818 | — |
| **B — Prime Active** | Score ≥ 700, multi-lender | 946 | — |
| **C — Near Prime** | 600 ≤ score < 700 | 4,341 | 1,218 |
| **D — Subprime** | Score < 600 | 2,846 | 119 |
| **E — Thin File** | Avg tradelines < 3 | 5,596 | 1,703 |

**Blind spot detection** overrides standard decline policy when the bureau data shows a positive signal — improving score, multi-lender demand, or active credit-seeking. **3,040 applicants flagged (17.3%)** — revenue standard policy would have left on the table.

---

## Repository Layout

```
credit-data/
├── config.py                       all paths, constants, scoring weights
├── requirements.txt
├── docker-compose.yml              ClickHouse + Metabase
├── scripts/
│   ├── 01_generate_data.py         50K synthetic bureau pulls
│   ├── 02_deduplicate.py           fuzzy dedup pipeline
│   ├── 03_feature_engineering.py   12 features per applicant
│   ├── 04_risk_segmentation.py     5 cohorts + blind-spot logic
│   ├── 05_clickhouse_setup.py      DDL + bulk load
│   ├── 06_validate.py              precision/recall evaluation
│   └── 07_generate_report.py       HTML executive summary
├── sql/
│   ├── schema.sql                  ClickHouse DDL + materialised view
│   └── analytics.sql               10 analytical queries
├── dashboard/
│   └── metabase_queries.sql        10 Metabase dashboard cards
└── viz/
    └── credit_bureau_report.html   standalone HTML report
```

---

## Data & Privacy

All data is **synthetically generated** — no real consumer information. The simulation follows real-world Indian bureau conventions:

- **PAN format** AAAAA9999A (RBI standard, position 4 = `P` for individuals)
- **Mobile** 10 digits, first digit 6–9
- **DOB primary format** DDMMYYYY (per CIBIL/Experian Uniform Credit Reporting Format v3.73)
- **Score range** 300–900 (TransUnion CIBIL)
- **Bureaus** CIBIL, Experian, Equifax, CRIF High Mark (all 4 RBI-licensed CICs)
- **Thin-file population** ~28% (consistent with India's new-to-credit segment)

---

## Author

**Sakshi Rajpal** — Data engineering portfolio project.
