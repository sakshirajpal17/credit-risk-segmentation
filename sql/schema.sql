-- ============================================================
-- Credit Bureau Analytics — ClickHouse Schema
-- Database: credit_bureau
-- Designed for: analytical read patterns, monthly partitioning,
--               fast cohort queries, and bureau dedup audits
-- ============================================================

CREATE DATABASE IF NOT EXISTS credit_bureau;

-- ----------------------------------------------------------
-- 1. Raw bureau pull records
--    One row per pull event. Partitioned by month.
--    Retention: 7 years (RBI mandate).
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS credit_bureau.bureau_pulls
(
    pull_id               String         COMMENT 'UUID for this pull event',
    pull_date             Date           COMMENT 'Date the bureau was queried',
    lender                LowCardinality(String) COMMENT 'Lender who initiated the pull',
    bureau                LowCardinality(String) COMMENT 'CIBIL | Experian | Equifax | CRIF',
    loan_type             LowCardinality(String) COMMENT 'personal_loan | home_loan | credit_card …',
    pan                   String         COMMENT 'Raw PAN as received (may have entry errors)',
    mobile                String         COMMENT 'Raw mobile (may have prefix/format variation)',
    dob                   String         COMMENT 'Raw DOB string (DDMMYYYY, DD/MM/YYYY, ISO …)',
    name                  String         COMMENT 'Applicant name as received',
    credit_score          Int16          COMMENT 'CIBIL score 300–900',
    num_tradelines        Int16          COMMENT 'Open credit accounts at time of pull',
    num_enquiries         Int16          COMMENT 'Bureau-reported enquiry count',
    num_delinquencies     Int16          COMMENT 'Past-due accounts',
    credit_limit          Int64          COMMENT 'Total sanctioned credit (INR)',
    credit_used           Int64          COMMENT 'Current outstanding balance (INR)',
    oldest_account_months Int16          COMMENT 'Age of oldest account in months',
    ingested_at           DateTime DEFAULT now()
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(pull_date)
ORDER BY (pan, pull_date)
TTL pull_date + INTERVAL 7 YEAR
SETTINGS index_granularity = 8192;


-- ----------------------------------------------------------
-- 2. Deduplicated applicant master
--    One row per unique applicant (after pipeline dedup).
--    Canonical identifiers chosen by most-frequent value.
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS credit_bureau.applicants
(
    applicant_id     String  COMMENT 'UUID assigned by dedup pipeline',
    canonical_pan    String  COMMENT 'Most-frequent normalised PAN across pulls',
    canonical_mobile String  COMMENT 'Most-frequent normalised mobile',
    canonical_name   String  COMMENT 'Most-frequent name across pulls',
    pull_count       Int16   COMMENT 'Total pulls collapsed into this applicant',
    first_pull_date  String  COMMENT 'Earliest pull date (ISO)',
    last_pull_date   String  COMMENT 'Most recent pull date (ISO)',
    lender_count     Int16   COMMENT 'Distinct lenders who pulled',
    bureau_count     Int16   COMMENT 'Distinct bureaus queried'
)
ENGINE = MergeTree()
ORDER BY applicant_id;


-- ----------------------------------------------------------
-- 3. Pull ↔ Applicant mapping (audit trail)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS credit_bureau.pull_applicant_map
(
    pull_id          String   COMMENT 'Foreign key to bureau_pulls.pull_id',
    applicant_id     String   COMMENT 'Foreign key to applicants.applicant_id',
    match_confidence Float32  COMMENT 'Composite similarity score that triggered match'
)
ENGINE = MergeTree()
ORDER BY (applicant_id, pull_id);


-- ----------------------------------------------------------
-- 4. Engineered features (12 core + supporting context)
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS credit_bureau.applicant_features
(
    applicant_id          String,
    -- Core 12 features
    latest_score          Int16,
    earliest_score        Int16,
    score_delta           Int16    COMMENT 'latest_score − earliest_score; negative = deteriorating',
    score_volatility      Float32  COMMENT 'Std dev of score across all pulls',
    pull_count            Int16,
    pull_freq_30d         Int16    COMMENT 'Pulls within 30 days of reference date',
    pull_freq_90d         Int16    COMMENT 'Pulls within 90 days (loan-shopping window)',
    days_since_first_pull Int16,
    days_since_last_pull  Int16,
    lender_diversity      Int16    COMMENT 'Distinct lenders who pulled this applicant',
    multi_lender_flag     UInt8    COMMENT '1 if 3+ distinct lenders (active rate-shopper)',
    thin_file_flag        UInt8    COMMENT '1 if avg tradelines < 3 (new-to-credit)',
    -- Supporting context
    avg_delinquencies     Float32,
    avg_enquiries         Float32,
    avg_utilization       Float32  COMMENT 'Avg credit utilisation ratio (0–1)',
    first_pull_date       String,
    last_pull_date        String
)
ENGINE = MergeTree()
ORDER BY applicant_id;


-- ----------------------------------------------------------
-- 5. Risk segments — final scoring output
-- ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS credit_bureau.segments
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
    risk_cohort       LowCardinality(String) COMMENT 'A_PrimeStable|B_PrimeActive|C_NearPrime|D_Subprime|E_ThinFile',
    activation_score  Float32  COMMENT '0–1 likelihood of responding to a credit offer',
    blind_spot_flag   UInt8    COMMENT '1 if applicant would fail standard policy but shows positive signals'
)
ENGINE = MergeTree()
ORDER BY (risk_cohort, applicant_id);


-- ----------------------------------------------------------
-- Materialised view: monthly pull volume by bureau
-- Refreshes automatically on insert into bureau_pulls.
-- ----------------------------------------------------------
CREATE MATERIALIZED VIEW IF NOT EXISTS credit_bureau.mv_monthly_pull_volume
ENGINE = SummingMergeTree()
ORDER BY (pull_month, bureau, lender)
POPULATE
AS
SELECT
    toYYYYMM(pull_date)    AS pull_month,
    bureau,
    lender,
    count()                AS pull_count,
    avg(credit_score)      AS avg_score,
    countIf(credit_score < 600) AS low_score_count
FROM credit_bureau.bureau_pulls
GROUP BY pull_month, bureau, lender;
