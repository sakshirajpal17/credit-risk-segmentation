-- ============================================================
-- Metabase Dashboard — Credit Bureau Intelligence
-- Connect Metabase → ClickHouse via JDBC driver
-- Each query below maps to one dashboard card.
-- ============================================================


-- Card 1 ── HEADLINE NUMBERS (Single-value cards)
-- Total applicants
SELECT count() FROM credit_bureau.segments;

-- Total raw pulls
SELECT count() FROM credit_bureau.bureau_pulls;

-- Blind-spot candidates
SELECT sum(blind_spot_flag) FROM credit_bureau.segments;

-- Blind-spot %
SELECT round(sum(blind_spot_flag) * 100.0 / count(), 1) AS blind_spot_pct
FROM credit_bureau.segments;

-- Avg activation score
SELECT round(avg(activation_score), 3) FROM credit_bureau.segments;


-- Card 2 ── COHORT PIE / BAR CHART
-- (Metabase: Question → Bar Chart, x=risk_cohort, y=applicants)
SELECT
    multiIf(
        risk_cohort = 'A_PrimeStable',  'A · Prime Stable  (≥750)',
        risk_cohort = 'B_PrimeActive',  'B · Prime Active  (700–749)',
        risk_cohort = 'C_NearPrime',    'C · Near Prime    (600–699)',
        risk_cohort = 'D_Subprime',     'D · Subprime      (<600)',
        'E · Thin File'
    )                                               AS cohort_label,
    count()                                         AS applicants,
    round(count() * 100.0 / sum(count()) OVER (), 1) AS pct
FROM credit_bureau.segments
GROUP BY risk_cohort, cohort_label
ORDER BY applicants DESC;


-- Card 3 ── SCORE DISTRIBUTION HISTOGRAM
-- (Metabase: Bar Chart, x=score_bucket, y=count)
SELECT
    intDiv(latest_score, 50) * 50  AS score_bucket,
    count()                        AS applicants
FROM credit_bureau.segments
GROUP BY score_bucket
ORDER BY score_bucket;


-- Card 4 ── BLIND SPOTS BY COHORT
-- (Metabase: Stacked Bar — cohort × blind_spot_flag)
SELECT
    risk_cohort,
    sum(blind_spot_flag)                              AS blind_spots,
    countIf(blind_spot_flag = 0)                      AS standard,
    round(sum(blind_spot_flag) * 100.0 / count(), 1)  AS blind_spot_pct
FROM credit_bureau.segments
GROUP BY risk_cohort
ORDER BY blind_spots DESC;


-- Card 5 ── MONTHLY PULL VOLUME (Time-series line chart)
-- (Metabase: Line Chart, x=pull_month, y=pulls, series=bureau)
SELECT
    pull_month,
    bureau,
    sum(pull_count) AS pulls
FROM credit_bureau.mv_monthly_pull_volume
GROUP BY pull_month, bureau
ORDER BY pull_month, bureau;


-- Card 6 ── LENDER ACTIVITY LEADERBOARD (Table)
SELECT
    lender,
    sum(pull_count)                AS total_pulls,
    round(avg(avg_score))          AS avg_applicant_score,
    sum(low_score_count)           AS high_risk_pulls,
    round(sum(low_score_count) * 100.0 / sum(pull_count), 1) AS high_risk_pct
FROM credit_bureau.mv_monthly_pull_volume
GROUP BY lender
ORDER BY total_pulls DESC
LIMIT 15;


-- Card 7 ── SCORE TREND WATERFALL
-- (Metabase: Bar Chart — score improvement distribution)
SELECT
    multiIf(
        score_delta >  50, '>+50  Strongly improving',
        score_delta >  10, '+10 to +50  Improving',
        score_delta >= -10, '±10  Stable',
        score_delta >= -50, '-10 to -50  Declining',
        '<-50  Strongly declining'
    )                                               AS trend_band,
    count()                                         AS applicants,
    round(count() * 100.0 / sum(count()) OVER (), 1) AS pct,
    round(avg(activation_score), 3)                 AS avg_activation
FROM credit_bureau.segments
GROUP BY trend_band
ORDER BY applicants DESC;


-- Card 8 ── ACTIVATION SCORE DISTRIBUTION (Box plot / histogram)
-- (Metabase: Bar Chart, x=activation_bucket, y=applicants, filter by cohort)
SELECT
    risk_cohort,
    round(activation_score, 1) AS activation_bucket,
    count()                    AS applicants
FROM credit_bureau.segments
GROUP BY risk_cohort, activation_bucket
ORDER BY risk_cohort, activation_bucket;


-- Card 9 ── HIGH-VALUE BLIND SPOTS (Table — exportable for ops)
SELECT
    s.applicant_id,
    a.canonical_name,
    s.risk_cohort,
    s.latest_score,
    s.score_delta,
    s.lender_diversity,
    s.pull_freq_90d,
    round(s.activation_score, 3) AS activation_score
FROM credit_bureau.segments s
JOIN credit_bureau.applicants a USING (applicant_id)
WHERE s.blind_spot_flag = 1
  AND s.activation_score >= 0.60
ORDER BY s.activation_score DESC
LIMIT 200;


-- Card 10 ── PULL-COUNT DISTRIBUTION (Rate-shopping intensity)
SELECT
    pull_count  AS pulls_per_applicant,
    count()     AS applicants,
    round(count() * 100.0 / sum(count()) OVER (), 1) AS pct
FROM credit_bureau.applicants
GROUP BY pulls_per_applicant
ORDER BY pulls_per_applicant;
