-- ============================================================
-- Credit Bureau Analytics — Key Query Library
-- Run against: ClickHouse database credit_bureau
-- ============================================================

-- ----------------------------------------------------------
-- 1. Cohort breakdown — counts, avg score, blind spots
-- ----------------------------------------------------------
SELECT
    risk_cohort,
    count()                                        AS applicants,
    round(count() * 100.0 / sum(count()) OVER (), 1) AS pct,
    round(avg(latest_score))                       AS avg_score,
    round(avg(activation_score), 3)                AS avg_activation,
    sum(blind_spot_flag)                           AS blind_spots,
    round(sum(blind_spot_flag) * 100.0 / count(), 1) AS blind_spot_pct
FROM credit_bureau.segments
GROUP BY risk_cohort
ORDER BY applicants DESC;


-- ----------------------------------------------------------
-- 2. Blind-spot deep-dive — who are they?
-- ----------------------------------------------------------
SELECT
    risk_cohort,
    count()                         AS blind_spot_count,
    round(avg(latest_score))        AS avg_score,
    round(avg(score_delta))         AS avg_score_delta,
    round(avg(activation_score), 3) AS avg_activation,
    round(avg(lender_diversity), 1) AS avg_lender_div,
    round(avg(pull_freq_90d), 1)    AS avg_pulls_90d
FROM credit_bureau.segments
WHERE blind_spot_flag = 1
GROUP BY risk_cohort
ORDER BY blind_spot_count DESC;


-- ----------------------------------------------------------
-- 3. Score-band distribution (CIBIL ranges)
-- ----------------------------------------------------------
SELECT
    multiIf(
        latest_score >= 750, '750–900  Prime',
        latest_score >= 700, '700–749  Preferred',
        latest_score >= 650, '650–699  Standard',
        latest_score >= 600, '600–649  Sub-Prime',
        '300–599  Deep Sub-Prime'
    )                                                 AS score_band,
    count()                                           AS applicants,
    round(count() * 100.0 / sum(count()) OVER (), 1)  AS pct,
    round(avg(activation_score), 3)                   AS avg_activation
FROM credit_bureau.segments
GROUP BY score_band
ORDER BY score_band;


-- ----------------------------------------------------------
-- 4. Lender diversity histogram
-- ----------------------------------------------------------
SELECT
    lender_diversity,
    count()    AS applicants,
    round(count() * 100.0 / sum(count()) OVER (), 1) AS pct
FROM credit_bureau.segments
GROUP BY lender_diversity
ORDER BY lender_diversity;


-- ----------------------------------------------------------
-- 5. Score trend classification
-- ----------------------------------------------------------
SELECT
    multiIf(
        score_delta >  50, 'Strongly Improving  (>50)',
        score_delta >  10, 'Improving           (10–50)',
        score_delta >= -10, 'Stable              (±10)',
        score_delta >= -50, 'Declining           (-10 to -50)',
        'Strongly Declining  (<-50)'
    )                                               AS trend,
    count()                                         AS applicants,
    round(count() * 100.0 / sum(count()) OVER (), 1) AS pct,
    round(avg(activation_score), 3)                 AS avg_activation
FROM credit_bureau.segments
GROUP BY trend
ORDER BY applicants DESC;


-- ----------------------------------------------------------
-- 6. Monthly pull volume by bureau (from materialised view)
-- ----------------------------------------------------------
SELECT
    pull_month,
    bureau,
    sum(pull_count) AS pulls,
    round(avg(avg_score))  AS avg_score
FROM credit_bureau.mv_monthly_pull_volume
GROUP BY pull_month, bureau
ORDER BY pull_month, bureau;


-- ----------------------------------------------------------
-- 7. Deduplication impact — pulls per applicant
-- ----------------------------------------------------------
SELECT
    pull_count,
    count() AS applicants,
    round(count() * 100.0 / sum(count()) OVER (), 1) AS pct
FROM credit_bureau.applicants
GROUP BY pull_count
ORDER BY pull_count;


-- ----------------------------------------------------------
-- 8. High-value blind-spot candidates
--    (activation ≥ 0.6, blind_spot_flag = 1)
--    These are the applicants to surface to product/credit teams.
-- ----------------------------------------------------------
SELECT
    s.applicant_id,
    a.canonical_pan,
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
  AND s.activation_score >= 0.6
ORDER BY s.activation_score DESC
LIMIT 50;


-- ----------------------------------------------------------
-- 9. Cross-bureau pull audit — applicants queried by 3+ bureaus
--    Signals that the applicant is shopping aggressively.
-- ----------------------------------------------------------
SELECT
    a.applicant_id,
    a.canonical_name,
    a.bureau_count,
    a.lender_count,
    a.pull_count,
    s.latest_score,
    s.risk_cohort,
    round(s.activation_score, 3) AS activation_score
FROM credit_bureau.applicants a
JOIN credit_bureau.segments s USING (applicant_id)
WHERE a.bureau_count >= 3
ORDER BY a.bureau_count DESC, a.lender_count DESC
LIMIT 100;


-- ----------------------------------------------------------
-- 10. Utilisation vs score correlation by cohort
-- ----------------------------------------------------------
SELECT
    s.risk_cohort,
    round(avg(f.avg_utilization), 3)  AS avg_utilization,
    round(avg(s.latest_score))        AS avg_score,
    round(avg(f.avg_delinquencies), 2) AS avg_delinquencies,
    count()                            AS applicants
FROM credit_bureau.segments s
JOIN credit_bureau.applicant_features f USING (applicant_id)
GROUP BY s.risk_cohort
ORDER BY avg_score DESC;
