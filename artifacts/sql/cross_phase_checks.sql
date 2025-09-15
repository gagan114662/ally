-- Cross-Phase Integrity Checks for Ally Research Pipeline
-- Ensures no strategy bypasses Phase 5.x validation gates

-- One-time helpers (run once)
CREATE OR REPLACE VIEW v_latest AS
SELECT *
FROM (
  SELECT r.*,
         ROW_NUMBER() OVER (PARTITION BY tool, params_hash ORDER BY ts DESC) AS rn
  FROM receipts r
) WHERE rn = 1;

CREATE OR REPLACE VIEW v_det AS
SELECT tool, params_hash, COUNT(DISTINCT receipt_hash) AS distinct_hashes, COUNT(*) AS runs
FROM receipts GROUP BY 1,2;

-- A) Evo → 5.x gate linkage (no orphan survivors)

-- A1. Any evo generations without a corresponding walkforward run?
SELECT COUNT(*) AS evo_without_wf
FROM receipts r1
WHERE r1.tool = 'evo.search'
  AND NOT EXISTS (
    SELECT 1 FROM receipts r2
    WHERE r2.tool = 'research.walkforward.run'
      AND r2.extra->>'strategy_hash' = r1.extra->>'strategy_hash'
  );

-- A2. Survivors must pass 5.1–5.3 (walkforward, costs, robustness)
SELECT
  SUM(CASE WHEN wf_ok=0 THEN 1 ELSE 0 END) AS no_wf,
  SUM(CASE WHEN costs_ok=0 THEN 1 ELSE 0 END) AS no_costs,
  SUM(CASE WHEN robust_ok=0 THEN 1 ELSE 0 END) AS no_robust
FROM (
  SELECT
    r1.extra->>'strategy_hash' AS sh,
    EXISTS (SELECT 1 FROM receipts w
            WHERE w.tool='research.walkforward.run'
              AND w.extra->>'strategy_hash'=r1.extra->>'strategy_hash'
              AND (w.extra->>'wf_pass')='true') AS wf_ok,
    EXISTS (SELECT 1 FROM receipts c
            WHERE c.tool='research.costs.apply'
              AND c.extra->>'strategy_hash'=r1.extra->>'strategy_hash'
              AND (c.extra->>'constraints_passed')='true') AS costs_ok,
    EXISTS (SELECT 1 FROM receipts b
            WHERE b.tool='research.robustness.battery'
              AND b.extra->>'strategy_hash'=r1.extra->>'strategy_hash'
              AND CAST(b.extra->>'pass_rate' AS DOUBLE) >=
                  CAST(b.extra->>'threshold' AS DOUBLE)) AS robust_ok
  FROM receipts r1
  WHERE r1.tool='evo.search'
) q;

-- B) FDR + Novelty enforcement for evo survivors

-- B1. Evo survivors missing FDR pass?
SELECT COUNT(*) AS evo_without_fdr
FROM receipts r1
WHERE r1.tool='evo.search'
  AND NOT EXISTS (
    SELECT 1 FROM receipts f
    WHERE f.tool='research.fdr'
      AND f.extra->>'strategy_hash'=r1.extra->>'strategy_hash'
      AND (f.extra->>'fdr_pass')='true'
  );

-- B2. Duplicates that slipped past novelty filter (corr>0.9 or same simhash bucket)
-- (Assumes novelty receipt stores 'dup'='true' for drops; survivors should have dup='false')
SELECT COUNT(*) AS novelty_violations
FROM receipts n
WHERE n.tool='research.novelty.filter'
  AND (n.extra->>'dup')='true'
  AND EXISTS (
    SELECT 1 FROM receipts e
    WHERE e.tool='evo.search'
      AND e.extra->>'strategy_hash'=n.extra->>'strategy_hash'
  );

-- C) Promotion bundle is truly "bundle v2" complete

-- C1. Every approved promotion shows all gates true and bundle SHA present
SELECT
  COUNT(*) AS approved,
  SUM((extra->>'wf_pass')='true')       AS wf,
  SUM((extra->>'costs_pass')='true')    AS costs,
  SUM((extra->>'robust_pass')='true')   AS robust,
  SUM((extra->>'fdr_pass')='true')      AS fdr,
  SUM((extra->>'novelty_pass')='true')  AS novelty,
  SUM((extra->>'causal_pass')='true')   AS causal,
  SUM((extra->>'ops_ok')='true')        AS ops_ok,
  SUM((extra->>'bundle_sha1') IS NOT NULL) AS has_bundle_sha1
FROM v_latest
WHERE tool='promotion.bundle' AND (extra->>'approved')='true';

-- D) Meta-learner only funds gated strategies

-- D1. Any allocations going to strategies that failed gates?
WITH allocs AS (
  SELECT ts, (json_each(extra->'allocations')).*  -- key=strategy_hash, value=budget
  FROM receipts WHERE tool='meta.learner'
),
bad AS (
  SELECT a.key AS strategy_hash
  FROM allocs a
  WHERE NOT EXISTS (
    SELECT 1 FROM v_latest p
    WHERE p.tool='promotion.bundle'
      AND p.extra->>'strategy_hash'=a.key
      AND (p.extra->>'approved')='true'
  )
)
SELECT COUNT(*) AS ungated_allocations FROM bad;

-- E) Portfolio & Ensemble use only approved components

-- E1. Portfolio constituents must be approved in latest promotion
WITH parts AS (
  SELECT (json_each(extra->'components')).*  -- key=strategy_hash, value=weight
  FROM v_latest WHERE tool='ensemble.combine'
)
SELECT COUNT(*) AS unapproved_components
FROM parts p
WHERE NOT EXISTS (
  SELECT 1 FROM v_latest b
  WHERE b.tool='promotion.bundle'
    AND b.extra->>'strategy_hash'=p.key
    AND (b.extra->>'approved')='true'
);

-- F) Ops gates before live (drift blocks)

-- F1. Any live promotions when latest drift status is not OK?
SELECT COUNT(*) AS live_blocks_should_trigger
FROM v_latest prom
JOIN v_latest drift
  ON drift.tool='ops.drift'
WHERE prom.tool='promotion.bundle'
  AND (prom.extra->>'approved')='true'
  AND (prom.extra->>'mode')='live'
  AND drift.extra->>'status' <> 'OK';

-- G) Determinism & orphans

-- G1. Global determinism breaches (same params, different hashes)
SELECT tool, COUNT(*) AS affected_param_sets
FROM v_det WHERE distinct_hashes > 1
GROUP BY 1 ORDER BY affected_param_sets DESC;

-- G2. Orphan receipts: runs that don't flow into any bundle (useful for cleanup/triage)
SELECT COUNT(*) AS orphans
FROM v_latest r
WHERE r.tool LIKE 'research.%' OR r.tool LIKE 'evo.%'
  AND NOT EXISTS (
    SELECT 1 FROM v_latest b
    WHERE b.tool='promotion.bundle'
      AND (b.extra->>'strategy_hash') = (r.extra->>'strategy_hash')
  );

-- H) Time coherence & recency (Toronto time)

-- H1. Ensure walkforward/robustness timestamps precede promotion
SELECT COUNT(*) AS time_inversions
FROM receipts p
JOIN receipts w
  ON p.tool='promotion.bundle'
 AND w.tool='research.walkforward.run'
 AND p.extra->>'strategy_hash'=w.extra->>'strategy_hash'
WHERE p.ts < w.ts;

-- H2. Today's activity heartbeat
SELECT tool, COUNT(*) AS today_runs
FROM receipts
WHERE DATE(ts AT TIME ZONE 'America/Toronto') = CURRENT_DATE
GROUP BY 1 ORDER BY today_runs DESC;

-- Critical Checks Summary (for CI automation)
-- Run this query and fail CI if any counts > 0

SELECT
  'A1_evo_without_wf' AS check_name,
  (SELECT COUNT(*) FROM receipts r1
   WHERE r1.tool = 'evo.search'
     AND NOT EXISTS (
       SELECT 1 FROM receipts r2
       WHERE r2.tool = 'research.walkforward.run'
         AND r2.extra->>'strategy_hash' = r1.extra->>'strategy_hash'
     )) AS violation_count
UNION ALL
SELECT
  'B1_evo_without_fdr' AS check_name,
  (SELECT COUNT(*) FROM receipts r1
   WHERE r1.tool='evo.search'
     AND NOT EXISTS (
       SELECT 1 FROM receipts f
       WHERE f.tool='research.fdr'
         AND f.extra->>'strategy_hash'=r1.extra->>'strategy_hash'
         AND (f.extra->>'fdr_pass')='true'
     )) AS violation_count
UNION ALL
SELECT
  'D1_ungated_allocations' AS check_name,
  (WITH allocs AS (
     SELECT ts, (json_each(extra->'allocations')).*
     FROM receipts WHERE tool='meta.learner'
   ),
   bad AS (
     SELECT a.key AS strategy_hash
     FROM allocs a
     WHERE NOT EXISTS (
       SELECT 1 FROM v_latest p
       WHERE p.tool='promotion.bundle'
         AND p.extra->>'strategy_hash'=a.key
         AND (p.extra->>'approved')='true'
     )
   )
   SELECT COUNT(*) FROM bad) AS violation_count
UNION ALL
SELECT
  'F1_live_blocks_should_trigger' AS check_name,
  (SELECT COUNT(*) FROM v_latest prom
   JOIN v_latest drift ON drift.tool='ops.drift'
   WHERE prom.tool='promotion.bundle'
     AND (prom.extra->>'approved')='true'
     AND (prom.extra->>'mode')='live'
     AND drift.extra->>'status' <> 'OK') AS violation_count
UNION ALL
SELECT
  'G1_determinism_breaches' AS check_name,
  (SELECT COUNT(*) FROM v_det WHERE distinct_hashes > 1) AS violation_count
UNION ALL
SELECT
  'H1_time_inversions' AS check_name,
  (SELECT COUNT(*) FROM receipts p
   JOIN receipts w
     ON p.tool='promotion.bundle'
    AND w.tool='research.walkforward.run'
    AND p.extra->>'strategy_hash'=w.extra->>'strategy_hash'
   WHERE p.ts < w.ts) AS violation_count;

-- End of Cross-Phase Integrity Checks