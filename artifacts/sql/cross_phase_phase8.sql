-- Cross-Phase Integrity Verification for Phase 8
-- Ensures drift sentinels properly gate promotions and maintain pipeline coherence

-- Import receipts from JSONL for testing (in real CI, receipts would be in DuckDB)
CREATE OR REPLACE TABLE receipts AS
SELECT
  tool,
  params_hash,
  receipt_hash,
  ts::TIMESTAMP as ts,
  extra
FROM read_json_auto('artifacts/receipts.jsonl');

-- LATEST helper view
CREATE OR REPLACE VIEW v_latest AS
SELECT *
FROM (
  SELECT r.*,
         ROW_NUMBER() OVER (PARTITION BY tool, params_hash ORDER BY ts DESC) rn
  FROM receipts r
)
WHERE rn = 1;

-- A) Every promotion guard must reference real sentinel runs
--    (i.e., a guard receipt exists only if we have current data/strategy/ops drift receipts)
CREATE OR REPLACE VIEW A_missing_sentinel_refs AS
SELECT 'A_missing_sentinel_refs' AS check_name,
       COUNT(*) AS offenders
FROM v_latest g
WHERE g.tool = 'ops.guard'
  AND (NOT EXISTS (SELECT 1 FROM v_latest d WHERE d.tool='ops.drift.data')
  OR NOT EXISTS (SELECT 1 FROM v_latest s WHERE s.tool='ops.drift.strategy')
  OR NOT EXISTS (SELECT 1 FROM v_latest o WHERE o.tool='ops.drift.ops'));

-- B) Guards must block when any sentinel is non-OK
CREATE OR REPLACE VIEW B_guard_allows_bad AS
WITH sents AS (
  SELECT
    MAX(CASE WHEN tool='ops.drift.data'     THEN (extra->>'status') END) AS data_status,
    MAX(CASE WHEN tool='ops.drift.strategy' THEN (extra->>'status') END) AS strat_status,
    MAX(CASE WHEN tool='ops.drift.ops'      THEN (extra->>'status') END) AS ops_status
  FROM v_latest
  WHERE tool LIKE 'ops.drift.%'
),
guard AS (
  SELECT extra->>'status' AS decision FROM v_latest WHERE tool='ops.guard'
)
SELECT 'B_guard_allows_bad' AS check_name,
       COUNT(*) AS offenders
FROM sents, guard
WHERE (data_status <> 'OK' OR strat_status <> 'OK' OR ops_status <> 'OK')
  AND decision = 'ALLOW';

-- C) Promotions must depend on guard=OK (no bypass)
-- Note: Using 'promotion.bundle' as the promotion tool name
CREATE OR REPLACE VIEW C_promotion_without_guard_ok AS
SELECT 'C_promotion_without_guard_ok' AS check_name,
       COUNT(*) AS offenders
FROM v_latest p
LEFT JOIN v_latest g ON g.tool='ops.guard'
WHERE p.tool IN ('promotion.bundle','promotion.apply')
  AND COALESCE(g.extra->>'status','BLOCK') <> 'ALLOW';

-- D) Determinism: same params must not yield multiple receipt hashes
CREATE OR REPLACE VIEW D_multihash_same_params AS
SELECT 'D_multihash_same_params' AS check_name,
       COUNT(*) AS offenders
FROM (
  SELECT tool, params_hash, COUNT(DISTINCT receipt_hash) AS dh
  FROM receipts
  WHERE tool IN ('ops.drift.data','ops.drift.strategy','ops.drift.ops','ops.guard')
  GROUP BY tool, params_hash
) t
WHERE t.dh > 1;

-- E) Time coherence: guard must be newer than its sentinel inputs
CREATE OR REPLACE VIEW E_guard_staler_than_sentinels AS
WITH times AS (
  SELECT
    (SELECT MAX(ts) FROM receipts WHERE tool='ops.drift.data')     AS t_data,
    (SELECT MAX(ts) FROM receipts WHERE tool='ops.drift.strategy') AS t_strat,
    (SELECT MAX(ts) FROM receipts WHERE tool='ops.drift.ops')      AS t_ops,
    (SELECT MAX(ts) FROM receipts WHERE tool='ops.guard')          AS t_guard
)
SELECT 'E_guard_staler_than_sentinels' AS check_name,
       CASE WHEN t_guard >= t_data AND t_guard >= t_strat AND t_guard >= t_ops THEN 0 ELSE 1 END AS offenders
FROM times;

-- F) Heartbeat today exists when system is healthy
CREATE OR REPLACE VIEW F_missing_heartbeat_when_guard_ok AS
WITH guard AS (
  SELECT extra->>'status' AS decision, ts FROM v_latest WHERE tool='ops.guard'
)
SELECT 'F_missing_heartbeat_when_guard_ok' AS check_name,
       CASE
         WHEN (SELECT decision FROM guard)='ALLOW' AND
              (SELECT COUNT(*) FROM receipts
                 WHERE tool='ops.heartbeat'
                   AND DATE(ts) = CURRENT_DATE) = 0
         THEN 1 ELSE 0
       END AS offenders;

-- Master results view
CREATE OR REPLACE VIEW phase8_integrity_results AS
SELECT check_name, offenders FROM A_missing_sentinel_refs
UNION ALL SELECT check_name, offenders FROM B_guard_allows_bad
UNION ALL SELECT check_name, offenders FROM C_promotion_without_guard_ok
UNION ALL SELECT check_name, offenders FROM D_multihash_same_params
UNION ALL SELECT check_name, offenders FROM E_guard_staler_than_sentinels
UNION ALL SELECT check_name, offenders FROM F_missing_heartbeat_when_guard_ok;