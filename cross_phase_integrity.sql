-- Cross-Phase Integrity Queries for Phase 8 CI Validation
-- Ensures drift sentinels properly guard the entire pipeline

-- Query 1: Verify drift sentinels block promotions when violations detected
WITH latest_drift AS (
  SELECT tool, extra->>'status' AS status, ts,
         ROW_NUMBER() OVER (PARTITION BY tool ORDER BY ts DESC) as rn
  FROM receipts
  WHERE tool IN ('ops.drift.data', 'ops.drift.strategy', 'ops.drift.ops')
),
latest_guard AS (
  SELECT extra->>'decision' AS decision, ts
  FROM receipts
  WHERE tool = 'ops.guard'
  ORDER BY ts DESC LIMIT 1
)
SELECT
  COUNT(CASE WHEN status != 'OK' THEN 1 END) as drift_violations,
  MAX(decision) as guard_decision,
  CASE
    WHEN COUNT(CASE WHEN status != 'OK' THEN 1 END) > 0 AND MAX(decision) = 'ALLOW'
    THEN 'INTEGRITY_VIOLATION: Guard should block when drift detected'
    ELSE 'OK'
  END as integrity_status
FROM latest_drift, latest_guard
WHERE latest_drift.rn = 1;

-- Query 2: Verify heartbeat reflects actual system health
WITH system_health AS (
  SELECT
    tool,
    extra->>'status' AS status,
    ts
  FROM receipts
  WHERE tool IN ('ops.drift.data', 'ops.drift.strategy', 'ops.drift.ops')
    AND ts >= datetime('now', '-24 hours')
),
heartbeat_health AS (
  SELECT extra->>'health_score' AS score, ts
  FROM receipts
  WHERE tool = 'ops.heartbeat'
  ORDER BY ts DESC LIMIT 1
)
SELECT
  COUNT(*) as total_checks,
  COUNT(CASE WHEN status = 'OK' THEN 1 END) as healthy_checks,
  ROUND(100.0 * COUNT(CASE WHEN status = 'OK' THEN 1 END) / COUNT(*), 1) as computed_health,
  CAST(score AS REAL) as reported_health,
  CASE
    WHEN ABS(ROUND(100.0 * COUNT(CASE WHEN status = 'OK' THEN 1 END) / COUNT(*), 1) - CAST(score AS REAL)) > 10
    THEN 'INTEGRITY_VIOLATION: Heartbeat health score inconsistent with actual checks'
    ELSE 'OK'
  END as integrity_status
FROM system_health, heartbeat_health;

-- Query 3: Verify deterministic drift calculations across runs
SELECT
  tool,
  params_hash,
  COUNT(DISTINCT receipt_hash) as unique_receipts,
  CASE
    WHEN COUNT(DISTINCT receipt_hash) > 1
    THEN 'INTEGRITY_VIOLATION: Non-deterministic results for same parameters'
    ELSE 'OK'
  END as determinism_status
FROM receipts
WHERE tool LIKE 'ops.drift.%'
GROUP BY tool, params_hash
HAVING COUNT(*) > 1;

-- Query 4: Verify pipeline chronology (drift checks before guard decisions)
WITH guard_decisions AS (
  SELECT ts as guard_ts FROM receipts WHERE tool = 'ops.guard' ORDER BY ts DESC LIMIT 1
),
drift_checks AS (
  SELECT tool, MAX(ts) as latest_drift_ts
  FROM receipts
  WHERE tool LIKE 'ops.drift.%'
  GROUP BY tool
)
SELECT
  tool,
  latest_drift_ts,
  guard_ts,
  CASE
    WHEN latest_drift_ts > guard_ts
    THEN 'INTEGRITY_VIOLATION: Guard decision made before latest drift check'
    ELSE 'OK'
  END as chronology_status
FROM drift_checks, guard_decisions;

-- Query 5: Verify all required sentinels are operational
WITH required_sentinels AS (
  SELECT 'ops.drift.data' as tool
  UNION SELECT 'ops.drift.strategy'
  UNION SELECT 'ops.drift.ops'
),
active_sentinels AS (
  SELECT DISTINCT tool
  FROM receipts
  WHERE tool LIKE 'ops.drift.%'
    AND ts >= datetime('now', '-24 hours')
)
SELECT
  r.tool,
  CASE WHEN a.tool IS NOT NULL THEN 'ACTIVE' ELSE 'MISSING' END as status,
  CASE
    WHEN a.tool IS NULL
    THEN 'INTEGRITY_VIOLATION: Required sentinel not operational in last 24h'
    ELSE 'OK'
  END as integrity_status
FROM required_sentinels r
LEFT JOIN active_sentinels a ON r.tool = a.tool;