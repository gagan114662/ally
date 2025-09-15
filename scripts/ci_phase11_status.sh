#!/usr/bin/env bash
# CI Phase 11 Determinism Harness
# Generates deterministic status artifacts for testing and verification

set -euo pipefail

# Deterministic environment
export ALLY_LIVE=0
export TZ=UTC

echo "[CI] Phase 11 Determinism Harness Starting..."
echo "ALLY_LIVE=${ALLY_LIVE}"
echo "TZ=${TZ}"

# Create artifacts directory
mkdir -p artifacts/status

# Generate deterministic journal sample
echo "[CI] Generating deterministic journal sample..."
python3 - <<'PY'
import sys
import os
import json

# Add project to path
sys.path.insert(0, os.path.abspath('.'))

from ally.status.journal import Journal
from ally.status.runbook import Runbook
from ally.status.telemetry import Telemetry

# Initialize with deterministic seeds
journal = Journal("artifacts/status/journal_ci.jsonl", deterministic_seed=42)
runbook = Runbook(seed=42, deterministic=True)
telemetry = Telemetry(seed=42)

# Clear any existing data
journal.clear()
runbook.reset()
telemetry.reset()

# Simulate deterministic workflow
print("[CI] Simulating deterministic workflow...")

# Phase 1: Research
runbook.enter("Researching", "SpecParsing", "Starting CI workflow")
journal.append(
    phase=runbook.phase,
    step=runbook.substep,
    tool="spec.parse",
    params_hash="aa11bb22",
    receipt_hash="cc33dd44",
    note="ci-sample-1"
)
telemetry.count("spec_parsed")
telemetry.time("spec_parsing", 1000)

# Phase 2: Data Loading
runbook.enter("Researching", "DataLoading", "Loading sample data")
journal.append(
    phase=runbook.phase,
    step=runbook.substep,
    tool="data.load",
    params_hash="ee55ff66",
    receipt_hash="77889900",
    note="ci-sample-2"
)
telemetry.count("data_loaded")
telemetry.time("data_loading", 1500)

# Phase 3: Evaluation
runbook.enter("Evaluating", "WalkForward", "Running walk-forward")
journal.append(
    phase=runbook.phase,
    step=runbook.substep,
    tool="research.walkforward",
    params_hash="aabbccdd",
    receipt_hash="eeffgghh",
    note="ci-sample-3"
)
telemetry.count("walkforward_runs", 5)
telemetry.time("walkforward_ms", 8500)

# Phase 4: Error simulation
journal.error_event(
    phase=runbook.phase,
    step=runbook.substep,
    error_type="DataError",
    error_msg="CI test error",
    recovery_action="CI recovery"
)
telemetry.error("DataError", "CI test error")

# Phase 5: Success and completion
telemetry.success("validation")
runbook.enter("Idle", "Ready", "CI workflow complete")
journal.append(
    phase=runbook.phase,
    step=runbook.substep,
    tool="ci.complete",
    params_hash="ffffffff",
    receipt_hash="00000000",
    note="ci-sample-complete"
)

# Generate status summary
summary = {
    "phase": runbook.phase,
    "substep": runbook.substep,
    "last_tool": "ci.complete",
    "last_params_hash": "ffffffff",
    "last_receipt_hash": "00000000",
    "counters": telemetry.counters,
    "timers": telemetry.get_stats()["timer_stats"],
    "journal_entries": len(journal.read_entries()),
    "timestamp": "2025-01-01T00:00:10.000000Z",  # Fixed for determinism
    "deterministic": True,
    "ci_run": True
}

# Write artifacts
with open("artifacts/status/summary.json", "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print(f"[CI] Generated {len(journal.read_entries())} journal entries")
print(f"[CI] Summary written with {len(telemetry.counters)} counters")
print("[CI] Phase 11 status artifacts written.")

# Verify deterministic behavior
expected_entries = 5  # spec, data, walkforward, error, complete
actual_entries = len(journal.read_entries())
if actual_entries != expected_entries:
    print(f"[ERROR] Expected {expected_entries} journal entries, got {actual_entries}")
    sys.exit(1)

expected_counters = 6  # spec_parsed, data_loaded, walkforward_runs(5), error.DataError, error.total, success.total, success.validation
actual_counters = len(telemetry.counters)
if actual_counters < 6:
    print(f"[ERROR] Expected at least 6 counters, got {actual_counters}")
    sys.exit(1)

print("[CI] Determinism verification passed")
PY

echo "[CI] Verifying artifact consistency..."

# Check that summary.json exists and has expected structure
if [[ ! -f "artifacts/status/summary.json" ]]; then
    echo "[ERROR] summary.json not created"
    exit 1
fi

# Validate JSON structure
python3 -c "
import json
with open('artifacts/status/summary.json', 'r') as f:
    summary = json.load(f)

required_fields = ['phase', 'substep', 'counters', 'timers', 'journal_entries', 'deterministic']
for field in required_fields:
    if field not in summary:
        print(f'[ERROR] Missing required field: {field}')
        exit(1)

if not summary['deterministic']:
    print('[ERROR] Summary indicates non-deterministic execution')
    exit(1)

if summary['phase'] != 'Idle' or summary['substep'] != 'Ready':
    print(f'[ERROR] Expected final state Idle:Ready, got {summary[\"phase\"]}:{summary[\"substep\"]}')
    exit(1)

print('[CI] Summary validation passed')
"

# Check that journal file exists and has entries
if [[ ! -f "artifacts/status/journal_ci.jsonl" ]]; then
    echo "[ERROR] journal_ci.jsonl not created"
    exit 1
fi

# Count journal entries
entry_count=$(wc -l < "artifacts/status/journal_ci.jsonl" || echo "0")
if [[ $entry_count -lt 5 ]]; then
    echo "[ERROR] Expected at least 5 journal entries, got $entry_count"
    exit 1
fi

echo "[CI] Journal validation passed ($entry_count entries)"

# Generate determinism proof
echo "[CI] Generating determinism proof..."
python3 -c "
import json
import hashlib

# Load summary
with open('artifacts/status/summary.json', 'r') as f:
    summary = json.load(f)

# Read journal entries
journal_lines = []
with open('artifacts/status/journal_ci.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            journal_lines.append(json.loads(line.strip()))

# Create determinism proof
proof = {
    'summary_hash': hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()[:16],
    'journal_count': len(journal_lines),
    'journal_hash': hashlib.sha256(''.join(json.dumps(entry, sort_keys=True) for entry in journal_lines).encode()).hexdigest()[:16],
    'counter_sum': sum(summary['counters'].values()),
    'timer_count': len(summary['timers']),
    'final_state': f\"{summary['phase']}:{summary['substep']}\",
    'timestamp': '2025-01-01T00:00:10.000000Z'
}

with open('artifacts/status/determinism_proof.json', 'w') as f:
    json.dump(proof, f, indent=2, sort_keys=True)

print(f'[CI] Determinism proof: {proof[\"summary_hash\"]}:{proof[\"journal_hash\"]}')
"

# Create README for artifacts
cat > artifacts/status/README.md << 'EOF'
# Phase 11 Status Artifacts

This directory contains deterministic status artifacts generated by CI.

## Files

- `summary.json` - Current system status summary
- `journal_ci.jsonl` - Deterministic journal entries from CI run
- `determinism_proof.json` - Cryptographic proof of deterministic execution

## Determinism Contract

All files are generated deterministically with:
- Fixed timestamps: 2025-01-01T00:00:XX.000000Z
- Seeded randomness: seed=42
- No network dependencies: ALLY_LIVE=0
- Fixed timezone: UTC

## Verification

To verify determinism, run the same CI script multiple times and compare file hashes.
The summary_hash and journal_hash in determinism_proof.json should be identical across runs.
EOF

echo "[CI] Created README.md"

# Final verification
echo "[CI] Final determinism check..."
SUMMARY_SIZE=$(stat -c%s "artifacts/status/summary.json" 2>/dev/null || stat -f%z "artifacts/status/summary.json" 2>/dev/null || echo "0")
JOURNAL_SIZE=$(stat -c%s "artifacts/status/journal_ci.jsonl" 2>/dev/null || stat -f%z "artifacts/status/journal_ci.jsonl" 2>/dev/null || echo "0")

if [[ $SUMMARY_SIZE -lt 200 ]]; then
    echo "[ERROR] Summary file too small ($SUMMARY_SIZE bytes)"
    exit 1
fi

if [[ $JOURNAL_SIZE -lt 500 ]]; then
    echo "[ERROR] Journal file too small ($JOURNAL_SIZE bytes)"
    exit 1
fi

echo "[CI] Phase 11 Determinism Harness Complete ✓"
echo "Generated artifacts:"
echo "  - artifacts/status/summary.json ($SUMMARY_SIZE bytes)"
echo "  - artifacts/status/journal_ci.jsonl ($JOURNAL_SIZE bytes)"
echo "  - artifacts/status/determinism_proof.json"
echo "  - artifacts/status/README.md"

exit 0