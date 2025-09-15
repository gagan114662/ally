#!/bin/bash
set -euo pipefail

# CI Orchestrator Diagnostics Script
# Runs research doctor + test loop to validate orchestrator fixes
# Generates deterministic artifacts for CI verification

echo "🩺 CI Orchestrator Diagnostics"
echo "ALLY_LIVE=${ALLY_LIVE:-0}"
echo "TZ=${TZ:-UTC}"
echo "PWD=$(pwd)"
echo

# Ensure deterministic mode
export ALLY_LIVE=0
export TZ=UTC

# Create artifacts directories
mkdir -p artifacts/research
mkdir -p artifacts/receipts

echo "📋 Step 1: Research Doctor (health check)"
echo "==========================================="

# Run research doctor with auto-fix
python -m ally.cli.research_cli research doctor \
  --fixtures data/fixtures \
  --create-missing \
  --json-output > artifacts/research/doctor_report.json 2>&1 || {
    echo "❌ Research doctor failed"
    cat artifacts/research/doctor_report.json
    exit 1
}

echo "✅ Research doctor completed"
echo

echo "📊 Step 2: Test Orchestrator Loop"
echo "=================================="

# Run test orchestration loop
python -m ally.cli.research_cli research test-loop \
  --dry-run \
  --budget 3 > artifacts/research/test_loop.log 2>&1 || {
    echo "❌ Test loop failed"
    cat artifacts/research/test_loop.log
    exit 1
}

echo "✅ Test orchestrator loop completed"
echo

echo "📄 Step 3: Generate Summary Report"
echo "==================================="

# Create comprehensive summary
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Load reports
doctor_report = {}
if Path("artifacts/research/doctor_report.json").exists():
    with open("artifacts/research/doctor_report.json") as f:
        doctor_report = json.load(f)

test_results = {}
if Path("artifacts/research/test_loop_results.json").exists():
    with open("artifacts/research/test_loop_results.json") as f:
        test_results = json.load(f)

# Create orchestrator summary
summary = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "ci_environment": {
        "ally_live": os.getenv("ALLY_LIVE", "0"),
        "tz": os.getenv("TZ", "UTC"),
        "ci": os.getenv("CI", "false")
    },
    "doctor_check": doctor_report,
    "test_orchestration": test_results,
    "overall_status": "healthy" if (
        doctor_report.get("overall_ok", False) and
        test_results.get("success", False)
    ) else "unhealthy",
    "artifacts_generated": [
        "artifacts/research/doctor_report.json",
        "artifacts/research/test_loop_results.json",
        "artifacts/research/orchestrator_summary.json"
    ]
}

# Save summary
with open("artifacts/research/orchestrator_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("✅ Orchestrator summary generated")
print(f"   Status: {summary['overall_status']}")
if "test_orchestration" in summary:
    t = summary["test_orchestration"]
    print(f"   Pipeline: {t.get('templates', 0)}→{t.get('variants', 0)}→{t.get('scored', 0)}→{t.get('survivors', 0)}")

PY

echo

echo "🧾 Step 4: Verify Artifacts"
echo "============================"

# List all generated artifacts with sizes
find artifacts/research -type f -name "*.json" -o -name "*.log" | while read -r file; do
    size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null || echo "unknown")
    echo "📄 $file (${size} bytes)"
done

echo

# Check for required receipts (would be in receipts.jsonl or DuckDB)
if [ -f "artifacts/receipts.jsonl" ]; then
    receipt_count=$(wc -l < artifacts/receipts.jsonl)
    echo "📋 Found ${receipt_count} receipts in artifacts/receipts.jsonl"
else
    echo "📋 No artifacts/receipts.jsonl found (receipts may be in DuckDB)"
fi

echo

echo "✅ CI Orchestrator Diagnostics Complete"
echo "======================================="

# Final status check
if [ -f "artifacts/research/orchestrator_summary.json" ]; then
    status=$(python -c "import json; print(json.load(open('artifacts/research/orchestrator_summary.json')).get('overall_status', 'unknown'))")
    echo "🎯 Overall Status: $status"

    if [ "$status" != "healthy" ]; then
        echo "❌ Orchestrator diagnostics indicate issues - check reports above"
        exit 1
    fi
else
    echo "❌ Missing orchestrator summary - diagnostics failed"
    exit 1
fi

echo "🎉 All orchestrator diagnostics passed!"