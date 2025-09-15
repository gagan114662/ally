#!/bin/bash
set -euo pipefail

echo "🔄 CI Phase 8 Determinism Test"
echo "================================="

# 0) Clean artifacts to simulate fresh CI
echo "🧹 Cleaning artifacts..."
rm -rf artifacts/ops artifacts/receipts.jsonl || true
mkdir -p artifacts/ops/drift artifacts/ops/guard artifacts/ops/snapshots

POLICY=ally/ops/policy.yaml

# 1) Produce deterministic sentinel receipts on fixtures (offline)
echo "📊 Running sentinel checks..."
python ally_ops_cli.py drift-data --panel artifacts/fixtures/features_small.json
python ally_ops_cli.py drift-strategy --strategy TEST_STRAT_XS_MOMENTUM
python ally_ops_cli.py drift-ops --fixture artifacts/fixtures/determinism.pkl

# 2) Snapshot the exact sentinel hashes we'll bind to the guard
echo "📸 Creating sentinel snapshot..."
python create_snapshot.py

# 3) Run guard twice, bound to the snapshot (must be identical)
echo "🔒 Testing deterministic guard (run 1)..."
python ally_ops_cli.py guard --bundle TEST_BUNDLE_SHA --snapshot artifacts/ops/snapshots/sentinels_ci.json

echo "🔒 Testing deterministic guard (run 2)..."
python ally_ops_cli.py guard --bundle TEST_BUNDLE_SHA --snapshot artifacts/ops/snapshots/sentinels_ci.json

# 4) Determinism checks
echo "🔍 Checking determinism..."
python - << 'PY'
import json
from collections import defaultdict

# Load receipts
receipts = []
with open("artifacts/receipts.jsonl", "r") as f:
    for line in f:
        receipts.append(json.loads(line.strip()))

# Check guard determinism
guard_params = defaultdict(set)
for r in receipts:
    if r["tool"] == "ops.guard":
        guard_params[r["params_hash"]].add(r["receipt_hash"])

# Report results
offenders = 0
for params_hash, receipt_hashes in guard_params.items():
    if len(receipt_hashes) > 1:
        print(f"❌ Non-deterministic: params {params_hash} -> {len(receipt_hashes)} receipt hashes")
        offenders += 1
    else:
        print(f"✅ Deterministic: params {params_hash} -> 1 receipt hash")

# Write results
with open("artifacts/ops/guard_determinism.txt", "w") as f:
    f.write(f"determinism_guard offenders {offenders}\n")

if offenders > 0:
    print(f"💥 {offenders} determinism violations found")
    exit(1)
else:
    print("🎉 All determinism checks passed")

PY

echo "================================="
echo "✅ CI Phase 8 determinism test passed!"