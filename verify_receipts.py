#!/usr/bin/env python3
"""
Simple verification script to check receipts and artifacts
"""

import json
import os
from collections import defaultdict

def check_receipts():
    """Check receipts file for determinism and coverage"""

    if not os.path.exists("artifacts/receipts.jsonl"):
        print("❌ No receipts file found")
        return False

    receipts = []
    with open("artifacts/receipts.jsonl", 'r') as f:
        for line in f:
            receipts.append(json.loads(line.strip()))

    print(f"📊 Found {len(receipts)} receipt entries")

    # Check determinism (same params -> single receipt hash)
    tool_params = defaultdict(set)
    for receipt in receipts:
        key = (receipt['tool'], receipt['params_hash'])
        tool_params[key].add(receipt['receipt_hash'])

    determinism_violations = []
    for (tool, params_hash), receipt_hashes in tool_params.items():
        if len(receipt_hashes) > 1:
            determinism_violations.append((tool, params_hash, len(receipt_hashes)))

    if determinism_violations:
        print("❌ Determinism violations found:")
        for tool, params_hash, count in determinism_violations:
            print(f"  {tool} {params_hash}: {count} different receipt hashes")
        return False
    else:
        print("✅ Determinism check passed")

    # Check latest statuses exist
    latest_tools = {}
    for receipt in receipts:
        tool = receipt['tool']
        if tool not in latest_tools or receipt['ts'] > latest_tools[tool]['ts']:
            latest_tools[tool] = receipt

    required_tools = ['ops.drift.data', 'ops.drift.strategy', 'ops.drift.ops']
    for tool in required_tools:
        if tool in latest_tools:
            status = latest_tools[tool]['extra'].get('status', 'UNKNOWN')
            print(f"✅ {tool}: {status}")
        else:
            print(f"❌ {tool}: No receipt found")
            return False

    # Check guard logic
    if 'ops.guard' in latest_tools:
        guard_status = latest_tools['ops.guard']['extra'].get('status', 'UNKNOWN')
        print(f"✅ ops.guard: {guard_status}")

        # Simple guard check: if any drift tool failed, guard should block
        drift_statuses = [latest_tools[tool]['extra'].get('status') for tool in required_tools if tool in latest_tools]
        any_failed = any(status != 'OK' for status in drift_statuses)

        if any_failed and guard_status == 'OK':
            print("❌ Guard logic error: should block when drift tools fail")
            return False
        else:
            print("✅ Guard logic check passed")

    # Check heartbeat
    if 'ops.heartbeat' in latest_tools:
        heartbeat_status = latest_tools['ops.heartbeat']['extra'].get('status', 'UNKNOWN')
        print(f"✅ ops.heartbeat: {heartbeat_status}")

    return True

def check_artifacts():
    """Check that artifacts were created by tools"""

    required_artifacts = [
        "artifacts/ops/drift/data_example.json",
        "artifacts/ops/drift/strategy_example.json",
        "artifacts/ops/drift/ops_example.json",
        "artifacts/ops/guard/bundle_example.json",
        "artifacts/report/status.json"
    ]

    all_exist = True
    for artifact in required_artifacts:
        if os.path.exists(artifact):
            print(f"✅ {artifact}")
        else:
            print(f"❌ {artifact} missing")
            all_exist = False

    return all_exist

def main():
    print("🔍 Verifying Phase 8 implementation...")
    print()

    receipts_ok = check_receipts()
    print()

    artifacts_ok = check_artifacts()
    print()

    if receipts_ok and artifacts_ok:
        print("🎉 All verification checks passed!")
        return True
    else:
        print("💥 Some verification checks failed!")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)