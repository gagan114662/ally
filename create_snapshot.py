#!/usr/bin/env python3
"""
Create snapshot of sentinel receipt hashes for deterministic guard testing
"""

import json
import os

def create_sentinel_snapshot():
    """Create snapshot from current receipts"""
    if not os.path.exists("artifacts/receipts.jsonl"):
        print("❌ No receipts file found")
        return False

    # Load receipts and find latest for each tool
    latest_receipts = {}
    with open("artifacts/receipts.jsonl", 'r') as f:
        for line in f:
            receipt = json.loads(line.strip())
            tool = receipt['tool']
            if tool not in latest_receipts or receipt['ts'] > latest_receipts[tool]['ts']:
                latest_receipts[tool] = receipt

    # Create snapshot for required sentinels
    required_tools = {
        'ops.drift.data': 'data',
        'ops.drift.strategy': 'strategy',
        'ops.drift.ops': 'ops'
    }

    snapshot = {}
    for tool, key in required_tools.items():
        if tool in latest_receipts:
            snapshot[key] = {
                "tool": tool,
                "receipt_hash": latest_receipts[tool]['receipt_hash']
            }
        else:
            print(f"❌ No receipt found for {tool}")
            return False

    # Write snapshot
    os.makedirs("artifacts/ops/snapshots", exist_ok=True)
    snapshot_path = "artifacts/ops/snapshots/sentinels_ci.json"

    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    print(f"✅ Created snapshot: {snapshot_path}")
    print(f"   Snapshot contains: {list(snapshot.keys())}")

    return True

if __name__ == '__main__':
    success = create_sentinel_snapshot()
    exit(0 if success else 1)