#!/usr/bin/env python3
import json, sys
from ally.tools import TOOL_REGISTRY

def main():
    # Mock strategy configurations for testing
    strategy_configs = [
        {"lookback": 20, "rebalance": "M", "vol_target": 0.15},
        {"lookback": 30, "rebalance": "M", "vol_target": 0.15},
        {"lookback": 40, "rebalance": "M", "vol_target": 0.15},
        {"lookback": 20, "rebalance": "W", "vol_target": 0.15},  # Duplicate config for dedup test
        {"lookback": 20, "rebalance": "M", "vol_target": 0.15},  # Exact duplicate
    ]

    result = TOOL_REGISTRY["grid.submit_jobs"](
        strategy_configs=strategy_configs,
        batch_id="test_batch_001",
        max_workers=4,
        dedup=True,
        resume=True
    )

    print(f"PROOF:GRID_JOBS: {result.data['n_submitted']}")
    print(f"PROOF:DEDUP_HITS: {result.data['n_deduped']}")
    print(f"PROOF:RESUMED: {result.data['n_resumed']}")

    # Status check
    status = TOOL_REGISTRY["grid.status"](batch_id="test_batch_001")
    if status.ok:
        summary = status.data['summary']
        print(f"PROOF:GRID_STATUS: {summary}")

    # Output full result
    out = {
        "grid": result.data,
        "status": status.data if status.ok else None
    }
    print(json.dumps(out))

if __name__ == "__main__":
    sys.exit(main())