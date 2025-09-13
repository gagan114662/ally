#!/usr/bin/env python3
import json, sys
from ally.tools import TOOL_REGISTRY

def main():
    strategy_id = "demo_strategy"
    selection_sha1 = "deadbeef"*5
    symbols = ["SPY","QQQ","TLT","GLD"]

    gate = TOOL_REGISTRY["promotion.holdout_gate"](
        strategy_id=strategy_id,
        selection_sha1=selection_sha1,
        symbols=symbols
    )
    print(f"PROOF:PROMO_DECISION: {gate.data['decision']}")
    print(f"PROOF:PROMO_HOLDOUT_T: {round(gate.data['alpha_tstat'], 6)}")
    print(f"PROOF:PROMO_BETAS_OK: {gate.data['betas_ok']}")
    print(f"PROOF:PROMO_TURNOVER_X: {gate.data['turnover_x']}")
    print(f"PROOF:PROMO_COST_BPS: {gate.data['impact_bps']}")

    bun = TOOL_REGISTRY["promotion.bundle"](
        strategy_id=strategy_id, selection_sha1=selection_sha1
    )
    print(f"PROOF:BUNDLE_SHA1: {bun.data['bundle_sha1']}")
    print(f"PROOF:BUNDLE_PATH: {bun.data['bundle_path']}")

    # Save json proof bundle
    out = {
        "gate": gate.data,
        "bundle": bun.data,
    }
    print(json.dumps(out))  # optional
if __name__ == "__main__":
    sys.exit(main())