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
        symbols=symbols,
        receipts_required=True,
        fdr_run_id="test_fdr_run_001",
        pit_snapshot_hash="test_pit_abc123"
    )

    # Core promotion decision proofs
    print(f"PROOF:PROMO_DECISION: {gate.data['decision']}")

    # Only emit additional proofs if gate passed (has full data)
    if gate.data['decision'] == 'PASS':
        print(f"PROOF:PROMO_HOLDOUT_T: {round(gate.data['alpha_tstat'], 6)}")
        print(f"PROOF:PROMO_BETAS_OK: {gate.data['betas_ok']}")
        print(f"PROOF:PROMO_TURNOVER_X: {gate.data['turnover_x']}")
        print(f"PROOF:PROMO_COST_BPS: {gate.data['impact_bps']}")

        # Hardening proofs
        print(f"PROOF:RECEIPTS_PRESENT: {gate.data['receipts_present']}")
        print(f"PROOF:PIT_SNAPSHOT_HASH: {gate.data['pit_snapshot_hash']}")
        print(f"PROOF:FDR_RUN_ID: {gate.data['fdr_run_id']}")
        print(f"PROOF:FDR_Q: {gate.data['fdr_q']}")
        print(f"PROOF:ADV_OK: {gate.data['adv_ok']}")
        print(f"PROOF:HOLDOUT_GAP_OK: {gate.data['holdout_gap_ok']}")
        print(f"PROOF:STRESS_T_OK: {gate.data['stress_t_ok']}")
        print(f"PROOF:TCOST_MODEL_HASH: {gate.data['tcost_model_hash']}")
        print(f"PROOF:PROMO_DET_HASH: {gate.data['promo_det_hash']}")
    else:
        # Gate failed - emit failure reason
        print(f"PROOF:FAILURE_REASON: {gate.data.get('failure_reason', 'Unknown')}")
        print(f"PROOF:RECEIPTS_PRESENT: {gate.data.get('receipts_present', False)}")

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