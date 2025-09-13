#!/usr/bin/env python3
"""
Emit M-FactorLens Gate Proofs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ally.tools import execute_tool
import json


def main():
    """Emit M-FactorLens Gate proofs for CI verification"""

    try:
        print("# M-FactorLens Gate Proof Emission")

        # Generate synthetic returns for testing
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta

        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
        returns = []
        for i, date in enumerate(dates):
            daily_return = np.random.randn() * 0.02 + 0.0003  # 3bps daily drift
            returns.append({
                "date": date.strftime("%Y-%m-%d"),
                "return": float(daily_return)
            })

        # Test gate with synthetic data
        gate_result = execute_tool("orchestrator.factor_gate", returns=returns)

        if not gate_result.ok:
            print(f"ERROR: Gate execution failed: {gate_result.errors}")
            sys.exit(1)

        gate_data = gate_result.data
        proofs = gate_data.get("proofs", {})

        # Emit core proofs
        print(f"PROOF:FACTLENS_GATE: {proofs.get('FACTLENS_GATE', 'UNKNOWN')}")
        print(f"PROOF:RES_ALPHA_T: {proofs.get('RES_ALPHA_T', 0.0)}")
        print(f"PROOF:BETAS_OK: {proofs.get('BETAS_OK', 'false')}")
        print(f"PROOF:FACTORLENS_HASH: {proofs.get('FACTORLENS_HASH', 'missing')}")

        # Bulletproof audit proofs
        print(f"PROOF:PIT_OK: {proofs.get('PIT_OK', 'false')}")
        print(f"PROOF:NW_LAGS: {proofs.get('NW_LAGS', 0)}")
        print(f"PROOF:WINDOW_DAYS: {proofs.get('WINDOW_DAYS', 0)}")
        print(f"PROOF:STEP_DAYS: {proofs.get('STEP_DAYS', 0)}")
        print(f"PROOF:MIN_OBS: {proofs.get('MIN_OBS', 0)}")
        print(f"PROOF:OOS_TSTAT: {proofs.get('OOS_TSTAT', 0.0)}")
        print(f"PROOF:FDR_ALPHA: {proofs.get('FDR_ALPHA', 'pending')}")
        print(f"PROOF:INSUFFICIENT_OOS: {proofs.get('INSUFFICIENT_OOS', 'false')}")

        # Additional verification proofs
        print(f"PROOF:GATE_LOGIC: {'pass' if gate_data.get('gate_pass') else 'fail'}")
        print(f"PROOF:ALPHA_BPS: {gate_data.get('alpha_bps', 0.0)}")

        # Factor exposure summary
        exposures = gate_data.get("exposures", [])
        factor_names = [exp["factor"] for exp in exposures if exp["factor"] != "alpha"]
        print(f"PROOF:FACTOR_COUNT: {len(factor_names)}")

        violations = gate_data.get("violations", [])
        print(f"PROOF:VIOLATIONS: {len(violations)}")

        # Test pipeline orchestrator
        pipeline_result = execute_tool("orchestrator.run_pipeline", enforce_gate=False)

        if pipeline_result.ok:
            pipeline_data = pipeline_result.data
            print(f"PROOF:PIPELINE_STATUS: {pipeline_data.get('pipeline_status', 'unknown')}")
            print(f"PROOF:PIPELINE_HASH: {hash(str(pipeline_data))}")
        else:
            print("PROOF:PIPELINE_STATUS: error")
            print("PROOF:PIPELINE_HASH: missing")

        print("# M-FactorLens Gate proofs emitted successfully")

    except Exception as e:
        print(f"ERROR: Proof emission failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()