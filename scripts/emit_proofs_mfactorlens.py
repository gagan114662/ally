#!/usr/bin/env python3
"""
M-FactorLens proof emitter
Generates deterministic proofs for factor exposure and residual alpha analysis
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from ally.tools.factors import load_ff, compute_exposures, compute_residual_alpha
from ally.utils.determinism import set_global_determinism

def main():
    """Generate and emit M-FactorLens proofs"""
    set_global_determinism(1337)

    # Load factor set
    ff_result = load_ff()
    assert ff_result.ok, "Failed to load factor set"

    factor_meta = ff_result.data["meta"]
    factor_cols = factor_meta["columns"]

    # Create deterministic synthetic returns for proof generation
    np.random.seed(1337)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')

    # Synthetic portfolio with known properties for proof verification
    synthetic_returns = []
    for i, date in enumerate(dates):
        # Deterministic return series with modest alpha
        base_return = 0.0003 + 0.0001 * np.sin(i * 0.1)  # 30 bps base + cyclical
        noise = 0.0002 * np.sin(i * 0.05 + 1.5)  # Low deterministic noise
        ret_val = base_return + noise

        synthetic_returns.append({
            "date": date.strftime("%Y-%m-%d"),
            "ret": ret_val
        })

    # Compute exposures
    exp_result = compute_exposures(returns=synthetic_returns, lags=5)
    assert exp_result.ok, "Failed to compute exposures"

    # Compute residual alpha
    alpha_result = compute_residual_alpha(returns=synthetic_returns, window=252, step=21, lags=5)
    assert alpha_result.ok, "Failed to compute residual alpha"

    # Extract proof values
    exposures_data = exp_result.data
    alpha_data = alpha_result.data

    r2 = exposures_data["r2"]
    alpha_tstat = alpha_data["residual"]["alpha_tstat"]
    alpha_bps = alpha_data["residual"]["alpha_bps"]
    det_hash = alpha_data["det_hash"]

    # Emit proofs
    print(f"PROOF:FACTOR_SET: {json.dumps(factor_cols)}")
    print(f"PROOF:ALPHA_TSTAT: {alpha_tstat:.6f}")
    print(f"PROOF:R2: {r2:.6f}")
    print(f"PROOF:RES_ALPHA_MEAN: {alpha_bps:.2f}")
    print(f"PROOF:FACTORLENS_HASH: {det_hash}")

    # Save detailed proof bundle
    proof_bundle = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
            "determinism_seed": 1337
        },
        "factor_set": {
            "name": factor_meta["name"],
            "columns": factor_cols,
            "frequency": factor_meta["frequency"]
        },
        "exposures": exposures_data,
        "residual_alpha": alpha_data,
        "proofs": {
            "FACTOR_SET": factor_cols,
            "ALPHA_TSTAT": alpha_tstat,
            "R2": r2,
            "RES_ALPHA_MEAN": alpha_bps,
            "FACTORLENS_HASH": det_hash
        }
    }

    with open("factorlens-proof-bundle/mfactorlens_proofs.json", "w") as f:
        json.dump(proof_bundle, f, indent=2, default=str)

    print(f"\nProof bundle saved to: factorlens-proof-bundle/mfactorlens_proofs.json")

if __name__ == "__main__":
    main()