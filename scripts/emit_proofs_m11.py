#!/usr/bin/env python3
"""
Generate deterministic proof bundle for M11 — Transaction Costs & Microstructure.
"""

import json
import sys
import os
import hashlib
sys.path.insert(0, os.path.abspath('.'))

from ally.tools import TOOL_REGISTRY

def generate_m11_proofs():
    """Generate deterministic M11 transaction cost proofs."""
    
    # 1. TCOST_CONFIG - Generate benchmark configuration
    config_result = TOOL_REGISTRY["tcost.benchmark_config"](
        market_regime="normal",
        asset_class="equity"
    )
    
    if not config_result.ok:
        raise Exception(f"Failed to generate tcost config: {config_result.data}")
    
    tcost_config = config_result.data
    
    # 2. FILLS_FINGERPRINT - Generate deterministic fills
    market_conditions = {
        "price": 100.0,
        "volatility": 0.02,
        "spread_bps": 10,
        "avg_volume": 100000
    }
    
    fills_result = TOOL_REGISTRY["tcost.simulate_fills"](
        symbol="TEST",
        target_quantity=2000,
        order_side="buy",
        market_conditions=market_conditions,
        execution_strategy="twap",
        time_horizon_minutes=30,
        seed=1337  # Deterministic seed
    )
    
    if not fills_result.ok:
        raise Exception(f"Failed to generate fills: {fills_result.data}")
    
    fills_fingerprint = fills_result.data["fills_fingerprint"]
    
    # 3. TCOST_IMPACT_BPS - Calculate typical impact for proof
    # Use a simple calculation for deterministic results
    avg_fill_price = fills_result.data["summary"]["avg_fill_price"]
    total_quantity = fills_result.data["summary"]["total_quantity"]
    
    # Simplified impact calculation for proof (basis points)
    impact_bps = round((total_quantity / 10000) * (tcost_config["config"]["market_impact_alpha"] * 10), 2)
    
    # Create proof bundle
    proofs = {
        "TCOST_CONFIG": {
            "regime": tcost_config["regime"],
            "asset_class": tcost_config["asset_class"],
            "commission_bps": tcost_config["config"]["commission_bps"],
            "market_impact_alpha": tcost_config["config"]["market_impact_alpha"],
            "config_hash": tcost_config["config_hash"]
        },
        "TCOST_IMPACT_BPS": impact_bps,
        "FILLS_FINGERPRINT": fills_fingerprint
    }
    
    # Output proofs
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/m11_proofs.json", "w") as f:
        json.dump(proofs, f, indent=2, sort_keys=True)
    
    # Print individual proof lines
    for key, value in proofs.items():
        print(f"PROOF:{key}:", json.dumps(value) if not isinstance(value, (str, int, float)) else value)

if __name__ == "__main__":
    generate_m11_proofs()