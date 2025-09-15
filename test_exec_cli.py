#!/usr/bin/env python3
"""
Simple test for execution CLI integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test imports
try:
    sys.path.append('./ally/ally/ally')
    from execution.backends.simulator import simulate_execution
    print("✅ Simulator import successful")
except ImportError as e:
    print(f"❌ Simulator import failed: {e}")
    sys.exit(1)

# Test simulation with fixtures
try:
    result = simulate_execution(
        target_weights_path="artifacts/fixtures/phase10/target_weights.json",
        last_positions_path="artifacts/fixtures/phase10/last_positions.json",
        symbols_path="artifacts/fixtures/phase10/symbols_meta.csv",
        prices_path="artifacts/fixtures/phase10/prices_intraday.csv",
        cost_model_path="artifacts/fixtures/phase10/cost_model.yaml",
        slippage_model='sqrt',
        impact_k=6.0,
        latency_bars=0,
        live=False
    )

    print("✅ Execution simulation successful")
    print(f"📊 Orders placed: {result.get('orders_placed', 0)}")
    print(f"📊 Orders filled: {result.get('orders_filled', 0)}")
    print(f"📊 Avg slippage: {result.get('avg_slippage_bps', 0):.2f} bps")
    print(f"📋 Orders receipt: {result.get('orders_receipt', 'N/A')}")
    print(f"📋 Trades receipt: {result.get('trades_receipt', 'N/A')}")

except Exception as e:
    print(f"❌ Execution simulation failed: {e}")
    sys.exit(1)

print("\n🎉 All execution tests passed!")