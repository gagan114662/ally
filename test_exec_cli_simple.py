#!/usr/bin/env python3
"""
Simple execution CLI test using direct function calls
"""

import sys
import os

# Add the execution path
sys.path.append('./ally/ally/ally')

def test_simulate_command():
    """Test the simulation functionality"""

    print("🧪 Testing execution simulation...")

    # Import the function directly
    try:
        from execution.backends.simulator import simulate_execution
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test bundle shortcuts
    fixture_base = "artifacts/fixtures/phase10"
    target = f"{fixture_base}/target_weights.json"
    positions = f"{fixture_base}/last_positions.json"
    symbols = f"{fixture_base}/symbols_meta.csv"
    prices = f"{fixture_base}/prices_intraday.csv"
    cost_model = f"{fixture_base}/cost_model.yaml"

    # Validate files exist
    required_files = {
        "target weights": target,
        "positions": positions,
        "symbols metadata": symbols,
        "prices": prices,
        "cost model": cost_model
    }

    for name, path in required_files.items():
        if not os.path.exists(path):
            print(f"❌ {name} file not found: {path}")
            return False

    print("✅ All required files found")

    # Run simulation
    try:
        result = simulate_execution(
            target_weights_path=target,
            last_positions_path=positions,
            symbols_path=symbols,
            prices_path=prices,
            cost_model_path=cost_model,
            slippage_model='sqrt',
            impact_k=6.0,
            latency_bars=0,
            live=False
        )

        print("✅ Execution simulation complete")
        print(f"   Orders placed: {result.get('orders_placed', 0)}")
        print(f"   Orders filled: {result.get('orders_filled', 0)}")
        print(f"   Total trades: {result.get('total_trades', 0)}")
        print(f"   Total notional: ${result.get('total_notional', 0):,.0f}")
        print(f"   Avg slippage: {result.get('avg_slippage_bps', 0):.2f} bps")
        print(f"   Orders receipt: {result.get('orders_receipt', 'N/A')}")
        print(f"   Trades receipt: {result.get('trades_receipt', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False

def test_kill_command():
    """Test the kill switch functionality"""

    print("\n🧪 Testing kill switch...")

    try:
        from execution.backends.simulator import SimulatorBackend

        # Create a fresh simulator
        simulator = SimulatorBackend()
        result = simulator.kill_all_orders()

        print("✅ Kill switch activated")
        print(f"   Canceled orders: {result.get('canceled_count', 0)}")
        print(f"   Remaining open: {result.get('open_order_count', 0)}")
        print(f"   Receipt: {result.get('receipt_hash', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ Kill switch failed: {e}")
        return False

def test_status_command():
    """Test the status functionality"""

    print("\n🧪 Testing status check...")

    # Check environment
    ally_live = os.environ.get("ALLY_LIVE", "0")
    live_status = "🔴 LIVE" if ally_live == "1" else "🟢 PAPER"
    print(f"   Environment: {live_status}")

    # Check directories
    directories = [
        "artifacts/execution",
        "artifacts/fixtures/phase10"
    ]

    for directory in directories:
        exists = "✅" if os.path.exists(directory) else "❌"
        print(f"   {exists} {directory}")

    # Check journal files
    journal_files = [
        "artifacts/execution/orders.jsonl",
        "artifacts/execution/trades.jsonl"
    ]

    for file_path in journal_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({size} bytes)")
        else:
            print(f"   ❌ {file_path}")

    print("✅ Status check complete")
    return True

if __name__ == "__main__":
    print("🚀 Execution CLI Integration Test")
    print("=" * 40)

    all_passed = True

    # Test simulation
    all_passed &= test_simulate_command()

    # Test kill switch
    all_passed &= test_kill_command()

    # Test status
    all_passed &= test_status_command()

    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All execution CLI tests passed!")
    else:
        print("❌ Some tests failed")
        sys.exit(1)