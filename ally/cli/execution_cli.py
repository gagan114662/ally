#!/usr/bin/env python3
"""
Execution CLI commands for Phase 10
"""

import os
import json
import typer
from pathlib import Path
from typing import Optional

# Create execution subcommand group
exec_app = typer.Typer(help="Execution and order management commands")

@exec_app.command("simulate")
def simulate_execution(
    target: str = typer.Option(..., "--target", help="Path to target weights JSON file"),
    positions: Optional[str] = typer.Option(None, "--positions", help="Path to last positions JSON file"),
    symbols: Optional[str] = typer.Option(None, "--symbols", help="Path to symbols metadata CSV file"),
    prices: Optional[str] = typer.Option(None, "--prices", help="Path to intraday prices CSV file"),
    cost_model: Optional[str] = typer.Option(None, "--cost-model", help="Path to cost model YAML file"),
    slippage_model: str = typer.Option("sqrt", "--slippage", help="Slippage model (linear or sqrt)"),
    impact_k: float = typer.Option(6.0, "--impact-k", help="Impact parameter k"),
    latency_bars: int = typer.Option(0, "--latency", help="Latency in bars"),
    live: bool = typer.Option(False, "--live", help="Live trading mode (requires double-gating)"),
    bundle: Optional[str] = typer.Option(None, "--bundle", help="Test bundle name for fixtures")
):
    """
    Simulate portfolio execution with slippage and constraints

    Examples:
        ally exec simulate --target target_weights.json --bundle TEST_BUNDLE
        ally exec simulate --target target_weights.json --positions positions.json --live=false
    """

    # Import here to avoid circular imports
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ally', 'ally'))
        from execution.backends.simulator import simulate_execution
    except ImportError as e:
        typer.echo(f"❌ Execution simulator not available: {e}")
        raise typer.Exit(code=1)

    # Handle bundle shortcuts
    if bundle:
        if bundle == "TEST_BUNDLE":
            # Use Phase 10 fixtures
            fixture_base = "artifacts/fixtures/phase10"
            target = target or f"{fixture_base}/target_weights.json"
            positions = positions or f"{fixture_base}/last_positions.json"
            symbols = symbols or f"{fixture_base}/symbols_meta.csv"
            prices = prices or f"{fixture_base}/prices_intraday.csv"
            cost_model = cost_model or f"{fixture_base}/cost_model.yaml"
        else:
            typer.echo(f"❌ Unknown bundle: {bundle}")
            raise typer.Exit(code=1)

    # Validate required files
    required_files = {
        "target weights": target,
        "symbols metadata": symbols,
        "prices": prices,
        "cost model": cost_model
    }

    for name, path in required_files.items():
        if not path or not os.path.exists(path):
            typer.echo(f"❌ {name} file not found: {path}")
            raise typer.Exit(code=1)

    # Use default positions if not provided
    if not positions:
        positions = "artifacts/fixtures/phase10/last_positions.json"

    if not os.path.exists(positions):
        typer.echo(f"❌ Positions file not found: {positions}")
        raise typer.Exit(code=1)

    # Safety check for live trading
    if live:
        ally_live = os.environ.get("ALLY_LIVE", "0")
        if ally_live != "1":
            typer.echo("❌ Live trading requires ALLY_LIVE=1 environment variable")
            typer.echo("   This is a safety measure to prevent accidental live trading")
            raise typer.Exit(code=1)

        typer.echo("⚠️  LIVE TRADING MODE ENABLED")
        typer.echo("   Double-gating confirmed: --live=true AND ALLY_LIVE=1")

    # Run simulation
    typer.echo(f"🎯 Starting execution simulation")
    typer.echo(f"   Target weights: {target}")
    typer.echo(f"   Last positions: {positions}")
    typer.echo(f"   Slippage model: {slippage_model}")
    typer.echo(f"   Impact K: {impact_k}")
    typer.echo(f"   Latency: {latency_bars} bars")
    typer.echo(f"   Live mode: {live}")
    typer.echo()

    try:
        result = simulate_execution(
            target_weights_path=target,
            last_positions_path=positions,
            symbols_path=symbols,
            prices_path=prices,
            cost_model_path=cost_model,
            slippage_model=slippage_model,
            impact_k=impact_k,
            latency_bars=latency_bars,
            live=live
        )

        # Display results
        typer.echo("✅ Execution simulation complete")
        typer.echo()
        typer.echo("📊 Execution Summary:")
        typer.echo(f"   Orders placed: {result.get('orders_placed', 0)}")
        typer.echo(f"   Orders filled: {result.get('orders_filled', 0)}")
        typer.echo(f"   Total trades: {result.get('total_trades', 0)}")
        typer.echo(f"   Total notional: ${result.get('total_notional', 0):,.0f}")
        typer.echo(f"   Avg slippage: {result.get('avg_slippage_bps', 0):.2f} bps")
        typer.echo()

        # Show receipts
        orders_receipt = result.get('orders_receipt')
        trades_receipt = result.get('trades_receipt')

        if orders_receipt:
            typer.echo(f"📋 Orders receipt: {orders_receipt}")
        if trades_receipt:
            typer.echo(f"📋 Trades receipt: {trades_receipt}")

        # Show status
        if result.get('status') == 'ERROR':
            typer.echo(f"❌ Error: {result.get('error', 'Unknown error')}")
            raise typer.Exit(code=1)

    except Exception as e:
        typer.echo(f"❌ Execution failed: {e}")
        raise typer.Exit(code=1)


@exec_app.command("kill")
def kill_all_orders():
    """
    Kill switch - cancel all open orders across all backends

    This is an emergency function that cancels all open orders.
    Use with caution in live trading environments.
    """

    # Import here to avoid circular imports
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ally', 'ally'))
        from execution.backends.simulator import SimulatorBackend
    except ImportError as e:
        typer.echo(f"❌ Execution backend not available: {e}")
        raise typer.Exit(code=1)

    # Check if this is live trading
    ally_live = os.environ.get("ALLY_LIVE", "0")
    if ally_live == "1":
        confirm = typer.confirm("⚠️  LIVE TRADING DETECTED. Are you sure you want to cancel ALL orders?")
        if not confirm:
            typer.echo("Kill switch canceled")
            return

    typer.echo("🚨 Activating kill switch...")

    try:
        # Initialize simulator backend (in real implementation, would iterate all backends)
        simulator = SimulatorBackend()
        result = simulator.kill_all_orders()

        typer.echo("✅ Kill switch activated")
        typer.echo(f"   Canceled orders: {result.get('canceled_count', 0)}")
        typer.echo(f"   Remaining open: {result.get('open_order_count', 0)}")
        typer.echo(f"   Receipt: {result.get('receipt_hash', 'N/A')}")

    except Exception as e:
        typer.echo(f"❌ Kill switch failed: {e}")
        raise typer.Exit(code=1)


@exec_app.command("orders")
def list_orders(
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status (NEW, FILLED, etc.)"),
    limit: int = typer.Option(10, "--limit", help="Number of orders to show")
):
    """
    List recent orders from execution journal
    """

    orders_file = Path("artifacts/execution/orders.jsonl")

    if not orders_file.exists():
        typer.echo("❌ No orders journal found")
        typer.echo(f"   Expected: {orders_file}")
        typer.echo("   Run an execution simulation first")
        return

    typer.echo(f"📋 Recent Orders (last {limit})")
    typer.echo("=" * 60)

    try:
        orders = []
        with open(orders_file, 'r') as f:
            for line in f:
                if line.strip():
                    order_event = json.loads(line)
                    if order_event.get('event') == 'ORDER_NEW':
                        orders.append(order_event.get('order', {}))

        # Filter by status if specified
        if status:
            orders = [o for o in orders if o.get('status') == status.upper()]

        # Show most recent first, limited
        orders = orders[-limit:][::-1]

        if not orders:
            filter_msg = f" with status {status}" if status else ""
            typer.echo(f"No orders found{filter_msg}")
            return

        for order in orders:
            order_id = order.get('order_id', 'Unknown')[:12]
            symbol = order.get('symbol', '???')
            side = order.get('side', '?')
            qty = order.get('quantity', 0)
            status = order.get('status', 'Unknown')
            strategy = order.get('strategy', 'Unknown')

            status_emoji = {
                'NEW': '🟡',
                'PARTIALLY_FILLED': '🟠',
                'FILLED': '🟢',
                'CANCELED': '🔴',
                'REJECTED': '❌'
            }.get(status, '⚪')

            typer.echo(f"{status_emoji} {order_id} {symbol:<8} {side:<4} {qty:>10.2f} [{strategy}]")

    except Exception as e:
        typer.echo(f"❌ Error reading orders: {e}")


@exec_app.command("trades")
def list_trades(
    symbol: Optional[str] = typer.Option(None, "--symbol", help="Filter by symbol"),
    limit: int = typer.Option(10, "--limit", help="Number of trades to show")
):
    """
    List recent trades from execution journal
    """

    trades_file = Path("artifacts/execution/trades.jsonl")

    if not trades_file.exists():
        typer.echo("❌ No trades journal found")
        typer.echo(f"   Expected: {trades_file}")
        typer.echo("   Run an execution simulation first")
        return

    typer.echo(f"💼 Recent Trades (last {limit})")
    typer.echo("=" * 70)

    try:
        trades = []
        with open(trades_file, 'r') as f:
            for line in f:
                if line.strip():
                    trade = json.loads(line)
                    trades.append(trade)

        # Filter by symbol if specified
        if symbol:
            trades = [t for t in trades if t.get('symbol', '').upper() == symbol.upper()]

        # Show most recent first, limited
        trades = trades[-limit:][::-1]

        if not trades:
            filter_msg = f" for {symbol}" if symbol else ""
            typer.echo(f"No trades found{filter_msg}")
            return

        for trade in trades:
            trade_id = trade.get('trade_id', 'Unknown')[:12]
            symbol = trade.get('symbol', '???')
            side = trade.get('side', '?')
            qty = trade.get('quantity', 0)
            price = trade.get('price', 0)
            notional = trade.get('notional', 0)
            slippage = trade.get('slippage_bps', 0)

            side_emoji = '🟢' if side == 'BUY' else '🔴'

            typer.echo(f"{side_emoji} {trade_id} {symbol:<8} {side:<4} {qty:>10.2f} @ ${price:>8.2f} = ${notional:>10.0f} ({slippage:>5.1f}bps)")

    except Exception as e:
        typer.echo(f"❌ Error reading trades: {e}")


@exec_app.command("status")
def execution_status():
    """
    Show execution system status and configuration
    """

    typer.echo("⚙️  Execution System Status")
    typer.echo("=" * 40)
    typer.echo()

    # Check environment
    ally_live = os.environ.get("ALLY_LIVE", "0")
    live_status = "🔴 LIVE" if ally_live == "1" else "🟢 PAPER"
    typer.echo(f"Environment: {live_status}")
    typer.echo()

    # Check artifact directories
    typer.echo("📁 Artifact Directories:")
    directories = [
        "artifacts/execution",
        "artifacts/fixtures/phase10"
    ]

    for directory in directories:
        exists = "✅" if os.path.exists(directory) else "❌"
        typer.echo(f"   {exists} {directory}")
    typer.echo()

    # Check journal files
    typer.echo("📋 Journal Files:")
    journal_files = [
        "artifacts/execution/orders.jsonl",
        "artifacts/execution/trades.jsonl"
    ]

    for file_path in journal_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            typer.echo(f"   ✅ {file_path} ({size} bytes)")
        else:
            typer.echo(f"   ❌ {file_path}")
    typer.echo()

    # Check fixture files
    typer.echo("🧪 Test Fixtures:")
    fixture_files = [
        "artifacts/fixtures/phase10/target_weights.json",
        "artifacts/fixtures/phase10/last_positions.json",
        "artifacts/fixtures/phase10/symbols_meta.csv",
        "artifacts/fixtures/phase10/prices_intraday.csv",
        "artifacts/fixtures/phase10/cost_model.yaml"
    ]

    for file_path in fixture_files:
        exists = "✅" if os.path.exists(file_path) else "❌"
        typer.echo(f"   {exists} {file_path}")
    typer.echo()

    # Show recent simulation artifacts
    simulation_dir = Path("artifacts/execution")
    if simulation_dir.exists():
        simulation_files = list(simulation_dir.glob("simulation_*.json"))
        if simulation_files:
            typer.echo("📊 Recent Simulations:")
            for sim_file in sorted(simulation_files)[-3:]:  # Last 3
                typer.echo(f"   📄 {sim_file.name}")
        else:
            typer.echo("📊 No simulation artifacts found")


if __name__ == "__main__":
    exec_app()