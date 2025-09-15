"""
Trading Router - Signal processing and order routing with risk management

Consumes YAML/JSON signal streams and routes orders through risk management to broker
"""

import json
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..tools import register
from ..schemas.base import ToolResult
from ..utils.hashing import hash_payload
from ..utils.receipts import store_tool_receipt
from .trading_risk import trading_risk_check_order, trading_risk_check_pnl
from .broker import broker_place_order, broker_get_account, broker_get_positions


@register("trading_router.run_strategy")
def trading_router_run_strategy(**kwargs) -> ToolResult:
    """
    Run a trading strategy from signal file
    
    Args:
        signals_file: Path to YAML/JSON file with trading signals
        backend: Broker backend (simulator/qc_paper)
        live: Enable live trading (requires ALLY_LIVE=1)
        risk_enabled: Enable risk checks (default: True)
        
    Returns:
        ToolResult with strategy execution summary
    """
    try:
        signals_file = kwargs.get('signals_file', 'signals.yaml')
        backend = kwargs.get('backend', 'simulator')
        live = kwargs.get('live', False)
        risk_enabled = kwargs.get('risk_enabled', True)
        
        # Load signals from file
        try:
            signals_path = Path(signals_file)
            if not signals_path.exists():
                return ToolResult.error([f"Signals file not found: {signals_file}"])
            
            with open(signals_path, 'r') as f:
                if signals_path.suffix.lower() in ['.yaml', '.yml']:
                    signals_data = yaml.safe_load(f)
                else:
                    signals_data = json.load(f)
                    
        except Exception as e:
            return ToolResult.error([f"Failed to load signals file: {e}"])
        
        # Extract signals list
        signals = signals_data.get('signals', [])
        if not signals:
            return ToolResult.error(["No signals found in file"])
        
        # Initialize execution state
        execution_summary = {
            "signals_processed": 0,
            "orders_placed": 0,
            "orders_rejected": 0,
            "risk_violations": 0,
            "kill_switch_triggered": False,
            "receipts": [],
            "errors": []
        }
        
        # Process each signal
        for i, signal in enumerate(signals):
            try:
                result = _process_signal(signal, backend, live, risk_enabled, i)
                
                execution_summary["signals_processed"] += 1
                
                if result["success"]:
                    execution_summary["orders_placed"] += 1
                    if result.get("receipt_hash"):
                        execution_summary["receipts"].append(result["receipt_hash"])
                else:
                    execution_summary["orders_rejected"] += 1
                    if result.get("risk_violation"):
                        execution_summary["risk_violations"] += 1
                    if result.get("kill_switch"):
                        execution_summary["kill_switch_triggered"] = True
                        # Stop processing if kill switch triggered
                        break
                
                if result.get("error"):
                    execution_summary["errors"].append(result["error"])
                    
            except Exception as e:
                execution_summary["errors"].append(f"Signal {i}: {e}")
        
        # Store strategy execution receipt
        strategy_receipt = _create_strategy_receipt(signals_file, execution_summary)
        execution_summary["strategy_receipt"] = strategy_receipt["receipt_hash"]
        
        # Add to audit log
        print(f"STRATEGY_RUN proof=receipt:{strategy_receipt['receipt_hash']} signals={execution_summary['signals_processed']} orders={execution_summary['orders_placed']}")
        
        return ToolResult.success(
            data=execution_summary,
            warnings=[f"Kill switch triggered - stopped processing signals"] if execution_summary["kill_switch_triggered"] else []
        )
        
    except Exception as e:
        return ToolResult.error([f"Strategy execution failed: {e}"])


@register("trading_router.process_signal")
def trading_router_process_signal(**kwargs) -> ToolResult:
    """
    Process a single trading signal
    
    Args:
        signal: Signal dict with ts, symbol, action, qty, type, limit, stop
        backend: Broker backend (simulator/qc_paper)
        live: Enable live trading
        risk_enabled: Enable risk checks
        
    Returns:
        ToolResult with signal processing result
    """
    try:
        signal = kwargs.get('signal', {})
        backend = kwargs.get('backend', 'simulator')
        live = kwargs.get('live', False)
        risk_enabled = kwargs.get('risk_enabled', True)
        
        result = _process_signal(signal, backend, live, risk_enabled, 0)
        
        return ToolResult.success(data=result)
        
    except Exception as e:
        return ToolResult.error([f"Signal processing failed: {e}"])


def _process_signal(signal: Dict[str, Any], backend: str, live: bool, risk_enabled: bool, signal_index: int) -> Dict[str, Any]:
    """Process a single signal and return result"""
    
    # Extract signal parameters
    timestamp = signal.get('ts', datetime.utcnow().isoformat())
    symbol = signal.get('symbol', '').upper()
    action = signal.get('action', '').lower()  # buy/sell
    qty = abs(int(signal.get('qty', 0)))
    order_type = signal.get('type', 'market').lower()
    limit_price = signal.get('limit', None)
    stop_price = signal.get('stop', None)
    
    # Validate signal
    if not symbol or not action or qty <= 0:
        return {
            "success": False,
            "error": f"Invalid signal parameters: symbol={symbol}, action={action}, qty={qty}",
            "signal_index": signal_index
        }
    
    if action not in ['buy', 'sell']:
        return {
            "success": False,
            "error": f"Invalid action: {action} (must be buy/sell)",
            "signal_index": signal_index
        }
    
    # Estimate price for risk check
    if order_type == 'limit' and limit_price:
        estimated_price = float(limit_price)
    elif order_type == 'stop' and stop_price:
        estimated_price = float(stop_price)
    else:
        # Use mock market price for risk check
        from .trading_risk import _get_reference_price
        estimated_price = _get_reference_price(symbol)
    
    # Risk check if enabled
    if risk_enabled:
        risk_result = trading_risk_check_order(
            symbol=symbol,
            side=action,
            qty=qty,
            price=estimated_price,
            order_type=order_type
        )
        
        if not risk_result.ok:
            return {
                "success": False,
                "error": f"Risk check failed: {risk_result.errors}",
                "signal_index": signal_index
            }
        
        if not risk_result.data.get("approved", False):
            return {
                "success": False,
                "error": f"Risk check rejected: {risk_result.data.get('reason', 'Unknown')}",
                "violations": risk_result.data.get("violations", []),
                "risk_violation": True,
                "risk_receipt": risk_result.data.get("receipt_hash"),
                "signal_index": signal_index
            }
        
        # Check for kill switch
        if "kill-switch" in risk_result.data.get("reason", "").lower():
            return {
                "success": False,
                "error": "Kill switch triggered",
                "kill_switch": True,
                "risk_receipt": risk_result.data.get("receipt_hash"),
                "signal_index": signal_index
            }
    
    # Place order through broker
    try:
        order_result = broker_place_order(
            symbol=symbol,
            side=action,
            qty=qty,
            type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            backend=backend,
            live=live
        )
        
        if not order_result.ok:
            return {
                "success": False,
                "error": f"Order placement failed: {order_result.errors}",
                "signal_index": signal_index
            }
        
        order_data = order_result.data.get("order", {})
        
        return {
            "success": True,
            "signal_index": signal_index,
            "order_id": order_data.get("order_id"),
            "symbol": symbol,
            "side": action,
            "qty": qty,
            "type": order_type,
            "status": order_data.get("status"),
            "receipt_hash": order_data.get("receipt_hash"),
            "message": f"Order placed: {order_data.get('order_id')}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Order placement exception: {e}",
            "signal_index": signal_index
        }


@register("trading_router.create_sample_signals")
def trading_router_create_sample_signals(**kwargs) -> ToolResult:
    """
    Create sample signals file for testing
    
    Args:
        output_file: Output file path (default: sample_signals.yaml)
        symbols: List of symbols (default: [AAPL, MSFT])
        count: Number of signals (default: 3)
        
    Returns:
        ToolResult with sample file creation status
    """
    try:
        output_file = kwargs.get('output_file', 'sample_signals.yaml')
        symbols = kwargs.get('symbols', ['AAPL', 'MSFT'])
        count = kwargs.get('count', 3)
        
        # Generate sample signals
        signals = []
        for i in range(count):
            symbol = symbols[i % len(symbols)]
            action = 'buy' if i % 2 == 0 else 'sell'
            
            signal = {
                'ts': datetime.utcnow().isoformat(),
                'symbol': symbol,
                'action': action,
                'qty': 100 * (i + 1),
                'type': 'market' if i % 2 == 0 else 'limit',
                'limit': 175.0 + i * 5.0 if i % 2 == 1 else None
            }
            
            # Remove None values
            signal = {k: v for k, v in signal.items() if v is not None}
            signals.append(signal)
        
        # Create signals file content
        signals_data = {
            'strategy': 'sample_strategy',
            'created_at': datetime.utcnow().isoformat(),
            'signals': signals
        }
        
        # Write to file
        with open(output_file, 'w') as f:
            yaml.dump(signals_data, f, default_flow_style=False, sort_keys=False)
        
        return ToolResult.success(
            data={
                "file_created": output_file,
                "signals_count": len(signals),
                "symbols": symbols,
                "sample_signals": signals
            }
        )
        
    except Exception as e:
        return ToolResult.error([f"Failed to create sample signals: {e}"])


@register("trading_router.get_portfolio_status")
def trading_router_get_portfolio_status(**kwargs) -> ToolResult:
    """
    Get current portfolio status with risk metrics
    
    Args:
        backend: Broker backend (simulator/qc_paper)
        live: Whether to get live data
        
    Returns:
        ToolResult with portfolio status and risk metrics
    """
    try:
        backend = kwargs.get('backend', 'simulator')
        live = kwargs.get('live', False)
        
        # Get account info
        account_result = broker_get_account(backend=backend, live=live)
        if not account_result.ok:
            return ToolResult.error([f"Failed to get account: {account_result.errors}"])
        
        account = account_result.data.get("account", {})
        
        # Get positions
        positions_result = broker_get_positions(backend=backend, live=live)
        if not positions_result.ok:
            return ToolResult.error([f"Failed to get positions: {positions_result.errors}"])
        
        positions = positions_result.data.get("positions", [])
        
        # Get risk state
        from .trading_risk import trading_risk_get_state
        risk_result = trading_risk_get_state()
        risk_state = risk_result.data if risk_result.ok else {}
        
        # Calculate portfolio metrics
        total_position_value = sum(pos.get("market_value", 0) for pos in positions)
        unrealized_pnl = sum(pos.get("unrealized_pnl", 0) for pos in positions)
        
        # Update risk with current PnL
        trading_risk_check_pnl(
            realized_pnl=account.get("total_pnl", 0),
            unrealized_pnl=unrealized_pnl,
            positions={pos["symbol"]: pos["qty"] for pos in positions}
        )
        
        portfolio_status = {
            "account": account,
            "positions": positions,
            "risk_state": risk_state.get("state", {}),
            "risk_limits": risk_state.get("limits", {}),
            "summary": {
                "cash": account.get("cash", 0),
                "total_value": account.get("total_value", 0),
                "position_count": len([p for p in positions if p.get("qty", 0) != 0]),
                "total_position_value": total_position_value,
                "unrealized_pnl": unrealized_pnl,
                "kill_switch_active": risk_state.get("state", {}).get("kill_switch_triggered", False)
            }
        }
        
        return ToolResult.success(data=portfolio_status)
        
    except Exception as e:
        return ToolResult.error([f"Failed to get portfolio status: {e}"])


def _create_strategy_receipt(signals_file: str, execution_summary: Dict[str, Any]) -> Dict[str, str]:
    """Create strategy execution receipt and store in database"""
    receipt_data = {
        "action": "strategy_execution",
        "signals_file": signals_file,
        "execution_summary": execution_summary,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    receipt_hash = hash_payload(receipt_data)[:16]
    
    # Store receipt
    store_tool_receipt(
        tool_name="trading_router.run_strategy",
        inputs={"signals_file": signals_file},
        raw_payload=receipt_data
    )
    
    return {"receipt_hash": receipt_hash, "data": receipt_data}


if __name__ == "__main__":
    # Test trading router functionality
    print("🧪 Testing Trading Router...")
    
    # Create sample signals
    sample_result = trading_router_create_sample_signals(
        output_file="test_signals.yaml",
        symbols=["AAPL", "MSFT"],
        count=3
    )
    print(f"Sample creation: {sample_result.ok}")
    
    # Process single signal
    test_signal = {
        "symbol": "AAPL",
        "action": "buy", 
        "qty": 100,
        "type": "market"
    }
    
    signal_result = trading_router_process_signal(
        signal=test_signal,
        backend="simulator",
        live=False
    )
    print(f"Signal processing: {signal_result.ok}")
    if signal_result.ok:
        print(f"Result: {signal_result.data}")
    else:
        print(f"Errors: {signal_result.errors}")