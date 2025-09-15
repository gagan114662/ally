"""
Trading Risk Management - Pre-trade checks and kill-switch for Phase 3

Provides:
- Pre-trade validation: max notional, position limits, price bands
- Session kill-switch: stop trading when PnL breaches limit
- Risk drill: simulate breach scenarios for testing
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from ..tools import register
from ..schemas.base import ToolResult
from ..utils.hashing import hash_payload
from ..utils.receipts import store_tool_receipt


@dataclass
class TradingRiskLimits:
    """Risk limit configuration for trading"""
    max_notional_per_order: float = 100000.0  # Max $ per single order
    max_position_per_symbol: int = 10000      # Max shares per symbol
    max_total_notional: float = 1000000.0     # Max $ total exposure
    price_band_pct: float = 0.10              # 10% price bands
    max_daily_loss: float = 50000.0           # Kill-switch at -$50k
    max_unrealized_loss: float = 25000.0      # Warn at -$25k unrealized
    enabled: bool = True                       # Master risk switch


@dataclass
class TradingRiskState:
    """Current risk state tracking for trading"""
    total_notional: float = 0.0
    positions: Dict[str, int] = field(default_factory=dict)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    orders_rejected: int = 0
    orders_approved: int = 0
    kill_switch_triggered: bool = False
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# Global risk state (in-memory for session)
_trading_risk_limits = TradingRiskLimits()
_trading_risk_state = TradingRiskState()


@register("trading_risk.check_order")
def trading_risk_check_order(**kwargs) -> ToolResult:
    """
    Check if an order passes trading risk limits
    
    Args:
        symbol: Trading symbol
        side: Order side (buy/sell)
        qty: Order quantity
        price: Expected price (for limit orders) or market estimate
        order_type: Order type (market/limit/stop)
        
    Returns:
        ToolResult with approval/rejection and reasons
    """
    try:
        symbol = kwargs.get('symbol', '').upper()
        side = kwargs.get('side', '').lower()
        qty = int(kwargs.get('qty', 0))
        price = float(kwargs.get('price', 0.0))
        order_type = kwargs.get('order_type', 'market')
        
        if not symbol or not side or qty <= 0 or price <= 0:
            return ToolResult.error(["Invalid order parameters"])
        
        # Kill-switch check
        if _trading_risk_state.kill_switch_triggered:
            rejection = _create_rejection_receipt(
                symbol, side, qty, price, 
                reason="Kill-switch triggered - trading halted"
            )
            return ToolResult.success(
                data={
                    "approved": False,
                    "reason": "Kill-switch triggered",
                    "receipt_hash": rejection["receipt_hash"]
                }
            )
        
        # Check if risk is enabled
        if not _trading_risk_limits.enabled:
            # Risk disabled, auto-approve
            approval = _create_approval_receipt(symbol, side, qty, price)
            return ToolResult.success(
                data={
                    "approved": True,
                    "reason": "Risk checks disabled",
                    "receipt_hash": approval["receipt_hash"]
                }
            )
        
        violations = []
        notional = qty * price
        
        # 1. Max notional per order
        if notional > _trading_risk_limits.max_notional_per_order:
            violations.append(f"Order notional ${notional:,.2f} exceeds limit ${_trading_risk_limits.max_notional_per_order:,.2f}")
        
        # 2. Max position per symbol
        current_pos = _trading_risk_state.positions.get(symbol, 0)
        new_pos = current_pos + qty if side == 'buy' else current_pos - qty
        
        if abs(new_pos) > _trading_risk_limits.max_position_per_symbol:
            violations.append(f"Position {new_pos} would exceed limit {_trading_risk_limits.max_position_per_symbol}")
        
        # 3. Max total notional
        new_total = _trading_risk_state.total_notional + notional
        if new_total > _trading_risk_limits.max_total_notional:
            violations.append(f"Total notional ${new_total:,.2f} would exceed limit ${_trading_risk_limits.max_total_notional:,.2f}")
        
        # 4. Price bands (for limit orders)
        if order_type == 'limit' and _get_reference_price(symbol) > 0:
            ref_price = _get_reference_price(symbol)
            band_low = ref_price * (1 - _trading_risk_limits.price_band_pct)
            band_high = ref_price * (1 + _trading_risk_limits.price_band_pct)
            
            if price < band_low or price > band_high:
                violations.append(f"Price ${price:.2f} outside bands [${band_low:.2f}, ${band_high:.2f}]")
        
        # Decision
        if violations:
            _trading_risk_state.orders_rejected += 1
            rejection = _create_rejection_receipt(
                symbol, side, qty, price,
                reason="; ".join(violations)
            )
            # Add to audit log
            print(f"RISK_REJECT proof=receipt:{rejection['receipt_hash']} {symbol} {side} {qty}@{price}")
            return ToolResult.success(
                data={
                    "approved": False,
                    "violations": violations,
                    "receipt_hash": rejection["receipt_hash"]
                }
            )
        else:
            _trading_risk_state.orders_approved += 1
            # Update state for approved order
            _trading_risk_state.positions[symbol] = new_pos
            _trading_risk_state.total_notional = new_total
            _trading_risk_state.last_update = datetime.utcnow().isoformat()
            
            approval = _create_approval_receipt(symbol, side, qty, price)
            # Add to audit log
            print(f"RISK_APPROVE proof=receipt:{approval['receipt_hash']} {symbol} {side} {qty}@{price}")
            return ToolResult.success(
                data={
                    "approved": True,
                    "updated_position": new_pos,
                    "updated_notional": new_total,
                    "receipt_hash": approval["receipt_hash"]
                }
            )
            
    except Exception as e:
        return ToolResult.error([f"Risk check failed: {e}"])


@register("trading_risk.check_pnl")
def trading_risk_check_pnl(**kwargs) -> ToolResult:
    """
    Check PnL against kill-switch limits
    
    Args:
        realized_pnl: Current realized PnL
        unrealized_pnl: Current unrealized PnL
        positions: Current positions dict
        
    Returns:
        ToolResult with kill-switch status
    """
    try:
        realized = float(kwargs.get('realized_pnl', 0.0))
        unrealized = float(kwargs.get('unrealized_pnl', 0.0))
        positions = kwargs.get('positions', {})
        
        # Update state
        _trading_risk_state.realized_pnl = realized
        _trading_risk_state.unrealized_pnl = unrealized
        _trading_risk_state.last_update = datetime.utcnow().isoformat()
        
        if isinstance(positions, dict):
            _trading_risk_state.positions.update(positions)
        
        # Check kill-switch conditions
        total_loss = realized + unrealized
        
        if realized <= -_trading_risk_limits.max_daily_loss:
            _trading_risk_state.kill_switch_triggered = True
            receipt = _create_kill_switch_receipt("realized_loss", realized)
            print(f"KILL_SWITCH proof=receipt:{receipt['receipt_hash']} trigger=realized_loss value={realized}")
            return ToolResult.success(
                data={
                    "kill_switch": True,
                    "reason": f"Realized loss ${-realized:,.2f} exceeds limit ${_trading_risk_limits.max_daily_loss:,.2f}",
                    "action": "HALT_TRADING",
                    "receipt_hash": receipt["receipt_hash"]
                }
            )
        
        if total_loss <= -_trading_risk_limits.max_daily_loss:
            _trading_risk_state.kill_switch_triggered = True
            receipt = _create_kill_switch_receipt("total_loss", total_loss)
            print(f"KILL_SWITCH proof=receipt:{receipt['receipt_hash']} trigger=total_loss value={total_loss}")
            return ToolResult.success(
                data={
                    "kill_switch": True,
                    "reason": f"Total loss ${-total_loss:,.2f} exceeds limit ${_trading_risk_limits.max_daily_loss:,.2f}",
                    "action": "HALT_TRADING",
                    "receipt_hash": receipt["receipt_hash"]
                }
            )
        
        # Warning for unrealized loss
        warning = None
        if unrealized <= -_trading_risk_limits.max_unrealized_loss:
            warning = f"Unrealized loss ${-unrealized:,.2f} exceeds warning level ${_trading_risk_limits.max_unrealized_loss:,.2f}"
        
        return ToolResult.success(
            data={
                "kill_switch": False,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "total_pnl": total_loss,
                "warning": warning
            }
        )
        
    except Exception as e:
        return ToolResult.error([f"PnL check failed: {e}"])


@register("trading_risk.drill")
def trading_risk_drill(**kwargs) -> ToolResult:
    """
    Simulate a risk breach scenario for testing
    
    Args:
        scenario: Type of breach (max_loss, max_position, max_notional)
        reset: Reset risk state after drill
        
    Returns:
        ToolResult with drill results and receipt
    """
    try:
        scenario = kwargs.get('scenario', 'max_loss')
        reset = kwargs.get('reset', True)
        
        # Save current state
        saved_state = {
            "realized_pnl": _trading_risk_state.realized_pnl,
            "unrealized_pnl": _trading_risk_state.unrealized_pnl,
            "kill_switch": _trading_risk_state.kill_switch_triggered,
            "positions": dict(_trading_risk_state.positions)
        }
        
        drill_results = {}
        
        if scenario == 'max_loss':
            # Simulate max loss breach
            _trading_risk_state.realized_pnl = -_trading_risk_limits.max_daily_loss - 1000
            pnl_result = trading_risk_check_pnl(
                realized_pnl=_trading_risk_state.realized_pnl,
                unrealized_pnl=0
            )
            drill_results = {
                "scenario": "max_loss",
                "simulated_loss": _trading_risk_state.realized_pnl,
                "kill_switch_triggered": pnl_result.data.get("kill_switch", False),
                "reason": pnl_result.data.get("reason")
            }
            
        elif scenario == 'max_position':
            # Simulate max position breach
            test_symbol = "TEST"
            order_result = trading_risk_check_order(
                symbol=test_symbol,
                side="buy",
                qty=_trading_risk_limits.max_position_per_symbol + 1000,
                price=100.0
            )
            drill_results = {
                "scenario": "max_position",
                "simulated_qty": _trading_risk_limits.max_position_per_symbol + 1000,
                "order_rejected": not order_result.data.get("approved", False),
                "violations": order_result.data.get("violations", [])
            }
            
        elif scenario == 'max_notional':
            # Simulate max notional breach
            order_result = trading_risk_check_order(
                symbol="TEST",
                side="buy",
                qty=1000,
                price=_trading_risk_limits.max_notional_per_order * 2 / 1000
            )
            drill_results = {
                "scenario": "max_notional",
                "simulated_notional": _trading_risk_limits.max_notional_per_order * 2,
                "order_rejected": not order_result.data.get("approved", False),
                "violations": order_result.data.get("violations", [])
            }
        else:
            return ToolResult.error([f"Unknown scenario: {scenario}"])
        
        # Generate drill receipt
        drill_receipt = _create_drill_receipt(scenario, drill_results)
        print(f"RISK_DRILL proof=receipt:{drill_receipt['receipt_hash']} scenario={scenario}")
        
        # Reset state if requested
        if reset:
            _trading_risk_state.realized_pnl = saved_state["realized_pnl"]
            _trading_risk_state.unrealized_pnl = saved_state["unrealized_pnl"]
            _trading_risk_state.kill_switch_triggered = saved_state["kill_switch"]
            _trading_risk_state.positions = saved_state["positions"]
            drill_results["state_reset"] = True
        
        drill_results["receipt_hash"] = drill_receipt["receipt_hash"]
        
        return ToolResult.success(
            data=drill_results,
            warnings=["This was a drill - state has been reset" if reset else "State not reset"]
        )
        
    except Exception as e:
        return ToolResult.error([f"Risk drill failed: {e}"])


@register("trading_risk.get_state")
def trading_risk_get_state(**kwargs) -> ToolResult:
    """Get current trading risk state and limits"""
    try:
        return ToolResult.success(
            data={
                "state": {
                    "total_notional": _trading_risk_state.total_notional,
                    "positions": dict(_trading_risk_state.positions),
                    "realized_pnl": _trading_risk_state.realized_pnl,
                    "unrealized_pnl": _trading_risk_state.unrealized_pnl,
                    "orders_rejected": _trading_risk_state.orders_rejected,
                    "orders_approved": _trading_risk_state.orders_approved,
                    "kill_switch_triggered": _trading_risk_state.kill_switch_triggered,
                    "last_update": _trading_risk_state.last_update
                },
                "limits": {
                    "max_notional_per_order": _trading_risk_limits.max_notional_per_order,
                    "max_position_per_symbol": _trading_risk_limits.max_position_per_symbol,
                    "max_total_notional": _trading_risk_limits.max_total_notional,
                    "price_band_pct": _trading_risk_limits.price_band_pct,
                    "max_daily_loss": _trading_risk_limits.max_daily_loss,
                    "max_unrealized_loss": _trading_risk_limits.max_unrealized_loss,
                    "enabled": _trading_risk_limits.enabled
                }
            }
        )
    except Exception as e:
        return ToolResult.error([f"Failed to get risk state: {e}"])


@register("trading_risk.reset")
def trading_risk_reset(**kwargs) -> ToolResult:
    """Reset trading risk state (for new session or testing)"""
    try:
        global _trading_risk_state
        _trading_risk_state = TradingRiskState()
        
        return ToolResult.success(
            data={
                "reset": True,
                "message": "Trading risk state reset to initial values"
            }
        )
    except Exception as e:
        return ToolResult.error([f"Failed to reset risk state: {e}"])


# Helper functions

def _get_reference_price(symbol: str) -> float:
    """Get reference price for price band checks (mock for now)"""
    import hashlib
    base_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    return 50.0 + (base_hash % 450)


def _create_approval_receipt(symbol: str, side: str, qty: int, price: float) -> Dict[str, str]:
    """Create approval receipt and store in database"""
    receipt_data = {
        "action": "risk_approval",
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "timestamp": datetime.utcnow().isoformat(),
        "risk_state": {
            "total_notional": _trading_risk_state.total_notional,
            "position": _trading_risk_state.positions.get(symbol, 0)
        }
    }
    
    receipt_hash = hash_payload(receipt_data)[:16]
    
    # Store receipt
    store_tool_receipt(
        tool_name="trading_risk.approve_order",
        inputs={"symbol": symbol, "side": side, "qty": qty, "price": price},
        raw_payload=receipt_data
    )
    
    return {"receipt_hash": receipt_hash, "data": receipt_data}


def _create_rejection_receipt(symbol: str, side: str, qty: int, price: float, reason: str) -> Dict[str, str]:
    """Create rejection receipt and store in database"""
    receipt_data = {
        "action": "risk_rejection",
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    receipt_hash = hash_payload(receipt_data)[:16]
    
    # Store receipt
    store_tool_receipt(
        tool_name="trading_risk.reject_order",
        inputs={"symbol": symbol, "side": side, "qty": qty, "price": price},
        raw_payload=receipt_data
    )
    
    return {"receipt_hash": receipt_hash, "data": receipt_data}


def _create_kill_switch_receipt(trigger: str, value: float) -> Dict[str, str]:
    """Create kill-switch receipt and store in database"""
    receipt_data = {
        "action": "kill_switch_triggered",
        "trigger": trigger,
        "value": value,
        "timestamp": datetime.utcnow().isoformat(),
        "risk_state": {
            "realized_pnl": _trading_risk_state.realized_pnl,
            "unrealized_pnl": _trading_risk_state.unrealized_pnl,
            "positions": dict(_trading_risk_state.positions)
        }
    }
    
    receipt_hash = hash_payload(receipt_data)[:16]
    
    # Store receipt
    store_tool_receipt(
        tool_name="trading_risk.kill_switch",
        inputs={"trigger": trigger, "value": value},
        raw_payload=receipt_data
    )
    
    return {"receipt_hash": receipt_hash, "data": receipt_data}


def _create_drill_receipt(scenario: str, results: Dict[str, Any]) -> Dict[str, str]:
    """Create drill receipt and store in database"""
    receipt_data = {
        "action": "risk_drill",
        "scenario": scenario,
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    receipt_hash = hash_payload(receipt_data)[:16]
    
    # Store receipt
    store_tool_receipt(
        tool_name="trading_risk.drill",
        inputs={"scenario": scenario},
        raw_payload=receipt_data
    )
    
    return {"receipt_hash": receipt_hash, "data": receipt_data}