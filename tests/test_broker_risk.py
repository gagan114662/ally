#!/usr/bin/env python3
"""
Broker risk integration tests - risk management with broker operations
"""

import os
import pytest

# Ensure CI stays dry
os.environ["ALLY_LIVE"] = "0"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_risk_check_order_approval():
    """Test risk check approves valid orders"""
    from ally.tools.trading_risk import trading_risk_check_order, trading_risk_reset
    
    # Reset risk state for clean test
    trading_risk_reset()
    
    result = trading_risk_check_order(
        symbol="AAPL",
        side="buy",
        qty=100,
        price=175.0,
        order_type="market"
    )
    
    assert result.ok == True, f"Risk check should succeed: {result.errors}"
    assert result.data["approved"] == True, "Order should be approved"
    assert "receipt_hash" in result.data, "Should have receipt hash"
    assert "updated_position" in result.data, "Should update position"
    assert "updated_notional" in result.data, "Should update notional"


def test_risk_check_order_rejection():
    """Test risk check rejects orders exceeding limits"""
    from ally.tools.trading_risk import trading_risk_check_order, trading_risk_reset
    
    # Reset risk state
    trading_risk_reset()
    
    # Order exceeding max notional per order (default: $100k)
    result = trading_risk_check_order(
        symbol="AAPL",
        side="buy",
        qty=1000,
        price=200.0,  # $200k notional > $100k limit
        order_type="limit"
    )
    
    assert result.ok == True, "Should return result (not error)"
    assert result.data["approved"] == False, "Order should be rejected"
    assert "violations" in result.data, "Should list violations"
    assert len(result.data["violations"]) > 0, "Should have violation details"
    assert "receipt_hash" in result.data, "Should have rejection receipt"


def test_risk_kill_switch_trigger():
    """Test kill switch triggers on max loss"""
    from ally.tools.trading_risk import trading_risk_check_pnl, trading_risk_reset
    
    # Reset risk state
    trading_risk_reset()
    
    # Trigger kill switch with large loss (default limit: -$50k)
    result = trading_risk_check_pnl(
        realized_pnl=-60000.0,  # Exceeds -$50k limit
        unrealized_pnl=0.0
    )
    
    assert result.ok == True, f"PnL check should succeed: {result.errors}"
    assert result.data["kill_switch"] == True, "Kill switch should be triggered"
    assert "HALT_TRADING" in result.data["action"], "Should halt trading"
    assert "receipt_hash" in result.data, "Should have kill switch receipt"


def test_risk_kill_switch_blocks_orders():
    """Test that kill switch blocks subsequent orders"""
    from ally.tools.trading_risk import (
        trading_risk_check_pnl, 
        trading_risk_check_order, 
        trading_risk_reset
    )
    
    # Reset and trigger kill switch
    trading_risk_reset()
    trading_risk_check_pnl(realized_pnl=-60000.0, unrealized_pnl=0.0)
    
    # Try to place order after kill switch
    result = trading_risk_check_order(
        symbol="AAPL",
        side="buy",
        qty=10,
        price=100.0,
        order_type="market"
    )
    
    assert result.ok == True, "Should return result"
    assert result.data["approved"] == False, "Order should be rejected"
    assert "kill-switch" in result.data["reason"].lower(), "Should mention kill switch"


def test_risk_drill_max_loss():
    """Test risk drill for max loss scenario"""
    from ally.tools.trading_risk import trading_risk_drill, trading_risk_reset
    
    # Reset state
    trading_risk_reset()
    
    result = trading_risk_drill(
        scenario="max_loss",
        reset=True
    )
    
    assert result.ok == True, f"Risk drill should succeed: {result.errors}"
    assert result.data["scenario"] == "max_loss", "Should simulate max loss"
    assert result.data["kill_switch_triggered"] == True, "Should trigger kill switch in drill"
    assert result.data["state_reset"] == True, "Should reset state after drill"
    assert "receipt_hash" in result.data, "Should have drill receipt"


def test_risk_drill_max_position():
    """Test risk drill for max position scenario"""
    from ally.tools.trading_risk import trading_risk_drill, trading_risk_reset
    
    # Reset state
    trading_risk_reset()
    
    result = trading_risk_drill(
        scenario="max_position",
        reset=True
    )
    
    assert result.ok == True, f"Position drill should succeed: {result.errors}"
    assert result.data["scenario"] == "max_position", "Should simulate position breach"
    assert result.data["order_rejected"] == True, "Should reject oversized order"
    assert len(result.data["violations"]) > 0, "Should have violation details"


def test_risk_drill_max_notional():
    """Test risk drill for max notional scenario"""
    from ally.tools.trading_risk import trading_risk_drill, trading_risk_reset
    
    # Reset state
    trading_risk_reset()
    
    result = trading_risk_drill(
        scenario="max_notional",
        reset=True
    )
    
    assert result.ok == True, f"Notional drill should succeed: {result.errors}"
    assert result.data["scenario"] == "max_notional", "Should simulate notional breach"
    assert result.data["order_rejected"] == True, "Should reject large notional order"
    assert len(result.data["violations"]) > 0, "Should have violation details"


def test_risk_state_management():
    """Test risk state tracking and updates"""
    from ally.tools.trading_risk import (
        trading_risk_get_state,
        trading_risk_check_order,
        trading_risk_reset
    )
    
    # Reset and get initial state
    trading_risk_reset()
    initial_state = trading_risk_get_state()
    
    assert initial_state.ok == True, "Should get state successfully"
    assert initial_state.data["state"]["orders_approved"] == 0, "Should start with 0 approved"
    assert initial_state.data["state"]["orders_rejected"] == 0, "Should start with 0 rejected"
    
    # Approve an order
    trading_risk_check_order(
        symbol="AAPL",
        side="buy", 
        qty=50,
        price=150.0
    )
    
    # Check updated state
    updated_state = trading_risk_get_state()
    assert updated_state.data["state"]["orders_approved"] == 1, "Should have 1 approved order"
    assert "AAPL" in updated_state.data["state"]["positions"], "Should track AAPL position"
    assert updated_state.data["state"]["positions"]["AAPL"] == 50, "Should have 50 share position"


def test_risk_price_bands():
    """Test price band validation for limit orders"""
    from ally.tools.trading_risk import trading_risk_check_order, trading_risk_reset, _get_reference_price
    
    # Reset state
    trading_risk_reset()
    
    # Get reference price for AAPL
    ref_price = _get_reference_price("AAPL")
    
    # Test order within price bands (should pass)
    within_bands_result = trading_risk_check_order(
        symbol="AAPL",
        side="buy",
        qty=10,
        price=ref_price * 1.05,  # 5% above reference (within 10% band)
        order_type="limit"
    )
    
    assert within_bands_result.ok == True, "Should succeed for in-band price"
    assert within_bands_result.data["approved"] == True, "In-band order should be approved"
    
    # Test order outside price bands (should fail)
    outside_bands_result = trading_risk_check_order(
        symbol="AAPL", 
        side="buy",
        qty=10,
        price=ref_price * 1.5,  # 50% above reference (outside 10% band)
        order_type="limit"
    )
    
    assert outside_bands_result.ok == True, "Should return result"
    assert outside_bands_result.data["approved"] == False, "Out-of-band order should be rejected"
    assert any("outside bands" in v.lower() for v in outside_bands_result.data.get("violations", [])), "Should mention price bands"


def test_risk_position_accumulation():
    """Test position tracking across multiple orders"""
    from ally.tools.trading_risk import trading_risk_check_order, trading_risk_reset
    
    # Reset state
    trading_risk_reset()
    
    # Place multiple buy orders
    symbols = ["AAPL", "MSFT"] 
    for i in range(3):
        for symbol in symbols:
            result = trading_risk_check_order(
                symbol=symbol,
                side="buy",
                qty=100,
                price=150.0,
                order_type="market"
            )
            assert result.data["approved"] == True, f"Order {i} for {symbol} should be approved"
    
    # Check final state
    from ally.tools.trading_risk import trading_risk_get_state
    state = trading_risk_get_state()
    
    assert state.data["state"]["positions"]["AAPL"] == 300, "AAPL position should be 300"
    assert state.data["state"]["positions"]["MSFT"] == 300, "MSFT position should be 300"
    assert state.data["state"]["orders_approved"] == 6, "Should have 6 approved orders"


def test_risk_pnl_warning_levels():
    """Test PnL warning levels without triggering kill switch"""
    from ally.tools.trading_risk import trading_risk_check_pnl, trading_risk_reset
    
    # Reset state
    trading_risk_reset()
    
    # Test unrealized loss warning (default: -$25k warning, -$50k kill switch)
    result = trading_risk_check_pnl(
        realized_pnl=0.0,
        unrealized_pnl=-30000.0  # Exceeds warning but not kill switch
    )
    
    assert result.ok == True, "PnL check should succeed"
    assert result.data["kill_switch"] == False, "Should not trigger kill switch"
    assert result.data["warning"] is not None, "Should have warning message"
    assert "unrealized loss" in result.data["warning"].lower(), "Warning should mention unrealized loss"


def test_risk_invalid_parameters():
    """Test risk checks with invalid parameters"""
    from ally.tools.trading_risk import trading_risk_check_order
    
    # Test missing symbol
    result = trading_risk_check_order(
        symbol="",
        side="buy",
        qty=100,
        price=150.0
    )
    
    assert result.ok == False, "Should fail with empty symbol"
    assert "invalid" in str(result.errors).lower(), "Should report invalid parameters"
    
    # Test zero quantity
    result = trading_risk_check_order(
        symbol="AAPL",
        side="buy", 
        qty=0,
        price=150.0
    )
    
    assert result.ok == False, "Should fail with zero quantity"
    
    # Test negative price
    result = trading_risk_check_order(
        symbol="AAPL",
        side="buy",
        qty=100,
        price=-50.0
    )
    
    assert result.ok == False, "Should fail with negative price"


def test_risk_receipts_generated():
    """Test that all risk operations generate receipts"""
    from ally.tools.trading_risk import (
        trading_risk_check_order,
        trading_risk_check_pnl, 
        trading_risk_drill,
        trading_risk_reset
    )
    
    # Reset state
    trading_risk_reset()
    
    # Test approval receipt
    approval = trading_risk_check_order(
        symbol="AAPL",
        side="buy",
        qty=10,
        price=100.0
    )
    assert "receipt_hash" in approval.data, "Approval should have receipt"
    assert len(approval.data["receipt_hash"]) == 16, "Receipt hash should be 16 chars"
    
    # Test rejection receipt
    rejection = trading_risk_check_order(
        symbol="TEST",
        side="buy",
        qty=10000,  # Large quantity to trigger rejection
        price=1000.0
    )
    assert "receipt_hash" in rejection.data, "Rejection should have receipt"
    
    # Test kill switch receipt
    kill_switch = trading_risk_check_pnl(realized_pnl=-60000.0, unrealized_pnl=0.0)
    assert "receipt_hash" in kill_switch.data, "Kill switch should have receipt"
    
    # Test drill receipt
    drill = trading_risk_drill(scenario="max_loss", reset=True)
    assert "receipt_hash" in drill.data, "Drill should have receipt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])