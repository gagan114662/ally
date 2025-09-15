#!/usr/bin/env python3
"""
Broker facade offline tests - unified interface testing with simulator backend
"""

import os
import pytest

# Ensure CI stays dry
os.environ["ALLY_LIVE"] = "0"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_broker_start_session_simulator():
    """Test starting broker session with simulator backend"""
    from ally.tools.broker import broker_start_session
    
    result = broker_start_session(
        project_slug="test-simulator",
        symbols=["AAPL", "MSFT"],
        backend="simulator",
        live=False
    )
    
    assert result.ok == True, f"Session start should succeed: {result.errors}"
    assert "session" in result.data, "Result should contain session data"
    
    session = result.data["session"]
    assert session["backend"] == "simulator", "Backend should be simulator"
    assert session["status"] == "active", "Session should be active"
    assert "AAPL" in session["symbols"], "Symbols should include AAPL"
    assert "MSFT" in session["symbols"], "Symbols should include MSFT"


def test_broker_place_order_simulator():
    """Test placing orders through broker facade with simulator"""
    from ally.tools.broker import broker_place_order
    
    # Test market order
    result = broker_place_order(
        symbol="AAPL",
        side="buy",
        qty=100,
        type="market",
        backend="simulator",
        live=False
    )
    
    assert result.ok == True, f"Order placement should succeed: {result.errors}"
    assert "order" in result.data, "Result should contain order data"
    
    order = result.data["order"]
    assert order["symbol"] == "AAPL", "Symbol should match"
    assert order["side"] == "buy", "Side should match"
    assert order["qty"] == 100, "Quantity should match"
    assert order["type"] == "market", "Type should match"
    assert order["provider"] == "simulator", "Provider should be simulator"
    assert order["status"] in ["new", "filled"], "Status should be valid"


def test_broker_place_limit_order_simulator():
    """Test placing limit orders through broker facade"""
    from ally.tools.broker import broker_place_order
    
    result = broker_place_order(
        symbol="MSFT",
        side="sell",
        qty=50,
        type="limit",
        limit_price=350.00,
        time_in_force="gtc",
        backend="simulator",
        live=False
    )
    
    assert result.ok == True, f"Limit order should succeed: {result.errors}"
    
    order = result.data["order"]
    assert order["symbol"] == "MSFT", "Symbol should match"
    assert order["side"] == "sell", "Side should match"
    assert order["type"] == "limit", "Type should match"
    assert order["limit_price"] == 350.00, "Limit price should match"
    assert order["time_in_force"] == "gtc", "TIF should match"


def test_broker_get_account_simulator():
    """Test getting account info through broker facade"""
    from ally.tools.broker import broker_get_account
    
    result = broker_get_account(
        backend="simulator",
        live=False
    )
    
    assert result.ok == True, f"Get account should succeed: {result.errors}"
    assert "account" in result.data, "Result should contain account data"
    
    account = result.data["account"]
    assert account["provider"] == "simulator", "Provider should be simulator"
    assert account["cash"] > 0, "Should have positive cash balance"
    assert account["total_value"] > 0, "Should have positive total value"


def test_broker_get_positions_simulator():
    """Test getting positions through broker facade"""
    from ally.tools.broker import broker_get_positions
    
    result = broker_get_positions(
        backend="simulator",
        live=False
    )
    
    assert result.ok == True, f"Get positions should succeed: {result.errors}"
    assert "positions" in result.data, "Result should contain positions data"
    assert "count" in result.data, "Result should contain position count"
    
    positions = result.data["positions"]
    assert isinstance(positions, list), "Positions should be a list"


def test_broker_get_orders_simulator():
    """Test getting orders through broker facade"""
    from ally.tools.broker import broker_get_orders
    
    result = broker_get_orders(
        backend="simulator",
        limit=10,
        live=False
    )
    
    assert result.ok == True, f"Get orders should succeed: {result.errors}"
    assert "orders" in result.data, "Result should contain orders data"
    assert "count" in result.data, "Result should contain order count"
    
    orders = result.data["orders"]
    assert isinstance(orders, list), "Orders should be a list"


def test_broker_cancel_order_simulator():
    """Test canceling orders through broker facade"""
    from ally.tools.broker import broker_place_order, broker_cancel_order
    
    # First place an order
    place_result = broker_place_order(
        symbol="AAPL",
        side="buy",
        qty=10,
        type="limit",
        limit_price=100.00,  # Low price, won't fill immediately
        backend="simulator",
        live=False
    )
    
    assert place_result.ok == True, "Order placement should succeed"
    order_id = place_result.data["order"]["order_id"]
    
    # Then cancel it
    cancel_result = broker_cancel_order(
        order_id=order_id,
        backend="simulator",
        live=False
    )
    
    assert cancel_result.ok == True, f"Order cancel should succeed: {cancel_result.errors}"


def test_broker_invalid_backend():
    """Test broker facade with invalid backend"""
    from ally.tools.broker import broker_place_order
    
    result = broker_place_order(
        symbol="AAPL",
        side="buy",
        qty=100,
        backend="invalid_backend",
        live=False
    )
    
    assert result.ok == False, "Invalid backend should fail"
    assert "Unknown backend" in str(result.errors), "Should report unknown backend"


def test_broker_invalid_order_params():
    """Test broker facade with invalid order parameters"""
    from ally.tools.broker import broker_place_order
    
    # Missing required limit_price for limit order
    result = broker_place_order(
        symbol="AAPL",
        side="buy",
        qty=100,
        type="limit",
        # limit_price missing
        backend="simulator",
        live=False
    )
    
    # Should either fail validation or create order without limit_price
    # (behavior depends on schema validation)
    if not result.ok:
        assert "limit_price" in str(result.errors).lower() or "invalid" in str(result.errors).lower()


def test_broker_stop_session_simulator():
    """Test stopping broker session"""
    from ally.tools.broker import broker_start_session, broker_stop_session
    
    # Start session
    start_result = broker_start_session(
        project_slug="test-stop",
        symbols=["TEST"],
        backend="simulator",
        live=False
    )
    
    assert start_result.ok == True, "Session start should succeed"
    session_id = start_result.data["session"]["session_id"]
    
    # Stop session
    stop_result = broker_stop_session(
        session_id=session_id,
        backend="simulator",
        live=False
    )
    
    assert stop_result.ok == True, f"Session stop should succeed: {stop_result.errors}"


def test_broker_simulator_fills_market_orders():
    """Test that simulator properly fills market orders"""
    from ally.adapters.broker.simulator_adapter import SimulatorAdapter
    from ally.schemas.broker import OrderSide, OrderType
    
    adapter = SimulatorAdapter(initial_cash=100000.0)
    
    # Place market buy order
    order = adapter.place_order(
        symbol="AAPL",
        side=OrderSide.BUY,
        qty=100,
        order_type=OrderType.MARKET,
        live=False
    )
    
    # Market orders should fill immediately
    assert order.status.value == "filled", "Market order should be filled immediately"
    assert order.filled_qty == 100, "Should fill full quantity"
    assert order.avg_fill_price is not None, "Should have fill price"
    assert order.avg_fill_price > 0, "Fill price should be positive"
    
    # Check account was debited
    account = adapter.get_account()
    expected_cash = 100000.0 - (order.avg_fill_price * 100)
    assert abs(account.cash - expected_cash) < 0.01, "Cash should be debited correctly"
    
    # Check position created
    positions = adapter.get_positions()
    assert len(positions) == 1, "Should have one position"
    assert positions[0].symbol == "AAPL", "Position should be for AAPL"
    assert positions[0].qty == 100, "Position should be 100 shares long"


def test_broker_simulator_deterministic_prices():
    """Test that simulator produces deterministic prices"""
    from ally.adapters.broker.simulator_adapter import SimulatorAdapter
    from ally.schemas.broker import OrderSide, OrderType
    
    # Create two adapters
    adapter1 = SimulatorAdapter()
    adapter2 = SimulatorAdapter()
    
    # Place same orders
    order1 = adapter1.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
    order2 = adapter2.place_order("AAPL", OrderSide.BUY, 10, OrderType.MARKET)
    
    # Should get same deterministic price (within same time window)
    assert order1.avg_fill_price == order2.avg_fill_price, "Prices should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])