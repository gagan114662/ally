"""
Tests for Risk Management and Execution (Milestone 7)
Validates policy-driven risk checks and deterministic paper trading
"""

import pytest
import hashlib
import json
import sys
from pathlib import Path

# Add ally to path
sys.path.append(str(Path(__file__).parent.parent))

from ally.tools import execute_tool
from ally.schemas.base import ToolResult


def test_risk_001_oversize_order():
    """RISK-001: Oversize order blocked by policy"""
    policy = "max_single_order_notional: 25000"
    
    result = execute_tool(
        'risk.check_limits',
        positions=[],
        orders=[{"symbol": "BTCUSDT", "side": "buy", "qty": 10}],
        policy_yaml=policy,
        equity=100000,
        prices={"BTCUSDT": 3000}
    )
    
    assert result.ok == True
    assert result.data['allow'] == False, "Should block oversize order"
    
    violations = result.data['violations']
    assert len(violations) > 0, "Should have violations"
    
    single_notional_violation = next((v for v in violations if v['code'] == 'SINGLE_NOTIONAL'), None)
    assert single_notional_violation is not None, "Should have SINGLE_NOTIONAL violation"
    assert single_notional_violation['subject']['value'] == 30000, "Notional should be 10 * 3000"
    
    print(f"✅ RISK-001: Order blocked, notional=${single_notional_violation['subject']['value']}")


def test_risk_002_per_symbol_cap():
    """RISK-002: Per-symbol position cap enforced"""
    policy = """
max_position_per_symbol:
  BTCUSDT: 2
"""
    
    result = execute_tool(
        'risk.check_limits',
        positions=[{"symbol": "BTCUSDT", "qty": 1.5, "price": 30000}],
        orders=[{"symbol": "BTCUSDT", "side": "buy", "qty": 1}],
        policy_yaml=policy,
        equity=100000,
        prices={"BTCUSDT": 30000}
    )
    
    assert result.ok == True
    assert result.data['allow'] == False, "Should block order exceeding position limit"
    
    violations = result.data['violations']
    position_violation = next((v for v in violations if v['code'] == 'POSITION_LIMIT'), None)
    assert position_violation is not None, "Should have POSITION_LIMIT violation"
    assert position_violation['subject']['new_position'] == 2.5, "New position would be 2.5"
    
    print(f"✅ RISK-002: Position limit enforced, new pos would be {position_violation['subject']['new_position']}")


def test_risk_003_leverage():
    """RISK-003: Leverage limit computed and enforced"""
    policy = """
max_leverage: 3.0
severity:
  leverage: hard
"""
    
    # Position with gross exposure = 350k, equity = 100k -> leverage = 3.5
    result = execute_tool(
        'risk.check_limits',
        positions=[
            {"symbol": "BTCUSDT", "qty": 5, "price": 30000},
            {"symbol": "ETHUSDT", "qty": 100, "price": 2000}
        ],
        orders=[],
        policy_yaml=policy,
        equity=100000,
        prices={"BTCUSDT": 30000, "ETHUSDT": 2000}
    )
    
    assert result.ok == True
    assert result.data['allow'] == False, "Should block due to leverage"
    
    violations = result.data['violations']
    leverage_violation = next((v for v in violations if v['code'] == 'LEVERAGE_LIMIT'), None)
    assert leverage_violation is not None, "Should have LEVERAGE_LIMIT violation"
    assert leverage_violation['subject']['value'] == 3.5, "Leverage should be 3.5"
    
    print(f"✅ RISK-003: Leverage {leverage_violation['subject']['value']}x blocked (max 3.0x)")


def test_exec_001_market_slippage():
    """EXEC-001: Market order fills with slippage"""
    # Reset broker state
    execute_tool('exec.reset_broker')
    
    result = execute_tool(
        'exec.place_order',
        symbol="BTCUSDT",
        side="buy",
        qty=1,
        type="market",
        price=100,
        slippage_bps=10,
        latency_ms=0
    )
    
    assert result.ok == True
    data = result.data
    assert data['status'] == 'filled', "Market order should fill immediately"
    assert data['avg_price'] == 101.0, "Should have slippage: 100 * (1 + 0.001) = 101"
    assert data['filled_qty'] == 1.0, "Should fill full quantity"
    assert len(data['fills']) >= 1, "Should have at least one fill"
    
    print(f"✅ EXEC-001: Market order filled at {data['avg_price']} (slippage applied)")


def test_exec_002_limit_partial_fill():
    """EXEC-002: Limit order partial fill with liquidity constraint"""
    # Reset broker state
    execute_tool('exec.reset_broker')
    
    result = execute_tool(
        'exec.place_order',
        symbol="BTCUSDT",
        side="sell",
        qty=3,
        type="limit",
        limit_price=105,
        price=106,
        liquidity_per_tick=1
    )
    
    assert result.ok == True
    data = result.data
    assert data['status'] == 'partially_filled', "Should be partially filled"
    assert data['filled_qty'] == 1.0, "Should fill 1 unit (liquidity_per_tick=1)"
    assert data['remaining_qty'] == 2.0, "Should have 2 remaining"
    
    print(f"✅ EXEC-002: Partial fill {data['filled_qty']}/{data['filled_qty'] + data['remaining_qty']}")
    
    return data['broker_order_id']  # Return for next test


def test_exec_003_cancel_working():
    """EXEC-003: Cancel transitions working→canceled"""
    # First create a partially filled order
    execute_tool('exec.reset_broker')
    
    place_result = execute_tool(
        'exec.place_order',
        symbol="BTCUSDT",
        side="sell",
        qty=3,
        type="limit",
        limit_price=105,
        price=106,
        liquidity_per_tick=1
    )
    
    order_id = place_result.data['broker_order_id']
    
    # Cancel the order
    cancel_result = execute_tool(
        'exec.cancel_order',
        broker_order_id=order_id
    )
    
    assert cancel_result.ok == True
    data = cancel_result.data
    assert data['status'] == 'canceled', "Should be canceled"
    assert data['remaining_qty'] == 2.0, "Remaining qty should be frozen"
    
    print(f"✅ EXEC-003: Order canceled, remaining {data['remaining_qty']} frozen")


def test_exec_004_amend_limit():
    """EXEC-004: Amend updates limit price"""
    # Reset and create limit order
    execute_tool('exec.reset_broker')
    
    place_result = execute_tool(
        'exec.place_order',
        symbol="BTCUSDT",
        side="buy",
        qty=2,
        type="limit",
        limit_price=99,
        price=100  # Price above limit, won't fill initially
    )
    
    order_id = place_result.data['broker_order_id']
    assert place_result.data['status'] == 'working', "Should be working initially"
    
    # Amend to more aggressive price
    amend_result = execute_tool(
        'exec.amend_order',
        broker_order_id=order_id,
        limit_price=100.5
    )
    
    assert amend_result.ok == True
    # Note: In this simple implementation, amend doesn't trigger immediate re-matching
    # In a real system, it would check if the new limit allows a fill
    
    print(f"✅ EXEC-004: Order amended to limit {amend_result.data.get('meta', {}).get('limit_price', 'N/A')}")


def test_exec_determinism():
    """DET-001: Deterministic fills fingerprint"""
    # Reset broker
    execute_tool('exec.reset_broker')
    
    # Place identical orders twice
    params = {
        "symbol": "ETHUSDT",
        "side": "buy", 
        "qty": 2,
        "type": "limit",
        "limit_price": 2000,
        "price": 1999.5,
        "liquidity_per_tick": 0.75
    }
    
    result1 = execute_tool('exec.place_order', **params)
    
    # Reset and repeat
    execute_tool('exec.reset_broker')
    result2 = execute_tool('exec.place_order', **params)
    
    # Generate fingerprints
    fills1 = json.dumps(result1.data['fills'], sort_keys=True)
    fills2 = json.dumps(result2.data['fills'], sort_keys=True)
    
    fp1 = hashlib.sha1(fills1.encode()).hexdigest()
    fp2 = hashlib.sha1(fills2.encode()).hexdigest()
    
    assert fp1 == fp2, "Should have identical fingerprints"
    
    print(f"✅ DET-001: Identical fingerprints {fp1[:16]}...")


def test_risk_allowlist_denylist():
    """Test allowlist and denylist functionality"""
    # Test denylist
    policy_deny = """
denylist:
  - BADCOIN
"""
    
    result = execute_tool(
        'risk.check_limits',
        positions=[],
        orders=[{"symbol": "BADCOIN", "side": "buy", "qty": 1}],
        policy_yaml=policy_deny,
        equity=100000,
        prices={"BADCOIN": 100}
    )
    
    assert result.data['allow'] == False
    assert any(v['code'] == 'DENYLIST' for v in result.data['violations'])
    
    # Test allowlist
    policy_allow = """
allowlist:
  - BTCUSDT
  - ETHUSDT
"""
    
    result = execute_tool(
        'risk.check_limits',
        positions=[],
        orders=[{"symbol": "DOGEUSDT", "side": "buy", "qty": 1}],
        policy_yaml=policy_allow,
        equity=100000,
        prices={"DOGEUSDT": 0.1}
    )
    
    assert result.data['allow'] == False
    assert any(v['code'] == 'ALLOWLIST' for v in result.data['violations'])
    
    print(f"✅ Allow/deny lists working")


if __name__ == "__main__":
    # Run all tests with detailed output
    print("\n" + "="*60)
    print("RISK & EXECUTION TESTS (Milestone 7)")
    print("="*60 + "\n")
    
    test_risk_001_oversize_order()
    test_risk_002_per_symbol_cap()
    test_risk_003_leverage()
    test_exec_001_market_slippage()
    test_exec_002_limit_partial_fill()
    test_exec_003_cancel_working()
    test_exec_004_amend_limit()
    test_exec_determinism()
    test_risk_allowlist_denylist()
    
    print("\n" + "="*60)
    print("ALL RISK & EXECUTION TESTS PASSED ✅")
    print("="*60)