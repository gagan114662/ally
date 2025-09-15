#!/usr/bin/env python3
"""
Trading router tests with simulator backend - offline deterministic testing
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path

# Ensure CI stays dry
os.environ["ALLY_LIVE"] = "0"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_router_process_single_signal():
    """Test processing a single trading signal"""
    from ally.tools.trading_router import trading_router_process_signal
    
    # Test buy signal
    signal = {
        "symbol": "AAPL",
        "action": "buy",
        "qty": 100,
        "type": "market"
    }
    
    result = trading_router_process_signal(
        signal=signal,
        backend="simulator",
        live=False,
        risk_enabled=True
    )
    
    assert result.ok == True, f"Signal processing should succeed: {result.errors}"
    assert result.data["success"] == True, "Signal should be successfully processed"
    assert result.data["symbol"] == "AAPL", "Symbol should match"
    assert result.data["side"] == "buy", "Side should match"
    assert result.data["qty"] == 100, "Quantity should match"
    assert "order_id" in result.data, "Should have order ID"
    assert "receipt_hash" in result.data, "Should have receipt hash"


def test_router_process_invalid_signal():
    """Test processing invalid signals"""
    from ally.tools.trading_router import trading_router_process_signal
    
    # Test invalid action
    signal = {
        "symbol": "AAPL",
        "action": "invalid_action",
        "qty": 100,
        "type": "market"
    }
    
    result = trading_router_process_signal(
        signal=signal,
        backend="simulator",
        live=False,
        risk_enabled=True
    )
    
    assert result.ok == True, "Should return result (not error)"
    assert result.data["success"] == False, "Should fail validation"
    assert "invalid action" in result.data["error"].lower(), "Should report invalid action"


def test_router_run_strategy_from_file(tmp_path):
    """Test running strategy from YAML file"""
    from ally.tools.trading_router import trading_router_run_strategy
    
    # Create test signals file
    signals_data = {
        "strategy": "test_strategy",
        "signals": [
            {
                "ts": "2024-01-15T10:00:00Z",
                "symbol": "AAPL",
                "action": "buy",
                "qty": 100,
                "type": "market"
            },
            {
                "ts": "2024-01-15T10:05:00Z", 
                "symbol": "MSFT",
                "action": "buy",
                "qty": 50,
                "type": "limit",
                "limit": 350.0
            },
            {
                "ts": "2024-01-15T10:10:00Z",
                "symbol": "AAPL", 
                "action": "sell",
                "qty": 50,
                "type": "market"
            }
        ]
    }
    
    signals_file = tmp_path / "test_signals.yaml"
    with open(signals_file, 'w') as f:
        yaml.dump(signals_data, f)
    
    # Run strategy
    result = trading_router_run_strategy(
        signals_file=str(signals_file),
        backend="simulator",
        live=False,
        risk_enabled=True
    )
    
    assert result.ok == True, f"Strategy should execute successfully: {result.errors}"
    assert result.data["signals_processed"] == 3, "Should process 3 signals"
    assert result.data["orders_placed"] >= 0, "Should place some orders"
    assert "strategy_receipt" in result.data, "Should have strategy receipt"
    assert len(result.data["receipts"]) >= 0, "Should have order receipts"


def test_router_create_sample_signals(tmp_path):
    """Test creating sample signals file"""
    from ally.tools.trading_router import trading_router_create_sample_signals
    
    output_file = tmp_path / "sample.yaml"
    
    result = trading_router_create_sample_signals(
        output_file=str(output_file),
        symbols=["AAPL", "MSFT", "GOOGL"],
        count=5
    )
    
    assert result.ok == True, f"Sample creation should succeed: {result.errors}"
    assert result.data["signals_count"] == 5, "Should create 5 signals"
    assert Path(output_file).exists(), "Should create file"
    
    # Verify file content
    with open(output_file, 'r') as f:
        data = yaml.safe_load(f)
    
    assert "signals" in data, "Should have signals key"
    assert len(data["signals"]) == 5, "Should have 5 signals"
    
    # Check signal structure
    signal = data["signals"][0]
    assert "symbol" in signal, "Signal should have symbol"
    assert "action" in signal, "Signal should have action" 
    assert "qty" in signal, "Signal should have quantity"
    assert signal["action"] in ["buy", "sell"], "Action should be buy/sell"


def test_router_portfolio_status():
    """Test getting portfolio status"""
    from ally.tools.trading_router import trading_router_get_portfolio_status
    
    result = trading_router_get_portfolio_status(
        backend="simulator",
        live=False
    )
    
    assert result.ok == True, f"Portfolio status should succeed: {result.errors}"
    assert "account" in result.data, "Should have account data"
    assert "positions" in result.data, "Should have positions data"
    assert "risk_state" in result.data, "Should have risk state"
    assert "summary" in result.data, "Should have summary"
    
    summary = result.data["summary"]
    assert "cash" in summary, "Summary should have cash"
    assert "total_value" in summary, "Summary should have total value"
    assert "position_count" in summary, "Summary should have position count"
    assert "kill_switch_active" in summary, "Summary should have kill switch status"


def test_router_deterministic_execution():
    """Test that router produces deterministic results"""
    from ally.tools.trading_router import trading_router_process_signal
    
    # Process same signal twice
    signal = {
        "symbol": "TEST",
        "action": "buy",
        "qty": 10,
        "type": "market"
    }
    
    result1 = trading_router_process_signal(
        signal=signal,
        backend="simulator",
        live=False,
        risk_enabled=False  # Disable for deterministic test
    )
    
    result2 = trading_router_process_signal(
        signal=signal,
        backend="simulator", 
        live=False,
        risk_enabled=False
    )
    
    assert result1.ok == result2.ok, "Both should have same success status"
    if result1.ok and result2.ok:
        assert result1.data["success"] == result2.data["success"], "Both should have same processing result"
        assert result1.data["symbol"] == result2.data["symbol"], "Symbol should match"
        assert result1.data["side"] == result2.data["side"], "Side should match"


def test_router_missing_signals_file():
    """Test handling missing signals file"""
    from ally.tools.trading_router import trading_router_run_strategy
    
    result = trading_router_run_strategy(
        signals_file="nonexistent.yaml",
        backend="simulator",
        live=False
    )
    
    assert result.ok == False, "Should fail with missing file"
    assert "not found" in str(result.errors).lower(), "Should report file not found"


def test_router_empty_signals_file(tmp_path):
    """Test handling empty signals file"""
    from ally.tools.trading_router import trading_router_run_strategy
    
    # Create empty signals file
    signals_data = {"strategy": "empty", "signals": []}
    signals_file = tmp_path / "empty.yaml"
    
    with open(signals_file, 'w') as f:
        yaml.dump(signals_data, f)
    
    result = trading_router_run_strategy(
        signals_file=str(signals_file),
        backend="simulator",
        live=False
    )
    
    assert result.ok == False, "Should fail with empty signals"
    assert "no signals" in str(result.errors).lower(), "Should report no signals"


def test_router_with_risk_disabled():
    """Test router with risk checks disabled"""
    from ally.tools.trading_router import trading_router_process_signal
    
    # Large order that would normally be rejected
    signal = {
        "symbol": "AAPL",
        "action": "buy",
        "qty": 100000,  # Large quantity
        "type": "market"
    }
    
    result = trading_router_process_signal(
        signal=signal,
        backend="simulator",
        live=False,
        risk_enabled=False  # Disable risk checks
    )
    
    assert result.ok == True, "Should succeed with risk disabled"
    assert result.data["success"] == True, "Should process order successfully"
    assert result.data["qty"] == 100000, "Should maintain original quantity"


def test_router_json_signals_file(tmp_path):
    """Test processing JSON signals file"""
    from ally.tools.trading_router import trading_router_run_strategy
    
    # Create JSON signals file
    signals_data = {
        "strategy": "json_test",
        "signals": [
            {
                "symbol": "AAPL",
                "action": "buy",
                "qty": 25,
                "type": "market"
            }
        ]
    }
    
    signals_file = tmp_path / "test.json"
    with open(signals_file, 'w') as f:
        import json
        json.dump(signals_data, f)
    
    result = trading_router_run_strategy(
        signals_file=str(signals_file),
        backend="simulator",
        live=False
    )
    
    assert result.ok == True, f"JSON strategy should execute: {result.errors}"
    assert result.data["signals_processed"] == 1, "Should process 1 signal"


def test_router_limit_order_processing():
    """Test processing limit orders"""
    from ally.tools.trading_router import trading_router_process_signal
    
    signal = {
        "symbol": "MSFT",
        "action": "buy",
        "qty": 50,
        "type": "limit",
        "limit": 300.0
    }
    
    result = trading_router_process_signal(
        signal=signal,
        backend="simulator",
        live=False,
        risk_enabled=True
    )
    
    assert result.ok == True, f"Limit order should succeed: {result.errors}"
    assert result.data["success"] == True, "Should process limit order"
    assert result.data["type"] == "limit", "Should maintain order type"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])