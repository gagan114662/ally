# tests/test_pit.py
"""Tests for Point-in-Time (PIT) tools - critical bias detection tests."""
import pytest
from datetime import datetime
from ally.tools import TOOL_REGISTRY
from ally.schemas.pit import PITUniverseRow, ActionRow, PITSnapshot
from ally.utils.pit_io import get_pit_fixture_summary

pytestmark = pytest.mark.pit


def test_pit_tools_registered():
    """Verify all PIT tools are registered."""
    required_tools = [
        "pit.load_universe",
        "pit.load_corporate_actions", 
        "pit.apply_delistings",
        "pit.get_snapshot",
        "pit.validate_backtest"
    ]
    
    for tool in required_tools:
        assert tool in TOOL_REGISTRY, f"Tool {tool} not registered"


def test_leakage_trap_future_snapshot():
    """CRITICAL: Verify future date snapshots are flagged as high risk."""
    future_date = "2030-01-01"  # Far future date
    result = TOOL_REGISTRY["pit.get_snapshot"](date=future_date)
    
    assert result.ok
    assert result.data["leakage_risk"]["is_future_date"] is True
    assert result.data["leakage_risk"]["risk_level"] == "high"
    assert "future" in result.data["leakage_risk"]["warning"].lower()


def test_survivorship_bias_detection():
    """CRITICAL: Detect when delisted symbols are missing from backtest."""
    # Simulate backtest with only surviving symbols - no DELISTED_CORP
    backtest_data = {
        "trades": [
            {"symbol": "AAPL", "date": "2023-02-01", "return": 0.05},
            {"symbol": "MSFT", "date": "2023-02-01", "return": 0.03},
            # Missing DELISTED_CORP which gets delisted in Aug 2023
        ]
    }
    
    result = TOOL_REGISTRY["pit.validate_backtest"](
        backtest_data=backtest_data,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    assert result.ok
    # Since DELISTED_CORP isn't in our backtest symbols, there's no survivorship bias detected
    # The bias detection only works if the symbol was used but delisting returns are missing
    
    # Better test: include DELISTED_CORP but without delisting returns
    backtest_data_with_bias = {
        "trades": [
            {"symbol": "AAPL", "date": "2023-02-01", "return": 0.05},
            {"symbol": "DELISTED_CORP", "date": "2023-07-01", "return": 0.02},
            # Missing the delisting return for DELISTED_CORP
        ]
    }
    
    result2 = TOOL_REGISTRY["pit.validate_backtest"](
        backtest_data=backtest_data_with_bias,
        start_date="2023-01-01", 
        end_date="2023-12-31"
    )
    
    assert result2.ok
    # Should warn about missing delisting coverage
    assert result2.data["delisting_coverage"] is False


def test_delisting_returns_inclusion():
    """CRITICAL: Ensure delisting returns are properly included."""
    returns_data = [
        {"date": "2023-01-15", "symbol": "DELISTED_CORP", "return": -0.05},
        {"date": "2023-07-15", "symbol": "DELISTED_CORP", "return": -0.12},
        # Missing the final delisting return for Aug 15
    ]
    
    result = TOOL_REGISTRY["pit.apply_delistings"](
        returns_data=returns_data,
        start_date="2023-01-01", 
        end_date="2023-12-31",
        include_delisting_returns=True
    )
    
    assert result.ok
    assert result.data["delisting_returns_added"] > 0
    assert result.data["adjusted_records"] > len(returns_data)
    
    # Check that delisting return was added
    delisting_returns = result.data["delisting_details"]
    assert len(delisting_returns) == 1
    assert delisting_returns[0]["symbol"] == "DELISTED_CORP"
    assert delisting_returns[0]["type"] == "delisting_return" 
    assert delisting_returns[0]["return"] < 0  # Negative return for delisting


def test_pit_universe_survivorship_warning():
    """Detect potential survivorship bias when all symbols active at end."""
    result = TOOL_REGISTRY["pit.load_universe"](
        start_date="2023-01-01",
        end_date="2023-12-01",  # Date where DELISTED_CORP is inactive
        symbols=["AAPL", "MSFT", "GOOGL", "TSLA"]  # Only survivors
    )
    
    assert result.ok
    # Should warn about potential survivorship bias since we excluded delisted symbol
    assert result.data["survivorship_warning"] is not None


def test_corporate_actions_delisting_detection():
    """Verify delisting events are properly loaded and flagged."""
    result = TOOL_REGISTRY["pit.load_corporate_actions"](
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    assert result.ok
    assert result.data["delisting_count"] > 0
    assert "DELISTED_CORP" in result.data["delisted_symbols"]
    assert "delisting" in result.data["action_counts"]


def test_pit_snapshot_consistency():
    """Verify snapshot data consistency across dates."""
    # Early snapshot should include DELISTED_CORP
    early_result = TOOL_REGISTRY["pit.get_snapshot"](date="2023-06-01")
    assert early_result.ok
    assert "DELISTED_CORP" in early_result.data["symbols"]
    
    # Late snapshot should exclude DELISTED_CORP  
    late_result = TOOL_REGISTRY["pit.get_snapshot"](date="2023-12-01")
    assert late_result.ok
    assert "DELISTED_CORP" not in late_result.data["symbols"]
    
    # Universe size should decrease due to delisting
    assert late_result.data["universe_size"] < early_result.data["universe_size"]


def test_backtest_validation_comprehensive():
    """Comprehensive backtest validation with multiple bias checks."""
    # Good backtest with delisting returns
    good_backtest = {
        "trades": [
            {"symbol": "AAPL", "date": "2023-02-01", "return": 0.05},
            {"symbol": "DELISTED_CORP", "date": "2023-07-01", "return": -0.10},
            {"symbol": "DELISTED_CORP", "date": "2023-08-15", "return": -0.80, "type": "delisting_return"}
        ],
        "positions": [
            {"symbol": "AAPL", "date": "2023-02-01"},
            {"symbol": "DELISTED_CORP", "date": "2023-07-01"}
        ]
    }
    
    result = TOOL_REGISTRY["pit.validate_backtest"](
        backtest_data=good_backtest,
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    assert result.ok
    assert result.data["valid"] is True
    assert result.data["survivorship_bias"] is False
    assert result.data["delisting_coverage"] is True
    

def test_look_ahead_bias_detection():
    """Detect trades outside backtest period (look-ahead bias)."""
    bad_backtest = {
        "trades": [
            {"symbol": "AAPL", "date": "2022-12-31", "return": 0.05},  # Before start
            {"symbol": "MSFT", "date": "2024-01-01", "return": 0.03},   # After end
        ]
    }
    
    result = TOOL_REGISTRY["pit.validate_backtest"](
        backtest_data=bad_backtest,
        start_date="2023-01-01", 
        end_date="2023-12-31"
    )
    
    assert result.ok
    assert result.data["look_ahead_bias"] is True
    assert result.data["valid"] is False


def test_fixture_data_integrity():
    """Verify fixture data has expected structure for testing."""
    summary = get_pit_fixture_summary()
    
    # Must have delisting events for bias testing
    assert summary["delisting_count"] > 0
    assert summary["symbol_count"] > 3
    assert summary["active_periods"] > 0
    assert summary["inactive_periods"] > 0
    
    # Date range should span test period
    assert "2023-01-01" in summary["date_range"]
    assert "2023-12-01" in summary["date_range"]


def test_delisting_adjustment_impact():
    """Verify delisting adjustments materially impact results."""
    returns_data = [
        {"date": "2023-07-01", "symbol": "DELISTED_CORP", "return": 0.02},
        {"date": "2023-08-01", "symbol": "DELISTED_CORP", "return": 0.01},
    ]
    
    # Without delisting adjustments
    result_no_delistings = TOOL_REGISTRY["pit.apply_delistings"](
        returns_data=returns_data,
        start_date="2023-01-01",
        end_date="2023-12-31", 
        include_delisting_returns=False
    )
    
    # With delisting adjustments
    result_with_delistings = TOOL_REGISTRY["pit.apply_delistings"](
        returns_data=returns_data,
        start_date="2023-01-01",
        end_date="2023-12-31",
        include_delisting_returns=True  
    )
    
    assert result_no_delistings.ok and result_with_delistings.ok
    
    # Should add material negative returns
    assert result_with_delistings.data["adjusted_records"] > result_no_delistings.data["adjusted_records"]
    assert result_with_delistings.data["delisting_returns_added"] > 0
    
    # Check bias impact
    bias_checks = result_with_delistings.data["bias_checks"]
    assert "missing_delisted_symbols" in bias_checks


def test_invalid_date_handling():
    """Test proper error handling for invalid dates."""
    result = TOOL_REGISTRY["pit.get_snapshot"](date="invalid-date")
    assert not result.ok
    assert "Invalid date format" in result.errors[0]


def test_universe_filtering():
    """Test universe filtering capabilities."""
    result = TOOL_REGISTRY["pit.load_universe"](
        start_date="2023-01-01",
        end_date="2023-06-01", 
        min_market_cap=1000000000000,  # 1T filter
        sectors=["Technology"]
    )
    
    assert result.ok
    # Should filter to only large-cap tech stocks
    assert result.data["unique_symbols"] >= 2  # At least AAPL, MSFT, GOOGL
    
    # Verify filtering worked
    universe_data = result.data["universe_data"]
    for record in universe_data:
        if record["market_cap"] is not None:
            assert record["market_cap"] >= 1000000000000
        if record["sector"] is not None:
            assert record["sector"] == "Technology"


def test_zero_leakage_proof():
    """PROOF GENERATION: Verify zero leakage trips in fixture data."""
    # Test multiple snapshots to ensure no future information
    test_dates = ["2023-01-01", "2023-06-01", "2023-12-01"]
    leakage_trips = 0
    
    for date in test_dates:
        result = TOOL_REGISTRY["pit.get_snapshot"](date=date)
        assert result.ok
        
        if result.data["leakage_risk"]["is_future_date"]:
            leakage_trips += 1
    
    # Should be zero for historical dates
    assert leakage_trips == 0, f"Found {leakage_trips} leakage trips in fixture data"


def test_delisting_inclusion_proof():
    """PROOF GENERATION: Verify delistings are properly included."""
    result = TOOL_REGISTRY["pit.apply_delistings"](
        returns_data=[{"date": "2023-07-01", "symbol": "DELISTED_CORP", "return": 0.01}],
        start_date="2023-01-01",
        end_date="2023-12-31",
        include_delisting_returns=True
    )
    
    assert result.ok
    delisting_included = result.data["delisting_returns_added"] > 0
    assert delisting_included, "Delisting returns not properly included"


def test_pit_snapshots_proof():
    """PROOF GENERATION: Verify PIT snapshots work correctly."""
    result = TOOL_REGISTRY["pit.get_snapshot"](date="2023-06-01")
    
    assert result.ok
    assert result.data["universe_size"] > 0
    assert len(result.data["symbols"]) == result.data["universe_size"]
    assert "data_hash" in result.data["meta"]