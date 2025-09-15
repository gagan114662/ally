#!/usr/bin/env python3
"""
Transaction costs and liquidity tests - Phase 5.2 testing
"""

import os
import pytest
import numpy as np
import pandas as pd

# Ensure CI stays dry
os.environ["ALLY_LIVE"] = "0"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_cost_model_parsing():
    """Test cost model string parsing"""
    from ally.research.costs import parse_cost_model_string
    
    # Test basic parsing
    model = parse_cost_model_string("bps=15,impact=k*sqrt(q),k=8.0,borrow_bps=75")
    
    assert model.bps_per_turnover == 15.0
    assert model.impact_model == "sqrt"
    assert model.impact_k == 8.0
    assert model.borrow_bps_annual == 75.0
    
    # Test linear impact model
    model_linear = parse_cost_model_string("bps=10,impact=linear,k=5.0")
    assert model_linear.impact_model == "linear"
    assert model_linear.impact_k == 5.0
    
    # Test no impact model
    model_none = parse_cost_model_string("bps=12,impact=none")
    assert model_none.impact_model == "none"
    
    # Test capacity specification
    model_cap = parse_cost_model_string("bps=8,capacity=50000000")
    assert model_cap.capacity_usd == 50_000_000


def test_market_impact_calculation():
    """Test market impact cost calculations"""
    from ally.research.costs import compute_market_impact_cost
    
    price = 100.0
    
    # Test sqrt model
    impact_small = compute_market_impact_cost(1000, price, "sqrt", k=6.0)
    impact_large = compute_market_impact_cost(10000, price, "sqrt", k=6.0)
    
    assert impact_large > impact_small, "Larger trades should have higher impact"
    assert impact_small >= 0, "Impact cost should be non-negative"
    
    # Test linear model
    impact_linear_small = compute_market_impact_cost(1000, price, "linear", k=6.0)
    impact_linear_large = compute_market_impact_cost(2000, price, "linear", k=6.0)
    
    # Linear model should scale proportionally
    assert abs(impact_linear_large - 2 * impact_linear_small) < 1e-6
    
    # Test no impact model
    impact_none = compute_market_impact_cost(1000, price, "none", k=6.0)
    assert impact_none == 0.0


def test_borrow_cost_calculation():
    """Test short borrow cost calculations"""
    from ally.research.costs import compute_borrow_cost
    
    price = 100.0
    borrow_rate = 50.0  # 50 bps annually
    
    # Test long position (no borrow cost)
    long_cost = compute_borrow_cost(1000, price, 30, borrow_rate)
    assert long_cost == 0.0
    
    # Test short position
    short_cost = compute_borrow_cost(-1000, price, 30, borrow_rate)
    assert short_cost > 0.0
    
    # Test scaling with position size
    short_cost_double = compute_borrow_cost(-2000, price, 30, borrow_rate)
    assert abs(short_cost_double - 2 * short_cost) < 1e-6
    
    # Test scaling with time
    short_cost_double_time = compute_borrow_cost(-1000, price, 60, borrow_rate)
    assert abs(short_cost_double_time - 2 * short_cost) < 1e-6


def test_transaction_cost_application():
    """Test applying transaction costs to portfolio"""
    from ally.research.costs import apply_transaction_costs, CostModel
    
    # Create mock data
    dates = pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01'])
    symbols = ['STOCK001', 'STOCK002']
    
    # Mock weights
    weights_data = []
    for i, date in enumerate(dates):
        for j, symbol in enumerate(symbols):
            weight = 0.4 + i * 0.1 + j * 0.1  # Changing weights
            weights_data.append({
                'date': date,
                'symbol': symbol,
                'weight': weight
            })
    
    weights_df = pd.DataFrame(weights_data)
    
    # Mock OHLCV
    ohlcv_data = []
    for date in dates:
        for symbol in symbols:
            price = 100.0 + np.random.random() * 10  # Random prices
            ohlcv_data.append({
                'date': date,
                'symbol': symbol,
                'close': price
            })
    
    ohlcv_df = pd.DataFrame(ohlcv_data)
    
    # Apply costs
    cost_model = CostModel(bps_per_turnover=10.0, impact_model="sqrt", impact_k=5.0)
    
    cost_analysis = apply_transaction_costs(
        weights_df=weights_df,
        ohlcv_df=ohlcv_df,
        cost_model=cost_model,
        initial_capital=1_000_000
    )
    
    # Validate results structure
    assert "total_costs" in cost_analysis
    assert "turnover_analysis" in cost_analysis
    assert "daily_costs" in cost_analysis
    assert "cost_breakdown" in cost_analysis
    
    # Check cost components
    total_costs = cost_analysis["total_costs"]
    assert "linear_costs" in total_costs
    assert "impact_costs" in total_costs
    assert "borrow_costs" in total_costs
    assert "total_cost_usd" in total_costs
    
    # Costs should be non-negative
    assert total_costs["linear_costs"] >= 0
    assert total_costs["impact_costs"] >= 0
    assert total_costs["borrow_costs"] >= 0
    assert total_costs["total_cost_usd"] >= 0


def test_capacity_constraints_validation():
    """Test capacity and turnover constraint validation"""
    from ally.research.costs import check_capacity_constraints
    
    # Mock backtest results
    backtest_results = {
        "annual_return": 0.12,
        "total_value": 75_000_000  # $75M portfolio
    }
    
    # Mock cost analysis
    cost_analysis = {
        "turnover_analysis": {
            "annual_turnover_ratio": 2.5  # 2.5x annual turnover
        }
    }
    
    # Test constraint validation
    constraints = check_capacity_constraints(
        backtest_results=backtest_results,
        cost_analysis=cost_analysis,
        capacity_usd=100_000_000,  # $100M capacity
        max_turnover_annual=3.0    # 3x max turnover
    )
    
    # Should pass constraints
    assert constraints["capacity_check"]["capacity_violated"] == False
    assert constraints["turnover_check"]["turnover_violated"] == False
    assert constraints["overall_status"]["constraints_passed"] == True
    
    # Test constraint violations
    constraints_violated = check_capacity_constraints(
        backtest_results={"total_value": 120_000_000},  # Exceeds capacity
        cost_analysis={"turnover_analysis": {"annual_turnover_ratio": 4.0}},  # Exceeds turnover
        capacity_usd=100_000_000,
        max_turnover_annual=3.0
    )
    
    # Should violate both constraints
    assert constraints_violated["capacity_check"]["capacity_violated"] == True
    assert constraints_violated["turnover_check"]["turnover_violated"] == True
    assert constraints_violated["overall_status"]["constraints_passed"] == False
    assert len(constraints_violated["overall_status"]["violations"]) == 2


def test_cost_model_validation_api():
    """Test cost model validation via API"""
    from ally.research.costs import research_costs_model
    
    result = research_costs_model(
        model_string="bps=12,impact=k*sqrt(q),k=7.0,borrow_bps=60",
        live=False
    )
    
    assert result.ok == True
    assert "model_receipt" in result.data
    assert "parsed_model" in result.data
    assert "model_validation" in result.data
    assert "summary_statistics" in result.data
    
    # Check parsed model
    parsed = result.data["parsed_model"]
    assert parsed["bps_per_turnover"] == 12.0
    assert parsed["impact_model"] == "sqrt"
    assert parsed["impact_k"] == 7.0
    assert parsed["borrow_bps_annual"] == 60.0
    
    # Check validation results
    validation = result.data["model_validation"]
    assert len(validation) > 0
    
    for test_result in validation:
        assert "trade" in test_result
        assert "costs" in test_result
        costs = test_result["costs"]
        assert "linear_cost" in costs
        assert "impact_cost" in costs
        assert "total_cost" in costs
        assert "cost_bps" in costs
        assert costs["total_cost"] >= 0


def test_cost_application_api():
    """Test cost application via API"""
    from ally.research.costs import research_costs_apply
    
    # Mock backtest results
    mock_backtest = {
        "annual_return": 0.15,
        "sharpe_ratio": 1.25,
        "annual_volatility": 0.16,
        "max_drawdown": -0.08
    }
    
    result = research_costs_apply(
        backtest_results=mock_backtest,
        model_string="bps=10,impact=k*sqrt(q),k=6.0,borrow_bps=50",
        capacity_usd=100_000_000,
        max_turnover=2.0,
        live=False
    )
    
    # Should pass (reasonable constraints)
    assert result.ok == True
    assert "costs_receipt" in result.data
    assert "original_metrics" in result.data
    assert "cost_adjusted_metrics" in result.data
    assert "constraints_validation" in result.data
    
    # Check cost adjustment
    orig_return = result.data["original_metrics"]["annual_return"]
    adj_return = result.data["cost_adjusted_metrics"]["annual_return"]
    cost_drag = result.data["cost_adjusted_metrics"]["cost_drag_annual"]
    
    assert adj_return < orig_return, "Cost-adjusted return should be lower"
    assert cost_drag > 0, "Cost drag should be positive"
    assert abs((orig_return - adj_return) - cost_drag) < 1e-6


def test_cost_application_constraint_violations():
    """Test cost application with constraint violations"""
    from ally.research.costs import research_costs_apply
    
    # Mock backtest with high turnover scenario
    mock_backtest = {
        "annual_return": 0.20,
        "sharpe_ratio": 1.5,
        "annual_volatility": 0.18,
        "total_value": 150_000_000  # Exceeds capacity
    }
    
    result = research_costs_apply(
        backtest_results=mock_backtest,
        model_string="bps=25,impact=k*sqrt(q),k=10.0",  # High cost model
        capacity_usd=100_000_000,  # Lower capacity
        max_turnover=1.0,          # Low turnover limit
        live=False
    )
    
    # Should fail due to capacity constraint
    assert result.ok == False
    assert len(result.errors) > 0
    assert "Capacity/turnover constraints violated" in result.errors[0]
    
    # But should still return analysis
    assert "constraints_validation" in result.data
    violations = result.data["constraints_validation"]["overall_status"]["violations"]
    assert len(violations) > 0


def test_deterministic_cost_calculation():
    """Test that cost calculations are deterministic"""
    from ally.research.costs import research_costs_apply
    
    mock_backtest = {
        "annual_return": 0.12,
        "sharpe_ratio": 1.0,
        "annual_volatility": 0.15
    }
    
    # Run twice with same parameters
    result1 = research_costs_apply(
        backtest_results=mock_backtest,
        model_string="bps=15,impact=k*sqrt(q),k=8.0",
        capacity_usd=200_000_000,
        max_turnover=3.0,
        live=False
    )
    
    result2 = research_costs_apply(
        backtest_results=mock_backtest,
        model_string="bps=15,impact=k*sqrt(q),k=8.0",
        capacity_usd=200_000_000,
        max_turnover=3.0,
        live=False
    )
    
    # Results should be identical
    assert result1.ok == result2.ok
    if result1.ok and result2.ok:
        drag1 = result1.data["cost_adjusted_metrics"]["cost_drag_annual"]
        drag2 = result2.data["cost_adjusted_metrics"]["cost_drag_annual"]
        assert abs(drag1 - drag2) < 1e-10, "Cost calculations should be deterministic"


def test_edge_case_empty_weights():
    """Test handling of empty portfolio weights"""
    from ally.research.costs import apply_transaction_costs, CostModel
    
    # Empty weights DataFrame
    empty_weights = pd.DataFrame(columns=['date', 'symbol', 'weight'])
    mock_ohlcv = pd.DataFrame([
        {'date': pd.to_datetime('2023-01-01'), 'symbol': 'TEST', 'close': 100.0}
    ])
    
    cost_model = CostModel()
    
    # Should raise clear error
    with pytest.raises(ValueError, match="No portfolio weights provided"):
        apply_transaction_costs(empty_weights, mock_ohlcv, cost_model)


def test_cost_model_error_handling():
    """Test cost model error handling"""
    from ally.research.costs import research_costs_model, research_costs_apply
    
    # Test invalid model string (should not crash)
    result = research_costs_model(
        model_string="invalid=string=format",
        live=False
    )
    
    # Should still parse (ignoring invalid parts)
    assert result.ok == True
    
    # Test cost application with invalid backtest results
    result = research_costs_apply(
        backtest_results={},  # Empty results
        model_string="bps=10",
        live=False
    )
    
    # Should handle gracefully
    assert result.ok == True  # Still processes with defaults


def test_cost_receipts_generation():
    """Test that cost operations generate proper receipts"""
    from ally.research.costs import research_costs_model, research_costs_apply
    
    # Test model validation receipt
    model_result = research_costs_model(
        model_string="bps=20,impact=k*sqrt(q),k=9.0",
        live=False
    )
    
    assert model_result.ok == True
    assert "model_receipt" in model_result.data
    assert len(model_result.data["model_receipt"]) == 16
    assert hasattr(model_result, 'receipt_hash')
    assert len(model_result.receipt_hash) == 40
    
    # Test cost application receipt
    mock_backtest = {"annual_return": 0.10, "sharpe_ratio": 0.8}
    
    costs_result = research_costs_apply(
        backtest_results=mock_backtest,
        model_string="bps=18,impact=linear,k=4.0",
        live=False
    )
    
    assert costs_result.ok == True
    assert "costs_receipt" in costs_result.data
    assert len(costs_result.data["costs_receipt"]) == 16
    assert hasattr(costs_result, 'receipt_hash')


def test_cost_model_impact_comparison():
    """Test different impact models produce expected relative costs"""
    from ally.research.costs import compute_market_impact_cost
    
    quantity = 5000
    price = 100.0
    k = 6.0
    
    # Calculate costs for different models
    cost_none = compute_market_impact_cost(quantity, price, "none", k)
    cost_linear = compute_market_impact_cost(quantity, price, "linear", k)
    cost_sqrt = compute_market_impact_cost(quantity, price, "sqrt", k)
    
    # Expected ordering
    assert cost_none == 0.0
    assert cost_linear > cost_none
    assert cost_sqrt > cost_none
    
    # For moderate quantities, sqrt should be between none and linear
    # (this depends on the specific scaling, but generally true for our implementation)
    
    # Test scaling properties
    cost_sqrt_double = compute_market_impact_cost(quantity * 4, price, "sqrt", k)
    # sqrt(4x) = 2*sqrt(x), so cost should roughly double
    assert cost_sqrt_double > cost_sqrt * 1.8  # Allow some tolerance
    assert cost_sqrt_double < cost_sqrt * 2.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])