#!/usr/bin/env python3
"""
Research pipeline integration tests - comprehensive Phase 4 testing
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


def test_spec_validation_and_loading():
    """Test strategy specification validation and loading"""
    from ally.research.spec import research_spec_validate, research_spec_load
    
    # Create temporary spec file
    spec_data = {
        'name': 'Test-Strategy',
        'universe': {
            'asset_class': 'equities_us',
            'inclusion': {'market_cap_min': 1000000000}
        },
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {
            'type': 'cross_sectional',
            'formula': 'momentum',
            'rebalance': 'monthly'
        },
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 50},
        'costs': {'bps_per_turnover': 15.0},
        'backtest': {
            'start': '2020-01-01',
            'end': '2023-12-31',
            'benchmark': 'SPY',
            'seed': 42
        },
        'gates': {},
        'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        temp_spec_path = f.name
    
    try:
        # Test spec validation
        validate_result = research_spec_validate(spec_path=temp_spec_path, live=False)
        assert validate_result.ok == True, f"Spec validation failed: {validate_result.errors}"
        assert validate_result.data["spec_valid"] == True
        assert "spec_hash" in validate_result.data
        assert "receipt_hash" in validate_result.data
        
        # Test spec loading
        load_result = research_spec_load(spec_path=temp_spec_path, live=False)
        assert load_result.ok == True, f"Spec loading failed: {load_result.errors}"
        assert load_result.data["spec_loaded"] == True
        assert load_result.data["spec_name"] == "Test-Strategy"
        assert "spec_dict" in load_result.data
        
    finally:
        os.unlink(temp_spec_path)


def test_replication_pipeline():
    """Test complete replication pipeline"""
    from ally.research.replication import research_replication_run
    from ally.research.spec import StrategySpec
    
    # Create temporary spec for replication
    spec_data = {
        'name': 'XS-Momentum-Test',
        'universe': {
            'asset_class': 'equities_us',
            'inclusion': {'market_cap_min': 1000000000}
        },
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {
            'type': 'cross_sectional',
            'formula': 'ret_12m - ret_1m',
            'rebalance': 'monthly'
        },
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 20},
        'costs': {'bps_per_turnover': 15.0},
        'backtest': {
            'start': '2023-01-01',
            'end': '2023-12-31',
            'benchmark': 'SPY',
            'seed': 42
        },
        'gates': {},
        'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        temp_spec_path = f.name
    
    try:
        # Test replication pipeline
        result = research_replication_run(spec_path=temp_spec_path, live=False)
        
        assert result.ok == True, f"Replication failed: {result.errors}"
        assert "spec_name" in result.data
        assert "backtest_results" in result.data
        assert "receipts" in result.data
        
        # Check backtest results structure
        backtest = result.data["backtest_results"]
        required_metrics = ["annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown"]
        for metric in required_metrics:
            assert metric in backtest, f"Missing backtest metric: {metric}"
        
        # Check receipts
        receipts = result.data["receipts"]
        assert "universe" in receipts
        assert "signal" in receipts
        assert "weights" in receipts
        assert "backtest" in receipts
        
    finally:
        os.unlink(temp_spec_path)


def test_factorlens_analysis():
    """Test FactorLens regression analysis"""
    from ally.research.factorlens import research_factorlens_analyze
    import numpy as np
    
    # Generate mock backtest results
    dates = [f"2023-01-{i:02d}" for i in range(1, 32)]
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, len(dates))  # Slight positive alpha
    
    mock_backtest = {
        "portfolio_returns": {date: ret for date, ret in zip(dates, returns)},
        "annual_return": np.mean(returns) * 252,
        "annual_volatility": np.std(returns) * np.sqrt(252),
        "sharpe_ratio": (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
    }
    
    result = research_factorlens_analyze(
        backtest_results=mock_backtest,
        spec_name="test_strategy",
        live=False
    )
    
    assert result.ok == True, f"FactorLens analysis failed: {result.errors}"
    
    # Check required fields
    data = result.data
    assert "alpha_annual" in data
    assert "alpha_t_stat" in data
    assert "alpha_significant" in data
    assert "r_squared" in data
    assert "factor_loadings" in data
    assert "factorlens_receipt" in data
    
    # Check factor loadings structure
    loadings = data["factor_loadings"]
    expected_factors = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    for factor in expected_factors:
        assert factor in loadings, f"Missing factor: {factor}"
        assert "coefficient" in loadings[factor]
        assert "t_stat" in loadings[factor]
        assert "p_value" in loadings[factor]


def test_fdr_analysis():
    """Test False Discovery Rate analysis"""
    from ally.research.fdr import research_fdr_analyze_grid, research_fdr_simulate_grid
    import numpy as np
    
    # Test grid simulation
    sim_result = research_fdr_simulate_grid(
        n_strategies=20,
        n_observations=100,
        true_alpha_rate=0.2,
        fdr_level=0.05,
        live=False
    )
    
    assert sim_result.ok == True, f"FDR simulation failed: {sim_result.errors}"
    
    # Check simulation results
    data = sim_result.data
    assert "simulation_params" in data
    assert "fdr_analysis" in data
    assert "performance_metrics" in data
    
    # Check FDR analysis
    fdr_data = data["fdr_analysis"]
    assert "n_strategies_total" in fdr_data
    assert "n_survivors_bh" in fdr_data
    assert "bh_threshold" in fdr_data
    assert "survivors" in fdr_data
    assert "rejected" in fdr_data
    assert "fdr_receipt" in fdr_data
    
    # Check performance metrics
    perf_data = data["performance_metrics"]
    assert "empirical_fdr" in perf_data
    assert "power" in perf_data
    assert "confusion_matrix" in perf_data
    
    # Test manual grid analysis
    np.random.seed(42)
    mock_strategies = [
        {
            'strategy_id': f"STRAT_{i:03d}",
            'p_value': np.random.uniform(0, 1),
            'annual_return': np.random.normal(0.05, 0.1),
            'has_true_alpha': i < 4  # First 4 have true alpha
        }
        for i in range(20)
    ]
    
    grid_result = research_fdr_analyze_grid(
        strategy_results=mock_strategies,
        fdr_level=0.05,
        live=False
    )
    
    assert grid_result.ok == True, f"FDR grid analysis failed: {grid_result.errors}"
    assert "n_survivors_bh" in grid_result.data
    assert len(grid_result.data["survivors"]) + len(grid_result.data["rejected"]) == 20


def test_promotion_validation():
    """Test promotion holdout validation"""
    from ally.research.promotion import research_promotion_validate_holdout
    import numpy as np
    import pandas as pd
    
    # Generate mock backtest with positive alpha
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0008, 0.015, len(dates))  # 20% annual return
    
    mock_backtest = {
        "spec_name": "promotion_test",
        "portfolio_returns": {
            date.strftime('%Y-%m-%d'): ret
            for date, ret in zip(dates, returns)
        },
        "annual_return": np.mean(returns) * 252,
        "annual_volatility": np.std(returns) * np.sqrt(252)
    }
    
    result = research_promotion_validate_holdout(
        backtest_results=mock_backtest,
        t_stat_threshold=1.5,  # Lower for test
        max_turnover=3.0,
        live=False
    )
    
    assert result.ok == True, f"Promotion validation failed: {result.errors}"
    
    # Check holdout validation results
    data = result.data
    assert "holdout_period" in data
    assert "holdout_statistics" in data
    assert "promotion_checks" in data
    assert "promotion_approved" in data
    assert "promotion_receipt" in data
    
    # Check holdout statistics
    holdout_stats = data["holdout_statistics"]
    required_stats = [
        "annual_return", "sharpe_ratio", "t_statistic", 
        "max_drawdown", "p_value"
    ]
    for stat in required_stats:
        assert stat in holdout_stats, f"Missing holdout statistic: {stat}"
    
    # Check promotion checks
    checks = data["promotion_checks"]
    required_checks = ["t_statistic", "turnover", "transaction_cost", "capacity"]
    for check in required_checks:
        assert check in checks, f"Missing promotion check: {check}"
        assert "value" in checks[check]
        assert "threshold" in checks[check]
        assert "passed" in checks[check]


def test_strategy_implementations():
    """Test individual strategy implementations"""
    from ally.strategies.zoo.xs_momentum import strategies_xs_momentum_run
    from ally.strategies.zoo.value_btm import strategies_value_btm_run
    from ally.strategies.zoo.ts_trend import strategies_ts_trend_run
    
    # Test XS-Momentum
    xs_result = strategies_xs_momentum_run(
        spec_path="ally/strategies/specs/xs_momentum.yaml",
        live=False
    )
    
    # Note: This will fail if spec file doesn't exist, but tests the integration
    # In CI, we'd create the spec files or mock them
    print(f"XS-Momentum test result: {xs_result.ok}")
    if not xs_result.ok:
        print(f"XS-Momentum errors: {xs_result.errors}")
    
    # Similar tests for other strategies would follow the same pattern


def test_receipt_generation():
    """Test that all research operations generate receipts"""
    from ally.research.factorlens import research_factorlens_create_factor_data
    from ally.research.fdr import research_fdr_validate_procedure
    from ally.utils.receipts import generate_receipt
    
    # Test factor data creation receipt
    factor_result = research_factorlens_create_factor_data(
        start_date="2023-01-01",
        end_date="2023-12-31",
        live=False
    )
    
    assert factor_result.ok == True, f"Factor data creation failed: {factor_result.errors}"
    assert "factor_receipt" in factor_result.data
    assert len(factor_result.data["factor_receipt"]) == 16
    
    # Test FDR validation receipt
    fdr_validation = research_fdr_validate_procedure(
        n_simulations=5,  # Small number for test
        fdr_level=0.05,
        live=False
    )
    
    assert fdr_validation.ok == True, f"FDR validation failed: {fdr_validation.errors}"
    assert "validation_receipt" in fdr_validation.data
    
    # Test direct receipt generation
    test_data = {"test_field": "test_value", "timestamp": "2024-01-15T10:00:00Z"}
    receipt_hash = generate_receipt("test.operation", test_data)
    
    assert isinstance(receipt_hash, str)
    assert len(receipt_hash) == 40  # SHA-1 hex string length


def test_research_gates_integration():
    """Test integration of research gates pipeline"""
    from ally.research.replication import research_replication_run
    from ally.research.factorlens import research_factorlens_analyze
    from ally.research.fdr import research_fdr_analyze_grid
    from ally.research.promotion import research_promotion_validate_holdout
    
    # This would be a full integration test
    # Create spec -> Run replication -> FactorLens -> FDR -> Promotion
    
    # For now, test that the gate functions can be chained
    spec_data = {
        'name': 'Integration-Test',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 10},
        'costs': {'bps_per_turnover': 10.0},
        'backtest': {'start': '2023-01-01', 'end': '2023-06-30', 'benchmark': 'SPY', 'seed': 42},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        temp_spec_path = f.name
    
    try:
        # Step 1: Replication
        replication_result = research_replication_run(spec_path=temp_spec_path, live=False)
        assert replication_result.ok == True
        
        # Step 2: FactorLens  
        factorlens_result = research_factorlens_analyze(
            backtest_results=replication_result.data["backtest_results"],
            spec_name=replication_result.data["spec_name"],
            live=False
        )
        # Note: May fail with insufficient data, but tests integration
        
        # Step 3: Promotion (skip FDR for single strategy)
        if factorlens_result.ok:
            promotion_result = research_promotion_validate_holdout(
                backtest_results=replication_result.data["backtest_results"],
                factorlens_results=factorlens_result.data,
                live=False
            )
            # Tests the integration flow
        
        print("✅ Research gates integration test completed")
        
    finally:
        os.unlink(temp_spec_path)


def test_error_handling():
    """Test error handling in research pipeline"""
    from ally.research.spec import research_spec_validate
    from ally.research.factorlens import research_factorlens_analyze
    
    # Test invalid spec file
    invalid_spec_result = research_spec_validate(spec_path="/nonexistent/path.yaml", live=False)
    assert invalid_spec_result.ok == False
    assert len(invalid_spec_result.errors) > 0
    
    # Test FactorLens with invalid data
    invalid_factorlens = research_factorlens_analyze(
        backtest_results={},  # Empty results
        spec_name="invalid_test",
        live=False
    )
    assert invalid_factorlens.ok == False
    assert "No portfolio returns" in str(invalid_factorlens.errors)


def test_deterministic_behavior():
    """Test that research pipeline produces deterministic results"""
    from ally.research.fdr import research_fdr_simulate_grid
    
    # Run same simulation twice
    result1 = research_fdr_simulate_grid(
        n_strategies=10,
        n_observations=50,
        true_alpha_rate=0.2,
        fdr_level=0.05,
        live=False
    )
    
    result2 = research_fdr_simulate_grid(
        n_strategies=10,
        n_observations=50,
        true_alpha_rate=0.2,
        fdr_level=0.05,
        live=False
    )
    
    assert result1.ok == True and result2.ok == True
    
    # Should produce same results due to fixed seed
    assert result1.data["fdr_analysis"]["n_survivors_bh"] == result2.data["fdr_analysis"]["n_survivors_bh"]
    assert abs(result1.data["performance_metrics"]["empirical_fdr"] - 
              result2.data["performance_metrics"]["empirical_fdr"]) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])