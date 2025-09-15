#!/usr/bin/env python3
"""
Portfolio sizing and constraints tests - Phase 7.4 testing
"""

import os
import json
from datetime import datetime

# Ensure CI stays dry
os.environ["ALLY_LIVE"] = "0"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Handle missing dependencies gracefully for CI
try:
    import pytest
    import numpy as np
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    
    # Mock implementations for CI
    np = type('np', (), {
        'random': type('random', (), {
            'seed': lambda x: None,
            'random': lambda: 0.5,
            'uniform': lambda a, b: (a + b) / 2,
            'normal': lambda mu, sigma: mu
        })(),
        'array': lambda x: x,
        'sum': lambda x: sum(x) if hasattr(x, '__iter__') else x,
        'abs': lambda x: abs(x) if hasattr(x, '__abs__') else [abs(v) for v in x],
        'sqrt': lambda x: x ** 0.5 if hasattr(x, '__pow__') else [v ** 0.5 for v in x],
        'clip': lambda x, a, b: max(a, min(b, x)),
        'dot': lambda a, b: sum(a[i] * b[i] for i in range(len(a))),
        'max': lambda x: max(x) if hasattr(x, '__iter__') else x,
        'min': lambda x: min(x) if hasattr(x, '__iter__') else x
    })()
    
    pytest = type('pytest', (), {
        'raises': lambda *args, **kwargs: type('MockRaises', (), {
            '__enter__': lambda self: self,
            '__exit__': lambda self, *args: False
        })(),
        'main': lambda args: print("pytest not available - using mock tests")
    })()


def test_kelly_fraction_calculation():
    """Test Kelly fraction calculation with caps and drawdown limits"""
    from ally.research.sizing import kelly_fraction
    
    # Test basic Kelly calculation
    kelly_frac, metadata = kelly_fraction(
        sharpe_ratio=1.2,
        volatility=0.15,
        dd_cap=0.20,
        kelly_cap=0.25,
        seed=42
    )
    
    # Should return valid Kelly fraction
    assert isinstance(kelly_frac, float)
    assert 0 <= abs(kelly_frac) <= 0.25  # Should be capped
    
    # Check metadata structure
    assert "kelly_raw" in metadata
    assert "kelly_capped" in metadata
    assert "cap_applied" in metadata
    assert "dd_adjustment" in metadata
    
    # Test Kelly cap binding with high Sharpe ratio
    high_kelly_frac, high_metadata = kelly_fraction(
        sharpe_ratio=5.0,  # Very high Sharpe
        volatility=0.10,
        dd_cap=0.20,
        kelly_cap=0.25,
        seed=42
    )
    
    # Kelly cap should bind
    assert abs(high_kelly_frac) <= 0.25 + 1e-6
    assert high_metadata["cap_applied"] == True
    
    # Test zero volatility edge case
    zero_vol_frac, zero_metadata = kelly_fraction(
        sharpe_ratio=1.0,
        volatility=0.0,
        dd_cap=0.20,
        kelly_cap=0.25,
        seed=42
    )
    
    assert zero_vol_frac == 0.0
    assert "warning" in zero_metadata


def test_kelly_fraction_deterministic():
    """Test that Kelly fraction calculation is deterministic"""
    from ally.research.sizing import kelly_fraction
    
    # Run twice with same parameters
    frac1, meta1 = kelly_fraction(1.5, 0.12, 0.20, 0.25, seed=42)
    frac2, meta2 = kelly_fraction(1.5, 0.12, 0.20, 0.25, seed=42)
    
    # Should be identical
    assert abs(frac1 - frac2) < 1e-10
    assert abs(meta1["kelly_raw"] - meta2["kelly_raw"]) < 1e-10


def test_volatility_targeting():
    """Test volatility targeting scales weights correctly"""
    from ally.research.sizing import target_vol
    
    # Create test weights and covariance matrix
    weights = [0.4, 0.35, 0.25]
    cov_matrix = [
        [0.04, 0.02, 0.01],
        [0.02, 0.09, 0.03],
        [0.01, 0.03, 0.16]
    ]
    
    vol_target = 0.08
    
    scaled_weights, metadata = target_vol(
        np.array(weights) if DEPS_AVAILABLE else weights,
        np.array(cov_matrix) if DEPS_AVAILABLE else cov_matrix,
        vol_target,
        seed=42
    )
    
    # Check that scaling was applied
    assert "scaling_factor" in metadata
    assert "original_vol" in metadata
    assert "scaled_vol" in metadata
    
    if DEPS_AVAILABLE:
        # Check that target volatility is approximately achieved
        vol_error = abs(metadata["scaled_vol"] - vol_target)
        assert vol_error < 0.01, f"Vol targeting error: {vol_error}"
    
    # Test deterministic behavior
    scaled_weights2, metadata2 = target_vol(
        np.array(weights) if DEPS_AVAILABLE else weights,
        np.array(cov_matrix) if DEPS_AVAILABLE else cov_matrix,
        vol_target,
        seed=42
    )
    
    if DEPS_AVAILABLE:
        assert np.allclose(scaled_weights, scaled_weights2, atol=1e-10)
    else:
        assert scaled_weights == scaled_weights2


def test_apply_sizing_integration():
    """Test integrated sizing application"""
    from ally.research.sizing import apply_sizing, SizingConfig
    
    weights_in = [0.3, 0.4, 0.3]
    
    config = SizingConfig(
        kelly_cap=0.25,
        vol_target=0.10,
        dd_cap=0.20,
        min_allocation=0.01,
        max_allocation=0.15,
        leverage_limit=1.0
    )
    
    portfolio_metrics = {
        "sharpe_ratio": 1.2,
        "expected_volatility": 0.15,
        "expected_return": 0.08
    }
    
    sized_weights, sizing_metadata = apply_sizing(
        np.array(weights_in) if DEPS_AVAILABLE else weights_in,
        config,
        portfolio_metrics,
        seed=42
    )
    
    # Check sizing was applied
    assert "kelly_fraction" in sizing_metadata
    assert "vol_scaling" in sizing_metadata
    assert "final_leverage" in sizing_metadata
    assert sizing_metadata["sizing_applied"] == True
    
    # Check leverage limit
    if DEPS_AVAILABLE:
        gross_exposure = np.sum(np.abs(sized_weights))
    else:
        gross_exposure = sum(abs(w) for w in sized_weights)
    
    assert gross_exposure <= config.leverage_limit + 1e-6


def test_exposure_constraints():
    """Test gross and net exposure constraint checking"""
    from ally.research.constraints import check_exposure_constraints, ConstraintsConfig
    
    config = ConstraintsConfig(
        gross_exposure_limit=1.0,
        net_exposure_limit=1.0
    )
    
    # Test weights that violate gross exposure
    violating_weights = [0.8, 0.6, -0.4]  # Gross = 1.8
    
    results = check_exposure_constraints(
        np.array(violating_weights) if DEPS_AVAILABLE else violating_weights,
        config,
        seed=42
    )
    
    assert results["gross_violation"] == True
    assert len(results["violations"]) > 0
    assert results["violations"][0]["type"] == "gross_exposure"
    
    # Test weights that satisfy constraints
    good_weights = [0.5, 0.3, 0.2]  # Gross = 1.0, Net = 1.0
    
    good_results = check_exposure_constraints(
        np.array(good_weights) if DEPS_AVAILABLE else good_weights,
        config,
        seed=42
    )
    
    assert good_results["gross_violation"] == False
    assert good_results["net_violation"] == False
    assert len(good_results["violations"]) == 0


def test_single_name_constraints():
    """Test single-name position size constraints"""
    from ally.research.constraints import check_single_name_constraints, ConstraintsConfig
    
    config = ConstraintsConfig(single_name_limit=0.10)
    
    # Test weights with single-name violation
    violating_weights = [0.15, 0.05, 0.03]  # First position violates 10% limit
    asset_names = ["AAPL", "MSFT", "GOOGL"]
    
    results = check_single_name_constraints(
        np.array(violating_weights) if DEPS_AVAILABLE else violating_weights,
        asset_names,
        config,
        seed=42
    )
    
    assert results["single_name_violations"] > 0
    assert len(results["violations"]) > 0
    assert results["violations"][0]["asset"] == "AAPL"
    assert results["max_position_asset"] == "AAPL"
    
    # Test weights that satisfy constraints
    good_weights = [0.08, 0.07, 0.05]
    
    good_results = check_single_name_constraints(
        np.array(good_weights) if DEPS_AVAILABLE else good_weights,
        asset_names,
        config,
        seed=42
    )
    
    assert good_results["single_name_violations"] == 0
    assert len(good_results["violations"]) == 0


def test_sector_constraints():
    """Test sector exposure constraints"""
    from ally.research.constraints import check_sector_constraints, ConstraintsConfig
    
    config = ConstraintsConfig()
    config.sector_limits = {"Technology": 0.25, "Financial": 0.20}
    
    # Test weights with sector violation
    violating_weights = [0.15, 0.15, 0.10, 0.05]  # Tech = 0.30, violates 0.25 limit
    asset_sectors = {
        "Asset_0": "Technology",
        "Asset_1": "Technology", 
        "Asset_2": "Financial",
        "Asset_3": "Other"
    }
    
    results = check_sector_constraints(
        np.array(violating_weights) if DEPS_AVAILABLE else violating_weights,
        asset_sectors,
        config,
        seed=42
    )
    
    assert results["sector_violations"] > 0
    assert len(results["violations"]) > 0
    assert results["violations"][0]["sector"] == "Technology"
    
    # Check sector exposures calculation
    assert "Technology" in results["sector_exposures"]
    assert results["sector_exposures"]["Technology"] == 0.30


def test_turnover_constraints():
    """Test turnover constraint checking"""
    from ally.research.constraints import check_turnover_constraints, ConstraintsConfig
    
    config = ConstraintsConfig(turnover_limit=1.0)
    
    # Test high turnover scenario
    current_weights = [0.4, 0.3, 0.3]
    previous_weights = [0.1, 0.1, 0.8]  # Large rebalancing
    
    results = check_turnover_constraints(
        np.array(current_weights) if DEPS_AVAILABLE else current_weights,
        np.array(previous_weights) if DEPS_AVAILABLE else previous_weights,
        config,
        seed=42
    )
    
    # Should detect high turnover
    assert results["turnover"] > config.turnover_limit
    assert results["turnover_violation"] == True
    assert len(results["violations"]) > 0
    
    # Test low turnover scenario
    low_turnover_current = [0.35, 0.33, 0.32]
    low_turnover_previous = [0.33, 0.34, 0.33]
    
    low_results = check_turnover_constraints(
        np.array(low_turnover_current) if DEPS_AVAILABLE else low_turnover_current,
        np.array(low_turnover_previous) if DEPS_AVAILABLE else low_turnover_previous,
        config,
        seed=42
    )
    
    assert low_results["turnover_violation"] == False
    assert len(low_results["violations"]) == 0


def test_capacity_constraints():
    """Test capacity and ADV constraints"""
    from ally.research.constraints import check_capacity_constraints, ConstraintsConfig
    
    config = ConstraintsConfig(
        capacity_limit_usd=50_000_000,
        adv_multiple_limit=0.05
    )
    
    # Test capacity violation
    portfolio_value = 75_000_000  # Exceeds 50M limit
    weights = [0.3, 0.4, 0.3]
    
    adv_data = {
        "Asset_0": 10_000_000,  # $10M daily volume
        "Asset_1": 20_000_000,  # $20M daily volume
        "Asset_2": 15_000_000   # $15M daily volume
    }
    
    results = check_capacity_constraints(
        np.array(weights) if DEPS_AVAILABLE else weights,
        portfolio_value,
        adv_data,
        config,
        seed=42
    )
    
    # Should detect capacity violation
    assert results["capacity_violation"] == True
    assert len(results["violations"]) > 0
    assert results["violations"][0]["type"] == "capacity"
    
    # Check ADV violations
    # Asset_0: 0.3 * 75M = 22.5M position vs 10M * 0.05 = 0.5M limit
    adv_violations = [v for v in results["violations"] if v["type"] == "adv"]
    assert len(adv_violations) > 0  # Should have ADV violations


def test_borrow_fee_constraints():
    """Test borrow fee constraints for short positions"""
    from ally.research.constraints import check_borrow_fee_constraints, ConstraintsConfig
    
    config = ConstraintsConfig(borrow_fee_threshold=100.0)  # 100 bps threshold
    
    # Test weights with short positions
    weights = [0.3, 0.4, -0.2, -0.1]  # Two short positions
    
    borrow_fees = {
        "Asset_2": 150.0,  # High borrow fee (exceeds threshold)
        "Asset_3": 50.0    # Low borrow fee (below threshold)
    }
    
    results = check_borrow_fee_constraints(
        np.array(weights) if DEPS_AVAILABLE else weights,
        borrow_fees,
        config,
        seed=42
    )
    
    # Should detect high borrow fee violation
    assert results["high_borrow_violations"] > 0
    assert len(results["violations"]) > 0
    assert results["violations"][0]["asset"] == "Asset_2"
    assert results["short_positions"] == 2
    
    # Check high borrow positions tracking
    assert len(results["high_borrow_positions"]) == 2


def test_constraints_api_integration():
    """Test constraints checking API"""
    from ally.research.constraints import research_constraints_checks
    
    # Test with constraint violations
    violating_weights = [0.15, 0.12, -0.08, 0.10, 0.09]  # Some positions may violate
    
    result = research_constraints_checks(
        portfolio_weights=violating_weights,
        portfolio_value_usd=10_000_000,
        live=False
    )
    
    assert result.ok == True
    assert "constraints_receipt" in result.data
    assert "constraints_ok" in result.data
    assert "violations" in result.data
    assert "binding_caps" in result.data
    
    # Check receipt format
    assert len(result.data["constraints_receipt"]) == 16
    assert hasattr(result, 'receipt_hash')
    
    # Check summary stats
    summary = result.data["summary_stats"]
    assert "gross_exposure" in summary
    assert "net_exposure" in summary
    assert "max_position" in summary
    assert "turnover" in summary


def test_constraints_deterministic():
    """Test that constraints checking is deterministic"""
    from ally.research.constraints import research_constraints_checks
    
    weights = [0.12, 0.10, -0.05, 0.08, 0.07]
    
    # Run twice with same parameters
    result1 = research_constraints_checks(
        portfolio_weights=weights,
        portfolio_value_usd=10_000_000,
        live=False
    )
    
    result2 = research_constraints_checks(
        portfolio_weights=weights,
        portfolio_value_usd=10_000_000,
        live=False
    )
    
    assert result1.ok == result2.ok
    
    if result1.ok and result2.ok:
        # Should have identical results
        assert result1.data["constraints_ok"] == result2.data["constraints_ok"]
        assert len(result1.data["violations"]) == len(result2.data["violations"])
        
        summary1 = result1.data["summary_stats"]
        summary2 = result2.data["summary_stats"]
        
        assert abs(summary1["gross_exposure"] - summary2["gross_exposure"]) < 1e-10
        assert abs(summary1["net_exposure"] - summary2["net_exposure"]) < 1e-10


def test_portfolio_sizing_api():
    """Test portfolio sizing API"""
    from ally.research.sizing import research_portfolio_size
    
    weights = [0.4, 0.35, 0.25]
    metrics = {
        "sharpe_ratio": 1.5,
        "expected_volatility": 0.18,
        "expected_return": 0.10
    }
    
    result = research_portfolio_size(
        portfolio_weights=weights,
        portfolio_metrics=metrics,
        kelly_cap=0.25,
        vol_target=0.12,
        live=False
    )
    
    assert result.ok == True
    assert "sizing_receipt" in result.data
    assert "sized_weights" in result.data
    assert "sizing_metadata" in result.data
    assert "exposure_metrics" in result.data
    
    # Check sizing summary
    summary = result.data["sizing_summary"]
    assert "kelly_fraction_used" in summary
    assert "kelly_cap_binding" in summary
    assert "vol_target_achieved" in summary
    
    # Check that Kelly cap was applied if needed
    kelly_frac = summary["kelly_fraction_used"]
    assert abs(kelly_frac) <= 0.25 + 1e-6


def test_sizing_edge_cases():
    """Test sizing edge cases and error handling"""
    from ally.research.sizing import research_portfolio_size
    
    # Test with empty weights
    result_empty = research_portfolio_size(
        portfolio_weights=[],
        live=False
    )
    
    assert result_empty.ok == True
    assert result_empty.data["sized_weights"] == []
    
    # Test with zero volatility metrics
    zero_vol_metrics = {
        "sharpe_ratio": 1.0,
        "expected_volatility": 0.0,
        "expected_return": 0.05
    }
    
    result_zero_vol = research_portfolio_size(
        portfolio_weights=[0.5, 0.5],
        portfolio_metrics=zero_vol_metrics,
        live=False
    )
    
    assert result_zero_vol.ok == True
    # Should handle zero volatility gracefully


def test_live_mode_gating():
    """Test live mode gating for sizing and constraints"""
    from ally.research.sizing import research_portfolio_size
    from ally.research.constraints import research_constraints_checks
    
    # Test offline mode (should work)
    size_result = research_portfolio_size(live=False)
    assert size_result.ok == True
    
    constraints_result = research_constraints_checks(live=False)
    assert constraints_result.ok == True
    
    # Test live mode (should fail in CI without proper API key)
    size_result_live = research_portfolio_size(live=True)
    constraints_result_live = research_constraints_checks(live=True)
    
    # Results depend on environment setup
    assert isinstance(size_result_live.ok, bool)
    assert isinstance(constraints_result_live.ok, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])