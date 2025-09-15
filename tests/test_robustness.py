#!/usr/bin/env python3
"""
Robustness Battery tests - Phase 5.3 testing
"""

import os
import json
from datetime import datetime, timedelta

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
    class MockSeries:
        def __init__(self, data=None, index=None):
            self.data = data or []
            self.index = index or list(range(len(self.data) if data else 0))
            
        def __len__(self):
            return len(self.data)
            
        def __getitem__(self, key):
            return self.data[key]
            
        def __setitem__(self, key, value):
            self.data[key] = value
            
        def mean(self):
            return sum(self.data) / len(self.data) if self.data else 0
            
        def std(self):
            if not self.data:
                return 0
            mean_val = self.mean()
            return (sum((x - mean_val) ** 2 for x in self.data) / len(self.data)) ** 0.5
    
    pd = type('pd', (), {
        'Series': MockSeries,
        'date_range': lambda start, periods, freq: [f"{start}+{i}d" for i in range(periods)]
    })
    
    np = type('np', (), {
        'random': type('random', (), {
            'seed': lambda x: None,
            'normal': lambda mu, sigma, n: [mu + sigma * (i % 3 - 1) for i in range(n)],
            'RandomState': lambda seed: type('RS', (), {
                'normal': lambda mu, sigma, n: [mu + sigma * (i % 3 - 1) for i in range(n)]
            })()
        })(),
        'sqrt': lambda x: x ** 0.5,
        'mean': lambda x: sum(x) / len(x) if x else 0,
        'isnan': lambda x: False
    })
    
    def pytest_raises(exc_type, match=None):
        class MockRaises:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
        return MockRaises()
    
    pytest = type('pytest', (), {
        'raises': pytest_raises,
        'main': lambda args: print("pytest not available - using mock tests")
    })()


def test_bootstrap_resampling_basic():
    """Test basic bootstrap resampling functionality"""
    from ally.research.robustness import bootstrap_resample_returns
    
    # Create deterministic test returns
    np.random.seed(42)
    returns = pd.Series([0.01, -0.005, 0.015, -0.01, 0.02, -0.008, 0.012, 0.003])
    
    # Test bootstrap resampling
    resampled = bootstrap_resample_returns(
        returns=returns,
        n_samples=100,
        block_size=2,
        seed=42
    )
    
    # Validate structure
    assert len(resampled) == 100
    assert all(len(sample) == len(returns) for sample in resampled)
    
    # Test deterministic behavior
    resampled2 = bootstrap_resample_returns(
        returns=returns,
        n_samples=100,
        block_size=2,
        seed=42
    )
    
    for i in range(len(resampled)):
        pd.testing.assert_series_equal(resampled[i], resampled2[i])


def test_bootstrap_block_sizes():
    """Test bootstrap with different block sizes"""
    from ally.research.robustness import bootstrap_resample_returns
    
    returns = pd.Series(np.random.RandomState(42).normal(0.001, 0.02, 50))
    
    # Test different block sizes
    for block_size in [1, 3, 5, 10]:
        resampled = bootstrap_resample_returns(
            returns=returns,
            n_samples=10,
            block_size=block_size,
            seed=42
        )
        
        assert len(resampled) == 10
        assert all(len(sample) == len(returns) for sample in resampled)
        
        # Check that samples preserve some block structure
        for sample in resampled[:3]:  # Check first few samples
            assert isinstance(sample, pd.Series)
            assert not sample.isna().any()


def test_regime_shuffling():
    """Test regime shuffling functionality"""
    from ally.research.robustness import shuffle_regime_blocks
    
    # Create test data with clear regimes
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    returns = pd.Series(index=dates, dtype=float)
    
    # High vol regime (first 30 days)
    returns.iloc[:30] = np.random.RandomState(42).normal(0, 0.03, 30)
    # Low vol regime (next 40 days)  
    returns.iloc[30:70] = np.random.RandomState(43).normal(0.001, 0.01, 40)
    # Medium vol regime (last 30 days)
    returns.iloc[70:] = np.random.RandomState(44).normal(-0.001, 0.02, 30)
    
    # Test regime shuffling
    shuffled = shuffle_regime_blocks(
        returns=returns,
        regime_length_days=30,
        n_shuffles=50,
        seed=42
    )
    
    # Validate structure
    assert len(shuffled) == 50
    assert all(len(sample) == len(returns) for sample in shuffled)
    
    # Test deterministic behavior
    shuffled2 = shuffle_regime_blocks(
        returns=returns,
        regime_length_days=30,
        n_shuffles=50,
        seed=42
    )
    
    for i in range(len(shuffled)):
        pd.testing.assert_series_equal(shuffled[i], shuffled2[i])


def test_noise_jitter_application():
    """Test noise jitter and volatility shock application"""
    from ally.research.robustness import apply_noise_jitter, apply_volatility_shock
    
    # Create test returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, 100))
    
    # Test noise jitter
    jittered = apply_noise_jitter(
        returns=returns,
        noise_std_ratio=0.1,
        seed=42
    )
    
    assert len(jittered) == len(returns)
    assert not jittered.equals(returns)  # Should be different
    
    # Jittered returns should have higher volatility
    orig_vol = returns.std()
    jitter_vol = jittered.std()
    assert jitter_vol > orig_vol
    
    # Test volatility shock
    shocked = apply_volatility_shock(
        returns=returns,
        vol_multiplier=1.5,
        seed=42
    )
    
    assert len(shocked) == len(returns)
    shock_vol = shocked.std()
    assert abs(shock_vol / orig_vol - 1.5) < 0.1  # Within 10% of target multiplier
    
    # Test deterministic behavior
    jittered2 = apply_noise_jitter(returns, noise_std_ratio=0.1, seed=42)
    shocked2 = apply_volatility_shock(returns, vol_multiplier=1.5, seed=42)
    
    pd.testing.assert_series_equal(jittered, jittered2)
    pd.testing.assert_series_equal(shocked, shocked2)


def test_robustness_battery_run():
    """Test full robustness battery execution"""
    from ally.research.robustness import research_robustness_battery
    
    # Mock strategy function
    def mock_strategy(returns_series):
        """Simple mean reversion strategy for testing"""
        # Simple moving average crossover
        short_ma = returns_series.rolling(5).mean()
        long_ma = returns_series.rolling(10).mean()
        
        signals = (short_ma > long_ma).astype(int)
        strategy_returns = signals.shift(1) * returns_series
        
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
        return {
            "sharpe_ratio": sharpe,
            "annual_return": strategy_returns.mean() * 252,
            "annual_volatility": strategy_returns.std() * np.sqrt(252),
            "strategy_returns": strategy_returns
        }
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(
        index=dates,
        data=np.random.normal(0.0005, 0.015, 252)
    )
    
    # Run robustness battery
    result = research_robustness_battery(
        returns_series=returns,
        strategy_function=mock_strategy,
        n_bootstrap=20,
        n_regime_shuffle=15,
        n_noise_tests=10,
        pass_threshold_pct=70.0,
        live=False
    )
    
    # Validate result structure
    assert result.ok == True
    assert "robustness_receipt" in result.data
    assert "baseline_metrics" in result.data
    assert "stress_tests" in result.data
    assert "summary_statistics" in result.data
    assert "pass_rates" in result.data
    
    # Check stress test components
    stress_tests = result.data["stress_tests"]
    assert "bootstrap_tests" in stress_tests
    assert "regime_shuffle_tests" in stress_tests
    assert "noise_jitter_tests" in stress_tests
    assert "volatility_shock_tests" in stress_tests
    
    # Validate pass rates
    pass_rates = result.data["pass_rates"]
    assert "bootstrap_pass_rate" in pass_rates
    assert "regime_shuffle_pass_rate" in pass_rates
    assert "noise_tests_pass_rate" in pass_rates
    assert "overall_pass_rate" in pass_rates
    
    # Check that all pass rates are between 0 and 1
    for rate_name, rate_value in pass_rates.items():
        assert 0 <= rate_value <= 1, f"{rate_name} should be between 0 and 1"


def test_robustness_battery_failure_scenarios():
    """Test robustness battery with strategies that should fail"""
    from ally.research.robustness import research_robustness_battery
    
    def unstable_strategy(returns_series):
        """Strategy that should fail robustness tests"""
        # Extremely volatile strategy that's sensitive to noise
        signals = (returns_series > returns_series.quantile(0.9)).astype(int)
        strategy_returns = signals.shift(1) * returns_series * 10  # High leverage
        
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
        return {
            "sharpe_ratio": sharpe,
            "annual_return": strategy_returns.mean() * 252,
            "annual_volatility": strategy_returns.std() * np.sqrt(252)
        }
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')  
    returns = pd.Series(
        index=dates,
        data=np.random.normal(0.001, 0.02, 100)
    )
    
    # Run with high pass threshold (should fail)
    result = research_robustness_battery(
        returns_series=returns,
        strategy_function=unstable_strategy,
        n_bootstrap=10,
        n_regime_shuffle=10, 
        n_noise_tests=10,
        pass_threshold_pct=90.0,  # High threshold
        live=False
    )
    
    # Should still return results but likely fail threshold
    assert result.ok == True
    pass_rates = result.data["pass_rates"]
    
    # At least one component should have low pass rate
    assert (pass_rates["bootstrap_pass_rate"] < 0.9 or 
            pass_rates["regime_shuffle_pass_rate"] < 0.9 or
            pass_rates["noise_tests_pass_rate"] < 0.9)


def test_robustness_edge_cases():
    """Test robustness battery edge cases"""
    from ally.research.robustness import research_robustness_battery
    
    def simple_strategy(returns_series):
        """Simple buy-and-hold strategy"""
        return {
            "sharpe_ratio": 1.0,
            "annual_return": 0.08,
            "annual_volatility": 0.15
        }
    
    # Test with very short time series
    short_returns = pd.Series([0.01, -0.005, 0.02])
    
    result = research_robustness_battery(
        returns_series=short_returns,
        strategy_function=simple_strategy,
        n_bootstrap=5,
        n_regime_shuffle=3,
        n_noise_tests=3,
        live=False
    )
    
    # Should handle gracefully
    assert result.ok == True
    
    # Test with constant returns (edge case)
    constant_returns = pd.Series([0.001] * 50)
    
    result = research_robustness_battery(
        returns_series=constant_returns,
        strategy_function=simple_strategy,
        n_bootstrap=5,
        n_regime_shuffle=3,
        n_noise_tests=3,
        live=False
    )
    
    # Should handle gracefully
    assert result.ok == True


def test_robustness_receipts_generation():
    """Test that robustness operations generate proper receipts"""
    from ally.research.robustness import research_robustness_battery
    
    def test_strategy(returns_series):
        return {
            "sharpe_ratio": 0.8,
            "annual_return": 0.06,
            "annual_volatility": 0.12
        }
    
    # Create test data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.015, 100))
    
    result = research_robustness_battery(
        returns_series=returns,
        strategy_function=test_strategy,
        n_bootstrap=5,
        n_regime_shuffle=5,
        n_noise_tests=5,
        live=False
    )
    
    assert result.ok == True
    assert "robustness_receipt" in result.data
    assert len(result.data["robustness_receipt"]) == 16
    assert hasattr(result, 'receipt_hash')
    assert len(result.receipt_hash) == 40


def test_robustness_deterministic_behavior():
    """Test that robustness tests are deterministic"""
    from ally.research.robustness import research_robustness_battery
    
    def deterministic_strategy(returns_series):
        """Deterministic strategy for testing"""
        mean_return = returns_series.mean()
        vol = returns_series.std()
        sharpe = mean_return / (vol + 1e-8) * np.sqrt(252)
        
        return {
            "sharpe_ratio": float(sharpe),
            "annual_return": float(mean_return * 252),
            "annual_volatility": float(vol * np.sqrt(252))
        }
    
    # Create deterministic test data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.015, 80))
    
    # Run twice with same parameters
    result1 = research_robustness_battery(
        returns_series=returns,
        strategy_function=deterministic_strategy,
        n_bootstrap=10,
        n_regime_shuffle=8,
        n_noise_tests=6,
        pass_threshold_pct=75.0,
        live=False
    )
    
    result2 = research_robustness_battery(
        returns_series=returns,
        strategy_function=deterministic_strategy,
        n_bootstrap=10,
        n_regime_shuffle=8,
        n_noise_tests=6,
        pass_threshold_pct=75.0,
        live=False
    )
    
    # Results should be identical
    assert result1.ok == result2.ok
    
    if result1.ok and result2.ok:
        pass_rates1 = result1.data["pass_rates"]
        pass_rates2 = result2.data["pass_rates"]
        
        for key in pass_rates1:
            assert abs(pass_rates1[key] - pass_rates2[key]) < 1e-10


def test_stress_test_parameter_validation():
    """Test parameter validation for stress tests"""
    from ally.research.robustness import bootstrap_resample_returns, shuffle_regime_blocks
    from ally.research.robustness import apply_noise_jitter, apply_volatility_shock
    
    returns = pd.Series([0.01, -0.005, 0.015, -0.01])
    
    # Test bootstrap with invalid parameters
    with pytest.raises(ValueError, match="n_samples must be positive"):
        bootstrap_resample_returns(returns, n_samples=0, seed=42)
    
    with pytest.raises(ValueError, match="block_size must be positive"):
        bootstrap_resample_returns(returns, n_samples=10, block_size=0, seed=42)
    
    # Test regime shuffle with invalid parameters
    with pytest.raises(ValueError, match="n_shuffles must be positive"):
        shuffle_regime_blocks(returns, regime_length_days=5, n_shuffles=0, seed=42)
    
    with pytest.raises(ValueError, match="regime_length_days must be positive"):
        shuffle_regime_blocks(returns, regime_length_days=0, n_shuffles=5, seed=42)
    
    # Test noise jitter with invalid parameters
    with pytest.raises(ValueError, match="noise_std_ratio must be non-negative"):
        apply_noise_jitter(returns, noise_std_ratio=-0.1, seed=42)
    
    # Test volatility shock with invalid parameters
    with pytest.raises(ValueError, match="vol_multiplier must be positive"):
        apply_volatility_shock(returns, vol_multiplier=0, seed=42)


def test_robustness_summary_statistics():
    """Test summary statistics generation"""
    from ally.research.robustness import research_robustness_battery
    
    def consistent_strategy(returns_series):
        """Strategy that should be fairly consistent"""
        # Simple momentum strategy
        momentum = returns_series.rolling(3).mean()
        signals = (momentum > 0).astype(int)
        strategy_returns = signals.shift(1) * returns_series
        
        return {
            "sharpe_ratio": 0.9,  # Consistent value
            "annual_return": 0.08,
            "annual_volatility": 0.14
        }
    
    # Create test data
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.015, 150))
    
    result = research_robustness_battery(
        returns_series=returns,
        strategy_function=consistent_strategy,
        n_bootstrap=15,
        n_regime_shuffle=12,
        n_noise_tests=10,
        live=False
    )
    
    assert result.ok == True
    
    # Check summary statistics structure
    summary = result.data["summary_statistics"]
    assert "total_stress_tests" in summary
    assert "tests_passed" in summary
    assert "overall_pass_rate" in summary
    assert "robustness_score" in summary
    assert "stress_test_breakdown" in summary
    
    # Validate summary values
    assert summary["total_stress_tests"] == (15 + 12 + 10 + 10)  # All test types
    assert 0 <= summary["overall_pass_rate"] <= 1
    assert 0 <= summary["robustness_score"] <= 100


def test_block_bootstrap_time_series_preservation():
    """Test that block bootstrap preserves time series structure"""
    from ally.research.robustness import bootstrap_resample_returns
    
    # Create autocorrelated time series
    np.random.seed(42)
    n = 50
    returns = pd.Series(index=range(n))
    returns.iloc[0] = 0.01
    
    # Generate AR(1) process
    phi = 0.3  # Autocorrelation coefficient
    for i in range(1, n):
        returns.iloc[i] = phi * returns.iloc[i-1] + np.random.normal(0, 0.01)
    
    # Test that larger block sizes preserve more autocorrelation
    small_block_samples = bootstrap_resample_returns(
        returns, n_samples=20, block_size=1, seed=42
    )
    
    large_block_samples = bootstrap_resample_returns(
        returns, n_samples=20, block_size=5, seed=42
    )
    
    # Calculate average autocorrelation for each set
    def avg_autocorr(samples):
        autocorrs = []
        for sample in samples:
            if len(sample) > 1:
                autocorr = sample.autocorr(lag=1)
                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)
        return np.mean(autocorrs) if autocorrs else 0
    
    small_autocorr = avg_autocorr(small_block_samples)
    large_autocorr = avg_autocorr(large_block_samples)
    
    # Larger blocks should preserve more autocorrelation structure
    assert large_autocorr >= small_autocorr


def test_regime_detection_and_shuffling():
    """Test regime detection and effective shuffling"""
    from ally.research.robustness import shuffle_regime_blocks
    
    # Create data with distinct regimes
    n_regime = 20
    regime1 = pd.Series(np.random.RandomState(42).normal(0.002, 0.01, n_regime))  # Low vol, positive mean
    regime2 = pd.Series(np.random.RandomState(43).normal(-0.001, 0.03, n_regime))  # High vol, negative mean
    regime3 = pd.Series(np.random.RandomState(44).normal(0.001, 0.015, n_regime))  # Medium vol
    
    # Combine regimes
    returns = pd.concat([regime1, regime2, regime3], ignore_index=True)
    
    # Shuffle regimes
    shuffled_samples = shuffle_regime_blocks(
        returns=returns,
        regime_length_days=n_regime,
        n_shuffles=10,
        seed=42
    )
    
    # Verify that shuffled samples have different regime ordering
    original_first_regime = returns.iloc[:n_regime]
    
    different_first_regimes = 0
    for sample in shuffled_samples:
        sample_first_regime = sample.iloc[:n_regime]
        if not sample_first_regime.equals(original_first_regime):
            different_first_regimes += 1
    
    # At least some shuffles should have different first regime
    assert different_first_regimes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])