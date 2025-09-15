#!/usr/bin/env python3
"""
Time-series cross-validation tests - Phase 5.1 testing
"""

import os
import pytest
import tempfile
import yaml
from datetime import datetime, timezone

# Ensure CI stays dry
os.environ["ALLY_LIVE"] = "0"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_expanding_window_folds():
    """Test expanding window CV fold generation"""
    from ally.research.ts_cv import generate_expanding_window_folds
    
    folds = generate_expanding_window_folds(
        start_date="2020-01-01",
        end_date="2023-12-31",
        n_splits=4,
        embargo_days=5,
        min_train_days=252
    )
    
    assert len(folds) >= 2, "Should generate at least 2 folds"
    
    # Validate expanding window properties
    for i, fold in enumerate(folds):
        assert fold.fold_id == i
        assert fold.method.value == "expanding_window"
        assert fold.embargo_days == 5
        
        # Check date ordering
        assert fold.train_start < fold.train_end
        assert fold.train_end < fold.test_start
        assert fold.test_start < fold.test_end
        
        # Training window should expand
        if i > 0:
            prev_fold = folds[i-1]
            current_train_days = (pd.to_datetime(fold.train_end) - pd.to_datetime(fold.train_start)).days
            prev_train_days = (pd.to_datetime(prev_fold.train_end) - pd.to_datetime(prev_fold.train_start)).days
            assert current_train_days >= prev_train_days, "Training window should expand"


def test_rolling_window_folds():
    """Test rolling window CV fold generation"""
    from ally.research.ts_cv import generate_rolling_window_folds
    
    folds = generate_rolling_window_folds(
        start_date="2020-01-01",
        end_date="2023-12-31",
        n_splits=3,
        train_days=252,
        embargo_days=7,
        test_days=63
    )
    
    assert len(folds) >= 2, "Should generate at least 2 folds"
    
    # Validate rolling window properties
    train_days_expected = 252
    test_days_expected = 63
    
    for fold in folds:
        assert fold.method.value == "rolling_window"
        assert fold.embargo_days == 7
        
        # Check fixed window sizes
        import pandas as pd
        actual_train_days = (pd.to_datetime(fold.train_end) - pd.to_datetime(fold.train_start)).days
        actual_test_days = (pd.to_datetime(fold.test_end) - pd.to_datetime(fold.test_start)).days
        
        assert abs(actual_train_days - train_days_expected) <= 1, "Training window should be fixed size"
        assert abs(actual_test_days - test_days_expected) <= 1, "Test window should be fixed size"


def test_purged_cv_folds():
    """Test purged CV fold generation"""
    from ally.research.ts_cv import generate_purged_cv_folds
    
    folds = generate_purged_cv_folds(
        start_date="2020-01-01",
        end_date="2023-12-31",
        n_splits=3,
        purge_pct=0.02,
        embargo_days=5
    )
    
    assert len(folds) >= 1, "Should generate at least 1 fold"
    
    # Validate purged CV properties
    for fold in folds:
        assert fold.method.value == "purged_cv"
        assert fold.embargo_days == 5
        assert fold.purge_pct == 0.02
        
        # Check date ordering
        assert fold.train_start < fold.train_end
        assert fold.train_end < fold.test_start  # Gap for embargo + purge
        assert fold.test_start < fold.test_end


def test_tscv_fold_generation_via_api():
    """Test TS-CV fold generation via registered API"""
    from ally.research.ts_cv import research_ts_cv_folds
    
    # Test different methods
    methods = ["expanding_window", "rolling_window", "purged_cv"]
    
    for method in methods:
        result = research_ts_cv_folds(
            start_date="2020-01-01",
            end_date="2023-06-30",
            method=method,
            n_splits=3,
            embargo_days=5,
            live=False
        )
        
        assert result.ok == True, f"Fold generation failed for {method}: {result.errors}"
        assert "folds_receipt" in result.data
        assert result.data["method"] == method
        assert result.data["n_folds"] >= 1
        assert len(result.data["folds"]) == result.data["n_folds"]
        
        # Check fold structure
        for fold_dict in result.data["folds"]:
            assert "fold_id" in fold_dict
            assert "train_start" in fold_dict
            assert "train_end" in fold_dict
            assert "test_start" in fold_dict
            assert "test_end" in fold_dict
            assert "embargo_days" in fold_dict


def test_tscv_fold_backtest():
    """Test TS-CV fold backtesting"""
    from ally.research.ts_cv import run_tscv_fold, TSCVFold, TSCVMethod
    from ally.research.spec import StrategySpec
    
    # Create test spec
    spec_data = {
        'name': 'Test-TSCV-Strategy',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 25},
        'costs': {'bps_per_turnover': 12.0},
        'backtest': {'start': '2020-01-01', 'end': '2023-12-31', 'benchmark': 'SPY', 'seed': 42},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        spec = StrategySpec.from_yaml(spec_path)
        
        # Create test fold
        fold = TSCVFold(
            fold_id=0,
            method=TSCVMethod.EXPANDING_WINDOW,
            train_start="2020-01-01",
            train_end="2020-12-31",
            test_start="2021-01-06",  # 5-day embargo
            test_end="2021-03-31",
            embargo_days=5
        )
        
        # Run fold backtest
        fold_result = run_tscv_fold(spec, fold, live=False, seed_offset=200)
        
        # Validate results structure
        assert "fold_id" in fold_result
        assert "cv_method" in fold_result
        assert "train_results" in fold_result
        assert "test_results" in fold_result
        assert "cv_metrics" in fold_result
        assert "embargo_respected" in fold_result
        
        # Check train results
        train_results = fold_result["train_results"]
        assert "annual_return" in train_results
        assert "sharpe_ratio" in train_results
        assert "period_type" in train_results
        assert train_results["period_type"] == "train"
        
        # Check test results
        test_results = fold_result["test_results"]
        assert "annual_return" in test_results
        assert "sharpe_ratio" in test_results
        assert "period_type" in test_results
        assert test_results["period_type"] == "test"
        
        # Check CV metrics
        cv_metrics = fold_result["cv_metrics"]
        assert "sharpe_degradation" in cv_metrics
        assert "return_consistency" in cv_metrics
        assert "volatility_increase" in cv_metrics
        
    finally:
        os.unlink(spec_path)


def test_tscv_aggregation():
    """Test TS-CV results aggregation"""
    from ally.research.ts_cv import aggregate_tscv_results
    
    # Create mock fold results
    fold_results = [
        {
            "fold_id": 0,
            "cv_method": "expanding_window",
            "train_results": {"sharpe_ratio": 1.4, "annual_return": 0.16},
            "test_results": {"sharpe_ratio": 0.9, "annual_return": 0.11},
            "cv_metrics": {"sharpe_degradation": 0.64, "return_consistency": 0.69}
        },
        {
            "fold_id": 1,
            "cv_method": "expanding_window", 
            "train_results": {"sharpe_ratio": 1.6, "annual_return": 0.19},
            "test_results": {"sharpe_ratio": 1.1, "annual_return": 0.13},
            "cv_metrics": {"sharpe_degradation": 0.69, "return_consistency": 0.68}
        },
        {
            "fold_id": 2,
            "cv_method": "expanding_window",
            "train_results": {"sharpe_ratio": 1.2, "annual_return": 0.14},
            "test_results": {"sharpe_ratio": 0.7, "annual_return": 0.09},
            "cv_metrics": {"sharpe_degradation": 0.58, "return_consistency": 0.64}
        }
    ]
    
    # Test aggregation
    cv_stats = aggregate_tscv_results(fold_results)
    
    # Validate structure
    assert "n_folds_total" in cv_stats
    assert "n_folds_successful" in cv_stats
    assert "train_performance" in cv_stats
    assert "test_performance" in cv_stats
    assert "generalization" in cv_stats
    assert "cv_score" in cv_stats
    
    # Check calculations
    assert cv_stats["n_folds_total"] == 3
    assert cv_stats["n_folds_successful"] == 3
    assert cv_stats["success_rate"] == 1.0
    
    # Check test performance (primary metric)
    test_perf = cv_stats["test_performance"]
    expected_mean_sharpe = (0.9 + 1.1 + 0.7) / 3
    assert abs(test_perf["sharpe_mean"] - expected_mean_sharpe) < 1e-6
    assert test_perf["positive_sharpe_rate"] == 1.0  # All positive
    
    # Check CV score
    assert cv_stats["cv_score"] == expected_mean_sharpe
    
    # Check generalization metrics
    gen_metrics = cv_stats["generalization"]
    assert "mean_degradation_ratio" in gen_metrics
    assert "overfitting_indicator" in gen_metrics
    
    # Test with failed fold
    fold_results_with_failure = fold_results + [{"fold_id": 3, "error": "Mock error"}]
    cv_stats_failure = aggregate_tscv_results(fold_results_with_failure)
    
    assert cv_stats_failure["n_folds_total"] == 4
    assert cv_stats_failure["n_folds_successful"] == 3
    assert cv_stats_failure["n_folds_failed"] == 1
    assert cv_stats_failure["failed_fold_ids"] == [3]


def test_tscv_full_pipeline():
    """Test complete time-series CV pipeline"""
    from ally.research.ts_cv import research_ts_cv_run
    
    # Create test spec
    spec_data = {
        'name': 'Full-Pipeline-TSCV-Test',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 30},
        'costs': {'bps_per_turnover': 15.0},
        'backtest': {'start': '2018-01-01', 'end': '2023-12-31', 'benchmark': 'SPY', 'seed': 456},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        # Test different CV methods
        for method in ["expanding_window", "rolling_window", "purged_cv"]:
            cv_result = research_ts_cv_run(
                spec_path=spec_path,
                method=method,
                n_splits=3,
                embargo_days=7,
                cv_score_threshold=0.1,  # Low threshold for test
                live=False
            )
            
            # Validate results structure
            assert "tscv_receipt" in cv_result.data
            assert "spec_name" in cv_result.data
            assert "fold_results" in cv_result.data
            assert "cv_statistics" in cv_result.data
            assert "threshold_analysis" in cv_result.data
            
            # Check CV parameters
            params = cv_result.data["cv_parameters"]
            assert params["method"] == method
            assert params["n_splits"] == 3
            assert params["embargo_days"] == 7
            
            # Check fold receipts
            fold_receipts = cv_result.data["fold_receipts"]
            assert len(fold_receipts) > 0
            assert all(len(receipt) == 16 for receipt in fold_receipts)
            
            # Check CV statistics
            cv_stats = cv_result.data["cv_statistics"]
            assert "cv_score" in cv_stats
            assert "test_performance" in cv_stats
            assert "generalization" in cv_stats
            
    finally:
        os.unlink(spec_path)


def test_tscv_method_validation():
    """Test TS-CV method validation"""
    from ally.research.ts_cv import research_ts_cv_run
    
    spec_data = {
        'name': 'Method-Validation-Test',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 20},
        'costs': {'bps_per_turnover': 10.0},
        'backtest': {'start': '2020-01-01', 'end': '2022-12-31', 'benchmark': 'SPY', 'seed': 789},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        # Test invalid method
        invalid_result = research_ts_cv_run(
            spec_path=spec_path,
            method="invalid_method",
            live=False
        )
        
        assert invalid_result.ok == False
        assert "Invalid CV method" in str(invalid_result.errors)
        
    finally:
        os.unlink(spec_path)


def test_tscv_deterministic_behavior():
    """Test that TS-CV produces deterministic results"""
    from ally.research.ts_cv import research_ts_cv_run
    
    spec_data = {
        'name': 'Deterministic-TSCV-Test',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 25},
        'costs': {'bps_per_turnover': 13.0},
        'backtest': {'start': '2020-01-01', 'end': '2022-12-31', 'benchmark': 'SPY', 'seed': 999},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        # Run twice with same parameters
        result1 = research_ts_cv_run(
            spec_path=spec_path,
            method="expanding_window",
            n_splits=2,
            embargo_days=5,
            cv_score_threshold=0.0,
            live=False
        )
        
        result2 = research_ts_cv_run(
            spec_path=spec_path,
            method="expanding_window",
            n_splits=2,
            embargo_days=5,
            cv_score_threshold=0.0,
            live=False
        )
        
        # Results should be identical (deterministic)
        assert result1.ok == result2.ok
        if result1.ok and result2.ok:
            cv_score1 = result1.data["cv_statistics"]["cv_score"]
            cv_score2 = result2.data["cv_statistics"]["cv_score"]
            assert abs(cv_score1 - cv_score2) < 1e-6, "Results should be deterministic"
        
    finally:
        os.unlink(spec_path)


def test_tscv_insufficient_data():
    """Test TS-CV with insufficient data"""
    from ally.research.ts_cv import generate_expanding_window_folds
    
    # Test with insufficient data for expanding window
    with pytest.raises(ValueError, match="Insufficient data"):
        generate_expanding_window_folds(
            start_date="2023-01-01",
            end_date="2023-02-28",  # Only 2 months
            n_splits=5,
            embargo_days=5,
            min_train_days=252
        )


def test_tscv_error_handling():
    """Test TS-CV error handling"""
    from ally.research.ts_cv import research_ts_cv_run
    
    # Test with non-existent spec file
    result = research_ts_cv_run(
        spec_path="/nonexistent/spec.yaml",
        live=False
    )
    
    assert result.ok == False
    assert len(result.errors) > 0
    assert "receipt_hash" in result.__dict__  # Should still generate error receipt


def test_tscv_receipts():
    """Test that TS-CV generates proper receipts"""
    from ally.research.ts_cv import research_ts_cv_run
    
    spec_data = {
        'name': 'TSCV-Receipt-Test',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 40},
        'costs': {'bps_per_turnover': 16.0},
        'backtest': {'start': '2020-01-01', 'end': '2022-06-30', 'benchmark': 'SPY', 'seed': 321},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        result = research_ts_cv_run(
            spec_path=spec_path,
            method="rolling_window",
            n_splits=2,
            live=False
        )
        
        # Check main receipt
        assert "tscv_receipt" in result.data
        assert len(result.data["tscv_receipt"]) == 16
        
        # Check fold receipts
        fold_receipts = result.data["fold_receipts"]
        assert len(fold_receipts) >= 1
        assert all(len(receipt) == 16 for receipt in fold_receipts)
        
        # Check receipt hash on result
        assert hasattr(result, 'receipt_hash')
        assert len(result.receipt_hash) == 40  # Full SHA-1
        
    finally:
        os.unlink(spec_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])