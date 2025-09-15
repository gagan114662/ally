#!/usr/bin/env python3
"""
Walk-forward analysis tests - Phase 5.1 testing
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


def test_walkforward_window_generation():
    """Test walk-forward window generation"""
    from ally.research.walkforward import generate_walkforward_windows, research_walkforward_windows
    
    # Test basic window generation
    windows = generate_walkforward_windows(
        start_date="2020-01-01",
        end_date="2023-12-31",
        n_folds=4,
        embargo_days=5,
        min_is_days=252,
        min_oos_days=63
    )
    
    assert len(windows) >= 2, "Should generate at least 2 windows"
    
    # Validate window structure
    for i, window in enumerate(windows):
        assert window.fold_id == i
        assert window.embargo_days == 5
        
        # Check date ordering
        assert window.is_start < window.is_end
        assert window.is_end < window.oos_start
        assert window.oos_start < window.oos_end
        
        # Check embargo period
        from datetime import timedelta
        import pandas as pd
        is_end_dt = pd.to_datetime(window.is_end)
        oos_start_dt = pd.to_datetime(window.oos_start)
        embargo_actual = (oos_start_dt - is_end_dt).days
        assert embargo_actual >= 5, f"Embargo period too short: {embargo_actual} days"
    
    # Test via registered tool
    windows_result = research_walkforward_windows(
        start_date="2021-01-01",
        end_date="2023-06-30",
        n_folds=3,
        embargo_days=7,
        live=False
    )
    
    assert windows_result.ok == True, f"Window generation failed: {windows_result.errors}"
    assert "windows_receipt" in windows_result.data
    assert windows_result.data["n_windows"] >= 2
    assert windows_result.data["embargo_days"] == 7


def test_walkforward_insufficient_data():
    """Test walk-forward with insufficient data"""
    from ally.research.walkforward import generate_walkforward_windows, research_walkforward_run
    
    # Test with too little data
    with pytest.raises(ValueError, match="Insufficient data"):
        generate_walkforward_windows(
            start_date="2023-01-01",
            end_date="2023-03-31",  # Only 3 months
            n_folds=6,
            embargo_days=5,
            min_is_days=252,
            min_oos_days=63
        )
    
    # Test via full pipeline with insufficient data
    spec_data = {
        'name': 'Insufficient-Data-Test',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 20},
        'costs': {'bps_per_turnover': 15.0},
        'backtest': {'start': '2023-01-01', 'end': '2023-02-28', 'benchmark': 'SPY', 'seed': 42},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        result = research_walkforward_run(
            spec_path=spec_path,
            n_folds=6,
            live=False
        )
        
        # Should fail with friendly error
        assert result.ok == False
        assert any("Insufficient data" in error for error in result.errors)
        
    finally:
        os.unlink(spec_path)


def test_walkforward_fold_backtest():
    """Test individual fold backtesting"""
    from ally.research.walkforward import run_fold_backtest, WalkForwardWindow
    from ally.research.spec import StrategySpec
    
    # Create test spec
    spec_data = {
        'name': 'Test-WF-Strategy',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 20},
        'costs': {'bps_per_turnover': 15.0},
        'backtest': {'start': '2020-01-01', 'end': '2023-12-31', 'benchmark': 'SPY', 'seed': 42},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        spec = StrategySpec.from_yaml(spec_path)
        
        # Create test window
        window = WalkForwardWindow(
            fold_id=0,
            is_start="2020-01-01",
            is_end="2020-12-31",
            oos_start="2021-01-06",  # 5-day embargo
            oos_end="2021-03-31",
            embargo_days=5
        )
        
        # Run fold backtest
        fold_result = run_fold_backtest(spec, window, live=False, seed_offset=100)
        
        # Validate results structure
        assert "fold_id" in fold_result
        assert "in_sample" in fold_result
        assert "out_of_sample" in fold_result
        assert "is_oos_ratio" in fold_result
        assert "embargo_respected" in fold_result
        
        # Check IS results
        is_results = fold_result["in_sample"]
        assert "annual_return" in is_results
        assert "sharpe_ratio" in is_results
        assert "period_type" in is_results
        assert is_results["period_type"] == "in_sample"
        
        # Check OOS results
        oos_results = fold_result["out_of_sample"]
        assert "annual_return" in oos_results
        assert "sharpe_ratio" in oos_results
        assert "period_type" in oos_results
        assert oos_results["period_type"] == "out_of_sample"
        
        # Check ratios
        ratios = fold_result["is_oos_ratio"]
        assert "sharpe_ratio" in ratios
        assert "annual_return" in ratios
        
    finally:
        os.unlink(spec_path)


def test_walkforward_aggregation():
    """Test walk-forward results aggregation"""
    from ally.research.walkforward import aggregate_walkforward_results
    
    # Create mock fold results
    fold_results = [
        {
            "fold_id": 0,
            "in_sample": {"sharpe_ratio": 1.2, "annual_return": 0.15, "annual_volatility": 0.12},
            "out_of_sample": {"sharpe_ratio": 0.8, "annual_return": 0.10, "annual_volatility": 0.14},
            "is_oos_ratio": {"sharpe_ratio": 0.67, "annual_return": 0.67}
        },
        {
            "fold_id": 1,
            "in_sample": {"sharpe_ratio": 1.5, "annual_return": 0.18, "annual_volatility": 0.11},
            "out_of_sample": {"sharpe_ratio": 1.0, "annual_return": 0.12, "annual_volatility": 0.13},
            "is_oos_ratio": {"sharpe_ratio": 0.67, "annual_return": 0.67}
        }
    ]
    
    # Test aggregation
    agg_stats = aggregate_walkforward_results(fold_results)
    
    # Validate aggregation structure
    assert "n_folds_total" in agg_stats
    assert "n_folds_successful" in agg_stats
    assert "in_sample_stats" in agg_stats
    assert "out_of_sample_stats" in agg_stats
    assert "degradation_stats" in agg_stats
    assert "median_oos_sharpe" in agg_stats
    
    # Check calculations
    assert agg_stats["n_folds_total"] == 2
    assert agg_stats["n_folds_successful"] == 2
    assert agg_stats["success_rate"] == 1.0
    
    # Check OOS statistics
    oos_stats = agg_stats["out_of_sample_stats"]
    assert oos_stats["sharpe_mean"] == 0.9  # (0.8 + 1.0) / 2
    assert oos_stats["sharpe_median"] == 0.9
    
    # Test with failed folds
    fold_results_with_failure = fold_results + [{"fold_id": 2, "error": "Test error"}]
    agg_stats_failure = aggregate_walkforward_results(fold_results_with_failure)
    
    assert agg_stats_failure["n_folds_total"] == 3
    assert agg_stats_failure["n_folds_successful"] == 2
    assert agg_stats_failure["n_folds_failed"] == 1
    assert agg_stats_failure["failed_fold_ids"] == [2]


def test_walkforward_full_pipeline():
    """Test complete walk-forward analysis pipeline"""
    from ally.research.walkforward import research_walkforward_run
    
    # Create test spec
    spec_data = {
        'name': 'Full-Pipeline-WF-Test',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 30},
        'costs': {'bps_per_turnover': 12.0},
        'backtest': {'start': '2018-01-01', 'end': '2023-12-31', 'benchmark': 'SPY', 'seed': 123},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        # Run walk-forward analysis
        wf_result = research_walkforward_run(
            spec_path=spec_path,
            n_folds=3,  # Small number for test
            embargo_days=7,
            oos_sharpe_threshold=0.2,  # Low threshold for test
            live=False
        )
        
        # Validate results
        # Note: Result may be ok=False due to random performance, but structure should be valid
        assert "walkforward_receipt" in wf_result.data
        assert "spec_name" in wf_result.data
        assert "fold_results" in wf_result.data
        assert "aggregate_statistics" in wf_result.data
        assert "threshold_analysis" in wf_result.data
        
        # Check analysis parameters
        params = wf_result.data["analysis_parameters"]
        assert params["n_folds"] == 3
        assert params["embargo_days"] == 7
        assert params["oos_sharpe_threshold"] == 0.2
        
        # Check fold receipts
        fold_receipts = wf_result.data["fold_receipts"]
        assert len(fold_receipts) > 0
        assert all(len(receipt) == 16 for receipt in fold_receipts)
        
        # Check aggregate statistics
        agg_stats = wf_result.data["aggregate_statistics"]
        assert "median_oos_sharpe" in agg_stats
        assert "degradation_stats" in agg_stats
        
        # Check threshold analysis
        threshold_analysis = wf_result.data["threshold_analysis"]
        assert "threshold_met" in threshold_analysis
        assert "performance_gap" in threshold_analysis
        
    finally:
        os.unlink(spec_path)


def test_walkforward_deterministic_behavior():
    """Test that walk-forward produces deterministic results"""
    from ally.research.walkforward import research_walkforward_run
    
    spec_data = {
        'name': 'Deterministic-Test',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 25},
        'costs': {'bps_per_turnover': 10.0},
        'backtest': {'start': '2020-01-01', 'end': '2022-12-31', 'benchmark': 'SPY', 'seed': 999},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        # Run twice with same parameters
        result1 = research_walkforward_run(
            spec_path=spec_path,
            n_folds=2,
            embargo_days=5,
            oos_sharpe_threshold=0.0,
            live=False
        )
        
        result2 = research_walkforward_run(
            spec_path=spec_path,
            n_folds=2,
            embargo_days=5,
            oos_sharpe_threshold=0.0,
            live=False
        )
        
        # Results should be identical (deterministic)
        assert result1.ok == result2.ok
        if result1.ok and result2.ok:
            sharpe1 = result1.data["aggregate_statistics"]["median_oos_sharpe"]
            sharpe2 = result2.data["aggregate_statistics"]["median_oos_sharpe"]
            assert abs(sharpe1 - sharpe2) < 1e-6, "Results should be deterministic"
        
    finally:
        os.unlink(spec_path)


def test_walkforward_replication_integration():
    """Test walk-forward integration with replication pipeline"""
    from ally.research.replication import research_replication_run
    
    spec_data = {
        'name': 'Replication-WF-Integration',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 40},
        'costs': {'bps_per_turnover': 18.0},
        'backtest': {'start': '2019-01-01', 'end': '2022-12-31', 'benchmark': 'SPY', 'seed': 456},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        # Test regular replication
        regular_result = research_replication_run(
            spec_path=spec_path,
            live=False,
            walkforward=False
        )
        
        assert regular_result.ok == True
        assert "backtest_results" in regular_result.data
        
        # Test walk-forward via replication
        wf_result = research_replication_run(
            spec_path=spec_path,
            live=False,
            walkforward=True,
            n_folds=2,
            embargo_days=5
        )
        
        # Should delegate to walk-forward
        assert "walkforward_receipt" in wf_result.data
        assert "fold_results" in wf_result.data
        
    finally:
        os.unlink(spec_path)


def test_walkforward_error_handling():
    """Test walk-forward error handling"""
    from ally.research.walkforward import research_walkforward_run
    
    # Test with non-existent spec file
    result = research_walkforward_run(
        spec_path="/nonexistent/spec.yaml",
        live=False
    )
    
    assert result.ok == False
    assert len(result.errors) > 0
    assert "receipt_hash" in result.__dict__  # Should still generate error receipt


def test_walkforward_receipts():
    """Test that walk-forward generates proper receipts"""
    from ally.research.walkforward import research_walkforward_run
    
    spec_data = {
        'name': 'Receipt-Test',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 35},
        'costs': {'bps_per_turnover': 14.0},
        'backtest': {'start': '2020-01-01', 'end': '2022-06-30', 'benchmark': 'SPY', 'seed': 789},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        result = research_walkforward_run(
            spec_path=spec_path,
            n_folds=2,
            live=False
        )
        
        # Check main receipt
        assert "walkforward_receipt" in result.data
        assert len(result.data["walkforward_receipt"]) == 16
        
        # Check fold receipts
        fold_receipts = result.data["fold_receipts"]
        assert len(fold_receipts) >= 1
        assert all(len(receipt) == 16 for receipt in fold_receipts)
        
        # Check receipt hash on result
        assert hasattr(result, 'receipt_hash')
        assert len(result.receipt_hash) == 40  # Full SHA-1
        
    finally:
        os.unlink(spec_path)


def test_walkforward_json_report():
    """Test that walk-forward generates stable JSON reports"""
    from ally.research.walkforward import research_walkforward_run
    import json
    
    spec_data = {
        'name': 'JSON-Report-Test',
        'universe': {'asset_class': 'equities_us', 'inclusion': {'market_cap_min': 1e9}},
        'data': {'ohlcv': {'frequency': 'daily'}},
        'signal': {'type': 'cross_sectional', 'formula': 'momentum', 'rebalance': 'monthly'},
        'portfolio': {'scheme': 'equal_weight_top_k', 'k': 30},
        'costs': {'bps_per_turnover': 12.0},
        'backtest': {'start': '2020-01-01', 'end': '2022-06-30', 'benchmark': 'SPY', 'seed': 555},
        'gates': {}, 'proof': {'emit': True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(spec_data, f)
        spec_path = f.name
    
    try:
        result = research_walkforward_run(
            spec_path=spec_path,
            n_folds=2,
            live=False
        )
        
        # Check JSON report was created
        assert "json_report_path" in result.data
        assert "report_receipt" in result.data
        
        json_path = result.data["json_report_path"]
        assert os.path.exists(json_path)
        
        # Load and validate JSON structure
        with open(json_path, 'r') as f:
            report = json.load(f)
        
        # Check required fields
        assert "spec_name" in report
        assert "oos_metrics" in report
        assert "per_fold_table" in report
        assert "thresholds" in report
        assert "parameters" in report
        
        # Check OOS metrics
        oos_metrics = report["oos_metrics"]
        assert "median" in oos_metrics
        assert "mean" in oos_metrics
        assert "q25" in oos_metrics
        assert "q75" in oos_metrics
        
        # Check per-fold table
        for fold in report["per_fold_table"]:
            assert "fold" in fold
            assert "is_sharpe" in fold
            assert "oos_sharpe" in fold
            assert "degradation" in fold
        
        # Clean up
        os.unlink(json_path)
        
    finally:
        os.unlink(spec_path)


def test_walkforward_timezone_handling():
    """Test timezone handling (UTC coercion)"""
    from ally.research.walkforward import generate_walkforward_windows
    import pandas as pd
    
    # Test with UTC (should be idempotent)
    windows_utc = generate_walkforward_windows(
        start_date="2020-01-01T00:00:00Z",
        end_date="2023-12-31T00:00:00Z",
        n_folds=2,
        embargo_days=5
    )
    
    # Test with non-UTC (should be coerced)
    windows_naive = generate_walkforward_windows(
        start_date="2020-01-01",
        end_date="2023-12-31", 
        n_folds=2,
        embargo_days=5
    )
    
    # Results should be equivalent (both UTC)
    assert len(windows_utc) == len(windows_naive)
    for w1, w2 in zip(windows_utc, windows_naive):
        assert w1.is_start == w2.is_start
        assert w1.oos_end == w2.oos_end


def test_walkforward_non_divisible_windows():
    """Test handling of non-divisible time periods"""
    from ally.research.walkforward import generate_walkforward_windows
    
    # Request more folds than the period can cleanly accommodate
    windows = generate_walkforward_windows(
        start_date="2020-01-01",
        end_date="2021-06-30",  # 18 months
        n_folds=8,  # More folds than ideal
        embargo_days=5,
        min_is_days=100,  # Lower minimum for test
        min_oos_days=30
    )
    
    # Should generate fewer folds and record what was created
    assert len(windows) < 8, "Should clip when windows don't fit"
    assert len(windows) >= 1, "Should generate at least one window"
    
    # Validate last fold is properly clipped but recorded
    if len(windows) > 1:
        last_window = windows[-1]
        assert last_window.oos_end <= "2021-06-30"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])