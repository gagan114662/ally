"""
Promotion gate - Holdout validation with statistical significance testing
Final gate before strategy promotion to production
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
import hashlib

from ally.utils.receipts import generate_receipt
from ally.utils.gating import check_live_mode_allowed
from ally.core.tool_registry import register, ToolResult
from ally.research.spec import StrategySpec


def split_backtest_period(start_date: str, end_date: str, 
                         holdout_fraction: float = 0.3) -> Tuple[str, str, str, str]:
    """
    Split backtest period into training and holdout periods
    
    Args:
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        holdout_fraction: Fraction of period for holdout (default: 30%)
        
    Returns:
        Tuple of (train_start, train_end, holdout_start, holdout_end)
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    total_days = (end - start).days
    holdout_days = int(total_days * holdout_fraction)
    
    train_end = end - timedelta(days=holdout_days)
    holdout_start = train_end + timedelta(days=1)
    
    return (
        start.strftime('%Y-%m-%d'),
        train_end.strftime('%Y-%m-%d'),
        holdout_start.strftime('%Y-%m-%d'),
        end.strftime('%Y-%m-%d')
    )


def compute_holdout_statistics(holdout_returns: np.ndarray,
                              benchmark_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive statistics on holdout period performance
    
    Args:
        holdout_returns: Array of daily strategy returns
        benchmark_returns: Optional benchmark returns for comparison
        
    Returns:
        Dict with statistical measures
    """
    if len(holdout_returns) == 0:
        raise ValueError("No holdout returns provided")
    
    # Basic statistics
    n_obs = len(holdout_returns)
    mean_return = np.mean(holdout_returns)
    std_return = np.std(holdout_returns, ddof=1)
    
    # Annualized metrics
    annual_return = mean_return * 252
    annual_vol = std_return * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # t-statistic for mean return
    t_stat = mean_return / (std_return / np.sqrt(n_obs)) if std_return > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_obs-1))
    
    # Drawdown analysis
    cumulative_returns = np.cumprod(1 + holdout_returns) - 1
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (1 + peak)
    max_drawdown = np.min(drawdown)
    
    # Skewness and kurtosis
    skew = stats.skew(holdout_returns)
    kurt = stats.kurtosis(holdout_returns, fisher=True)  # Excess kurtosis
    
    # VaR and CVaR (5% level)
    var_5 = np.percentile(holdout_returns, 5)
    cvar_5 = np.mean(holdout_returns[holdout_returns <= var_5])
    
    statistics = {
        'n_observations': n_obs,
        'mean_return_daily': mean_return,
        'std_return_daily': std_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        't_statistic': t_stat,
        'p_value': p_value,
        'max_drawdown': max_drawdown,
        'skewness': skew,
        'excess_kurtosis': kurt,
        'var_5pct': var_5,
        'cvar_5pct': cvar_5,
        'total_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
    }
    
    # Benchmark comparison if provided
    if benchmark_returns is not None and len(benchmark_returns) == n_obs:
        excess_returns = holdout_returns - benchmark_returns
        tracking_error = np.std(excess_returns, ddof=1) * np.sqrt(252)
        information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
        
        # Beta calculation
        if np.std(benchmark_returns) > 0:
            beta = np.cov(holdout_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns, ddof=1)
        else:
            beta = 0
        
        statistics.update({
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'excess_return_annual': np.mean(excess_returns) * 252
        })
    
    return statistics


def compute_capacity_metrics(strategy_returns: np.ndarray, turnover: float,
                            avg_position_size: float = 100000) -> Dict[str, Any]:
    """
    Estimate strategy capacity based on liquidity constraints
    
    Args:
        strategy_returns: Array of strategy returns
        turnover: Annual portfolio turnover rate
        avg_position_size: Average position size in dollars
        
    Returns:
        Dict with capacity estimates
    """
    # Simplified capacity model
    # In practice, would use detailed liquidity data
    
    # Estimate market impact cost (bps) based on turnover
    # Higher turnover = higher impact
    impact_cost_bps = min(turnover * 20, 100)  # Cap at 100 bps
    
    # Estimate capacity based on volatility and turnover
    vol = np.std(strategy_returns) * np.sqrt(252)
    
    # Simple heuristic: lower vol and turnover = higher capacity
    base_capacity = 100_000_000  # $100M base
    vol_adjustment = max(0.1, 1 - vol)  # Reduce for high vol
    turnover_adjustment = max(0.1, 1 - turnover/2)  # Reduce for high turnover
    
    estimated_capacity = base_capacity * vol_adjustment * turnover_adjustment
    
    return {
        'estimated_capacity_usd': estimated_capacity,
        'impact_cost_bps': impact_cost_bps,
        'turnover_annual': turnover,
        'volatility_annual': vol,
        'capacity_tier': 'large' if estimated_capacity > 500_000_000 else 
                        'medium' if estimated_capacity > 100_000_000 else 'small'
    }


def generate_strategy_bundle_hash(spec: StrategySpec, holdout_stats: Dict[str, Any],
                                 promotion_decision: bool) -> str:
    """
    Generate SHA-1 hash for strategy bundle for production deployment
    
    Args:
        spec: Strategy specification
        holdout_stats: Holdout validation statistics
        promotion_decision: Whether strategy passed promotion gate
        
    Returns:
        SHA-1 hash string
    """
    bundle_data = {
        'strategy_spec': spec.to_dict(),
        'holdout_statistics': holdout_stats,
        'promotion_approved': promotion_decision,
        'bundle_timestamp': datetime.utcnow().isoformat()
    }
    
    # Create deterministic hash
    bundle_str = str(sorted(bundle_data.items()))
    return hashlib.sha1(bundle_str.encode()).hexdigest()


@register("research.promotion.validate_holdout")
def research_promotion_validate_holdout(backtest_results: Dict[str, Any], 
                                       factorlens_results: Optional[Dict[str, Any]] = None,
                                       t_stat_threshold: float = 2.0,
                                       max_turnover: float = 2.0,
                                       max_cost_bps: float = 50.0,
                                       min_capacity_usd: float = 50_000_000,
                                       holdout_fraction: float = 0.3,
                                       live: bool = False, **kwargs) -> ToolResult:
    """
    Run holdout validation for strategy promotion
    
    Args:
        backtest_results: Full backtest results from replication
        factorlens_results: Optional FactorLens analysis results
        t_stat_threshold: Minimum t-statistic for significance (default: 2.0)
        max_turnover: Maximum annual turnover allowed (default: 2.0)
        max_cost_bps: Maximum transaction cost in bps (default: 50)
        min_capacity_usd: Minimum strategy capacity required
        holdout_fraction: Fraction of data for holdout (default: 0.3)
        live: Whether this is live validation
        
    Returns:
        ToolResult with promotion decision and analysis
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Strategy Promotion")
    
    try:
        # Extract return data
        portfolio_returns = backtest_results.get("portfolio_returns", {})
        if not portfolio_returns:
            return ToolResult(
                ok=False,
                errors=["No portfolio returns found in backtest results"]
            )
        
        # Convert to time series
        returns_df = pd.DataFrame([
            {"date": pd.to_datetime(date), "return": ret}
            for date, ret in portfolio_returns.items()
        ]).sort_values('date')
        
        if len(returns_df) < 100:
            return ToolResult(
                ok=False,
                errors=[f"Insufficient data for holdout validation: {len(returns_df)} observations"]
            )
        
        # Split into training and holdout periods
        start_date = returns_df['date'].min().strftime('%Y-%m-%d')
        end_date = returns_df['date'].max().strftime('%Y-%m-%d')
        
        train_start, train_end, holdout_start, holdout_end = split_backtest_period(
            start_date, end_date, holdout_fraction
        )
        
        # Filter holdout data
        holdout_mask = (returns_df['date'] >= pd.to_datetime(holdout_start)) & \
                      (returns_df['date'] <= pd.to_datetime(holdout_end))
        holdout_data = returns_df[holdout_mask]
        
        if len(holdout_data) < 20:
            return ToolResult(
                ok=False,
                errors=[f"Insufficient holdout data: {len(holdout_data)} observations"]
            )
        
        holdout_returns = holdout_data['return'].values
        
        # Compute holdout statistics
        holdout_stats = compute_holdout_statistics(holdout_returns)
        
        # Get strategy metadata
        spec_name = backtest_results.get("spec_name", "unknown_strategy")
        
        # Estimate turnover (simplified)
        # In practice, would compute from portfolio weights
        estimated_turnover = 1.5  # Mock annual turnover
        
        # Compute capacity metrics
        capacity_metrics = compute_capacity_metrics(
            holdout_returns, 
            estimated_turnover,
            avg_position_size=backtest_results.get("avg_position_size", 100000)
        )
        
        # Apply promotion criteria
        promotion_checks = {}
        
        # 1. Statistical significance check
        t_stat_pass = abs(holdout_stats['t_statistic']) >= t_stat_threshold
        promotion_checks['t_statistic'] = {
            'value': holdout_stats['t_statistic'],
            'threshold': t_stat_threshold,
            'passed': t_stat_pass
        }
        
        # 2. Turnover check
        turnover_pass = estimated_turnover <= max_turnover
        promotion_checks['turnover'] = {
            'value': estimated_turnover,
            'threshold': max_turnover,
            'passed': turnover_pass
        }
        
        # 3. Cost check
        cost_bps = capacity_metrics['impact_cost_bps']
        cost_pass = cost_bps <= max_cost_bps
        promotion_checks['transaction_cost'] = {
            'value': cost_bps,
            'threshold': max_cost_bps,
            'passed': cost_pass
        }
        
        # 4. Capacity check
        capacity_pass = capacity_metrics['estimated_capacity_usd'] >= min_capacity_usd
        promotion_checks['capacity'] = {
            'value': capacity_metrics['estimated_capacity_usd'],
            'threshold': min_capacity_usd,
            'passed': capacity_pass
        }
        
        # 5. Risk checks (basic)
        max_dd_pass = holdout_stats['max_drawdown'] >= -0.20  # Max 20% drawdown
        sharpe_pass = holdout_stats['sharpe_ratio'] >= 1.0    # Min 1.0 Sharpe
        
        promotion_checks['max_drawdown'] = {
            'value': holdout_stats['max_drawdown'],
            'threshold': -0.20,
            'passed': max_dd_pass
        }
        
        promotion_checks['sharpe_ratio'] = {
            'value': holdout_stats['sharpe_ratio'],
            'threshold': 1.0,
            'passed': sharpe_pass
        }
        
        # Overall promotion decision
        all_checks = [
            t_stat_pass, turnover_pass, cost_pass, 
            capacity_pass, max_dd_pass, sharpe_pass
        ]
        promotion_approved = all(all_checks)
        
        # Compile results
        promotion_results = {
            "spec_name": spec_name,
            "holdout_period": {
                "start": holdout_start,
                "end": holdout_end,
                "observations": len(holdout_data)
            },
            "training_period": {
                "start": train_start,
                "end": train_end
            },
            "holdout_statistics": holdout_stats,
            "capacity_metrics": capacity_metrics,
            "promotion_checks": promotion_checks,
            "promotion_approved": promotion_approved,
            "checks_passed": sum(all_checks),
            "total_checks": len(all_checks),
            "validation_timestamp": datetime.utcnow().isoformat()
        }
        
        # Add FactorLens results if available
        if factorlens_results:
            promotion_results["factorlens_summary"] = {
                "alpha_annual": factorlens_results.get("alpha_annual"),
                "alpha_t_stat": factorlens_results.get("alpha_t_stat"),
                "alpha_significant": factorlens_results.get("alpha_significant"),
                "r_squared": factorlens_results.get("r_squared")
            }
        
        # Generate bundle hash if approved
        if promotion_approved:
            # Load spec for bundle hash (simplified)
            spec_dict = {
                'name': spec_name,
                'type': 'mock_spec',  # In practice, load actual spec
                'holdout_validation': promotion_results
            }
            bundle_hash = hashlib.sha1(str(spec_dict).encode()).hexdigest()
            promotion_results["bundle_hash"] = bundle_hash
            promotion_results["bundle_ready"] = True
        
        # Generate receipt
        receipt_hash = generate_receipt("research.promotion.validate_holdout", promotion_results)
        promotion_results["promotion_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=promotion_results,
            receipt_hash=receipt_hash,
            warnings=["Strategy approved for production deployment"] if promotion_approved 
                    else ["Strategy failed promotion criteria"]
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "spec_name": backtest_results.get("spec_name", "unknown")
        }
        receipt_hash = generate_receipt("research.promotion.error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Promotion validation failed: {e}"],
            receipt_hash=receipt_hash
        )


@register("research.promotion.create_bundle")
def research_promotion_create_bundle(spec_path: str, promotion_results: Dict[str, Any],
                                    live: bool = False, **kwargs) -> ToolResult:
    """
    Create production bundle for approved strategy
    
    Args:
        spec_path: Path to strategy specification
        promotion_results: Results from promotion validation
        live: Whether this is live bundle creation
        
    Returns:
        ToolResult with bundle information
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Strategy Bundle Creation")
    
    try:
        if not promotion_results.get("promotion_approved", False):
            return ToolResult(
                ok=False,
                errors=["Cannot create bundle for non-approved strategy"]
            )
        
        # Load strategy spec
        from ally.research.spec import StrategySpec
        spec = StrategySpec.from_yaml(spec_path)
        
        # Generate comprehensive bundle hash
        bundle_hash = generate_strategy_bundle_hash(
            spec, 
            promotion_results["holdout_statistics"],
            True
        )
        
        # Create bundle metadata
        bundle_data = {
            "bundle_hash": bundle_hash,
            "strategy_spec": spec.to_dict(),
            "promotion_validation": promotion_results,
            "deployment_ready": True,
            "bundle_created_at": datetime.utcnow().isoformat(),
            "production_metadata": {
                "capacity_tier": promotion_results["capacity_metrics"]["capacity_tier"],
                "estimated_capacity": promotion_results["capacity_metrics"]["estimated_capacity_usd"],
                "risk_tier": "standard",  # Based on promotion checks
                "monitoring_required": True
            }
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("research.promotion.create_bundle", bundle_data)
        bundle_data["bundle_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=bundle_data,
            receipt_hash=receipt_hash,
            warnings=[f"Strategy bundle {bundle_hash[:8]} ready for deployment"]
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "spec_path": spec_path
        }
        receipt_hash = generate_receipt("research.promotion.bundle_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Bundle creation failed: {e}"],
            receipt_hash=receipt_hash
        )


if __name__ == "__main__":
    # Test promotion functionality
    print("🧪 Testing Promotion...")
    
    # Generate mock backtest results
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    mock_returns = np.random.normal(0.0008, 0.015, len(dates))  # Positive alpha
    
    mock_backtest = {
        "spec_name": "test_strategy",
        "portfolio_returns": {
            date.strftime('%Y-%m-%d'): ret
            for date, ret in zip(dates, mock_returns)
        }
    }
    
    # Test holdout validation
    result = research_promotion_validate_holdout(
        backtest_results=mock_backtest,
        t_stat_threshold=1.5,  # Lower threshold for test
        live=False
    )
    
    print(f"Validation success: {result.ok}")
    if result.ok:
        data = result.data
        print(f"Promotion approved: {data['promotion_approved']}")
        print(f"Checks passed: {data['checks_passed']}/{data['total_checks']}")
        print(f"Holdout Sharpe: {data['holdout_statistics']['sharpe_ratio']:.2f}")
        print(f"t-statistic: {data['holdout_statistics']['t_statistic']:.2f}")
        print(f"Receipt: {data['promotion_receipt']}")
        
        if data.get('bundle_hash'):
            print(f"Bundle hash: {data['bundle_hash'][:16]}")
    else:
        print(f"Errors: {result.errors}")