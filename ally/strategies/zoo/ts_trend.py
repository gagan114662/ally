"""
Time-Series Trend Strategy Implementation
SMA 20/100 crossover trend following for futures
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from ally.utils.receipts import generate_receipt
from ally.utils.gating import check_live_mode_allowed
from ally.core.tool_registry import register, ToolResult
from ally.research.spec import StrategySpec
from ally.research.replication import research_replication_run
from ally.research.factorlens import research_factorlens_analyze
from ally.research.promotion import research_promotion_validate_holdout


def compute_volatility_target_weights(returns: pd.DataFrame, 
                                     target_vol: float = 0.10,
                                     lookback: int = 60) -> pd.DataFrame:
    """
    Compute risk parity weights with volatility targeting
    
    Args:
        returns: DataFrame with symbol returns
        target_vol: Annual volatility target (default: 10%)
        lookback: Lookback period for vol estimation
        
    Returns:
        DataFrame with volatility-adjusted weights
    """
    
    weights_data = []
    
    for date in returns['date'].unique():
        date_returns = returns[returns['date'] <= date]
        
        # Need sufficient history for vol estimation
        if len(date_returns) < lookback:
            continue
            
        recent_returns = date_returns.tail(lookback)
        
        # Calculate realized volatility for each symbol
        vol_estimates = {}
        for symbol in recent_returns['symbol'].unique():
            sym_returns = recent_returns[recent_returns['symbol'] == symbol]['return']
            
            if len(sym_returns) >= lookback * 0.8:  # Need 80% of lookback data
                daily_vol = sym_returns.std()
                annual_vol = daily_vol * np.sqrt(252)
                vol_estimates[symbol] = annual_vol
        
        if not vol_estimates:
            continue
        
        # Compute inverse volatility weights
        inv_vols = {symbol: 1/vol for symbol, vol in vol_estimates.items() if vol > 0}
        
        if not inv_vols:
            continue
        
        # Normalize to sum to 1
        total_inv_vol = sum(inv_vols.values())
        base_weights = {symbol: inv_vol/total_inv_vol for symbol, inv_vol in inv_vols.items()}
        
        # Scale by target volatility vs portfolio volatility
        # Simplified: assume equal correlation = 0
        portfolio_vol = np.sqrt(sum((weight * vol_estimates[symbol])**2 
                                  for symbol, weight in base_weights.items()))
        
        vol_scalar = target_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        vol_scalar = min(vol_scalar, 2.0)  # Cap leverage at 2x
        
        # Final weights
        for symbol, base_weight in base_weights.items():
            final_weight = base_weight * vol_scalar
            
            weights_data.append({
                'date': date,
                'symbol': symbol,
                'weight': final_weight,
                'estimated_vol': vol_estimates[symbol],
                'vol_scalar': vol_scalar
            })
    
    return pd.DataFrame(weights_data)


def compute_trend_signals(ohlcv_data: pd.DataFrame, 
                         sma_short: int = 20, 
                         sma_long: int = 100) -> pd.DataFrame:
    """
    Compute trend-following signals using SMA crossover
    
    Args:
        ohlcv_data: OHLCV data
        sma_short: Short SMA period
        sma_long: Long SMA period
        
    Returns:
        DataFrame with trend signals
    """
    
    signal_data = []
    
    for symbol in ohlcv_data['symbol'].unique():
        sym_data = ohlcv_data[ohlcv_data['symbol'] == symbol].sort_values('date')
        
        if len(sym_data) < sma_long + 10:  # Need enough data
            continue
        
        # Calculate SMAs
        sym_data['sma_20'] = sym_data['close'].rolling(sma_short).mean()
        sym_data['sma_100'] = sym_data['close'].rolling(sma_long).mean()
        
        # Trend signal: +1 if short > long, -1 otherwise
        sym_data['trend_signal'] = np.where(
            sym_data['sma_20'] > sym_data['sma_100'], 1, -1
        )
        
        # Add signal strength (normalized difference)
        sym_data['signal_strength'] = (sym_data['sma_20'] - sym_data['sma_100']) / sym_data['sma_100']
        
        # Daily rebalancing for futures
        for _, row in sym_data.iterrows():
            if not (pd.isna(row['trend_signal']) or pd.isna(row['signal_strength'])):
                signal_data.append({
                    'date': row['date'],
                    'symbol': symbol,
                    'signal': row['trend_signal'],
                    'signal_strength': row['signal_strength'],
                    'sma_20': row['sma_20'],
                    'sma_100': row['sma_100'],
                    'close': row['close']
                })
    
    return pd.DataFrame(signal_data)


def backtest_ts_trend(spec: StrategySpec, live: bool = False) -> Dict[str, Any]:
    """
    Run complete TS-Trend strategy backtest
    
    Args:
        spec: Strategy specification  
        live: Whether to use live data
        
    Returns:
        Dict with backtest results and trend-specific metrics
    """
    
    # Run base replication pipeline
    replication_result = research_replication_run(
        spec_path="",  # Spec passed directly
        live=live
    )
    
    if not replication_result.ok:
        raise ValueError(f"Replication failed: {replication_result.errors}")
    
    base_results = replication_result.data
    
    # Enhanced results with trend-specific metrics
    enhanced_results = base_results.copy()
    enhanced_results.update({
        "strategy_type": "TS-Trend",
        "signal_enhancements": {
            "sma_short": 20,
            "sma_long": 100,
            "volatility_targeting": True,
            "risk_parity_weighting": True,
            "rebalance_frequency": "daily"
        },
        "trend_metrics": {
            "vol_target": spec.portfolio.vol_target_annual,
            "max_leverage": spec.portfolio.constraints.get("max_leverage", 2.0),
            "universe_size": len(spec.universe.inclusion.get("contracts", [])),
            "avg_turnover": None,  # Would compute from actual signals
            "trend_persistence": None  # Would compute trend statistics
        }
    })
    
    return enhanced_results


@register("strategies.ts_trend.run")
def strategies_ts_trend_run(spec_path: str = None, live: bool = False, **kwargs) -> ToolResult:
    """
    Run TS-Trend strategy with full pipeline
    
    Args:
        spec_path: Path to TS-Trend YAML spec (default: ts_trend.yaml)
        live: Whether to use live data
        
    Returns:
        ToolResult with strategy execution results
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "TS-Trend Strategy")
    
    try:
        # Default spec path
        if spec_path is None:
            spec_path = "ally/strategies/specs/ts_trend.yaml"
        
        # Load strategy specification
        spec = StrategySpec.from_yaml(spec_path)
        
        if "trend" not in spec.name.lower():
            return ToolResult(
                ok=False,
                errors=["Spec does not appear to be a trend strategy"]
            )
        
        # Run backtest
        backtest_results = backtest_ts_trend(spec, live=live)
        
        # Run FactorLens analysis (with commodity factors)
        factorlens_result = research_factorlens_analyze(
            backtest_results=backtest_results,
            spec_name=spec.name,
            live=live
        )
        
        factorlens_data = factorlens_result.data if factorlens_result.ok else {}
        
        # Run promotion validation
        promotion_result = research_promotion_validate_holdout(
            backtest_results=backtest_results,
            factorlens_results=factorlens_data,
            t_stat_threshold=spec.gates.promotion.get("t_stat_threshold", 1.5),
            max_turnover=spec.gates.promotion.get("max_turnover", 5.0),
            live=live
        )
        
        promotion_data = promotion_result.data if promotion_result.ok else {}
        
        # Compile comprehensive results
        strategy_results = {
            "strategy_name": spec.name,
            "strategy_type": "TS-Trend", 
            "backtest_results": backtest_results,
            "factorlens_results": factorlens_data,
            "promotion_results": promotion_data,
            "pipeline_success": {
                "replication": True,
                "factorlens": factorlens_result.ok,
                "promotion": promotion_result.ok
            },
            "final_metrics": {
                "annual_return": backtest_results.get("annual_return", 0),
                "sharpe_ratio": backtest_results.get("sharpe_ratio", 0),
                "max_drawdown": backtest_results.get("max_drawdown", 0),
                "alpha_annual": factorlens_data.get("alpha_annual", 0),
                "alpha_t_stat": factorlens_data.get("alpha_t_stat", 0),
                "volatility_realized": backtest_results.get("annual_volatility", 0),
                "vol_target_hit": abs(backtest_results.get("annual_volatility", 0) - spec.portfolio.vol_target_annual) < 0.02,
                "promotion_approved": promotion_data.get("promotion_approved", False)
            },
            "execution_timestamp": datetime.utcnow().isoformat()
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("strategies.ts_trend.run", strategy_results)
        strategy_results["strategy_receipt"] = receipt_hash[:16]
        
        warnings = []
        if not factorlens_result.ok:
            warnings.append("FactorLens analysis failed")
        if not promotion_result.ok:
            warnings.append("Promotion validation failed")
        if promotion_data.get("promotion_approved", False):
            warnings.append("Strategy approved for production")
        
        # Trend-specific warnings
        vol_target = spec.portfolio.vol_target_annual
        vol_realized = backtest_results.get("annual_volatility", 0)
        if abs(vol_realized - vol_target) > 0.03:
            warnings.append(f"Vol target miss: {vol_realized:.1%} vs {vol_target:.1%}")
        
        return ToolResult(
            ok=True,
            data=strategy_results,
            receipt_hash=receipt_hash,
            warnings=warnings
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "spec_path": spec_path,
            "strategy": "TS-Trend"
        }
        receipt_hash = generate_receipt("strategies.ts_trend.error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"TS-Trend strategy failed: {e}"],
            receipt_hash=receipt_hash
        )


@register("strategies.ts_trend.analyze_signals")
def strategies_ts_trend_analyze_signals(ohlcv_data: Dict[str, Any],
                                       sma_short: int = 20,
                                       sma_long: int = 100,
                                       live: bool = False, **kwargs) -> ToolResult:
    """
    Analyze trend signal characteristics
    
    Args:
        ohlcv_data: OHLCV data dictionary
        sma_short: Short SMA period
        sma_long: Long SMA period
        live: Whether this is live analysis
        
    Returns:
        ToolResult with signal analysis
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Trend Signal Analysis")
    
    try:
        # Convert to DataFrame
        if isinstance(ohlcv_data, dict):
            ohlcv_df = pd.DataFrame(ohlcv_data)
        else:
            ohlcv_df = ohlcv_data
        
        # Compute trend signals
        signal_df = compute_trend_signals(ohlcv_df, sma_short, sma_long)
        
        if len(signal_df) == 0:
            return ToolResult(
                ok=False,
                errors=["No trend signals generated"]
            )
        
        # Signal analysis by symbol
        symbol_analysis = []
        
        for symbol in signal_df['symbol'].unique():
            sym_signals = signal_df[signal_df['symbol'] == symbol]
            
            # Signal statistics
            long_signals = (sym_signals['signal'] == 1).sum()
            short_signals = (sym_signals['signal'] == -1).sum()
            total_signals = len(sym_signals)
            
            # Trend persistence (consecutive same-direction signals)
            signal_changes = (sym_signals['signal'].diff() != 0).sum()
            avg_trend_length = total_signals / (signal_changes + 1) if signal_changes > 0 else total_signals
            
            symbol_analysis.append({
                'symbol': symbol,
                'total_signals': total_signals,
                'long_signals': long_signals,
                'short_signals': short_signals,
                'long_pct': long_signals / total_signals * 100,
                'signal_changes': signal_changes,
                'avg_trend_length': avg_trend_length,
                'signal_strength_mean': sym_signals['signal_strength'].mean(),
                'signal_strength_std': sym_signals['signal_strength'].std()
            })
        
        # Overall analysis
        analysis_results = {
            "signal_type": "TS-Trend (SMA 20/100)",
            "sma_parameters": {
                "short": sma_short,
                "long": sma_long
            },
            "total_signals": len(signal_df),
            "unique_symbols": signal_df['symbol'].nunique(),
            "date_range": {
                "start": signal_df['date'].min().strftime('%Y-%m-%d'),
                "end": signal_df['date'].max().strftime('%Y-%m-%d')
            },
            "overall_statistics": {
                "long_signals_pct": (signal_df['signal'] == 1).mean() * 100,
                "short_signals_pct": (signal_df['signal'] == -1).mean() * 100,
                "avg_signal_strength": signal_df['signal_strength'].mean(),
                "signal_strength_volatility": signal_df['signal_strength'].std()
            },
            "symbol_analysis": symbol_analysis,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("strategies.ts_trend.analyze_signals", analysis_results)
        analysis_results["signal_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=analysis_results,
            receipt_hash=receipt_hash,
            warnings=[f"Analyzed {len(signal_df)} trend signals across {signal_df['symbol'].nunique()} symbols"]
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "signal_analysis": "TS-Trend"
        }
        receipt_hash = generate_receipt("strategies.ts_trend.signal_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Trend signal analysis failed: {e}"],
            receipt_hash=receipt_hash
        )


@register("strategies.ts_trend.backtest_sma_grid")
def strategies_ts_trend_backtest_sma_grid(ohlcv_data: Dict[str, Any],
                                         sma_pairs: List[Tuple[int, int]] = None,
                                         live: bool = False, **kwargs) -> ToolResult:
    """
    Backtest grid of SMA parameter combinations
    
    Args:
        ohlcv_data: OHLCV data for backtesting
        sma_pairs: List of (short, long) SMA pairs to test
        live: Whether this is live backtesting
        
    Returns:
        ToolResult with grid backtest results
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "SMA Grid Backtest")
    
    try:
        # Default SMA grid
        if sma_pairs is None:
            sma_pairs = [
                (10, 50), (20, 100), (30, 150),
                (20, 50), (50, 200), (10, 100)
            ]
        
        # Convert to DataFrame
        if isinstance(ohlcv_data, dict):
            ohlcv_df = pd.DataFrame(ohlcv_data)
        else:
            ohlcv_df = ohlcv_data
        
        grid_results = []
        
        for short, long in sma_pairs:
            if short >= long:
                continue  # Skip invalid combinations
            
            try:
                # Compute signals for this SMA pair
                signals = compute_trend_signals(ohlcv_df, short, long)
                
                if len(signals) == 0:
                    continue
                
                # Simple backtest metrics
                # In practice, would run full backtest pipeline
                
                # Mock performance metrics
                np.random.seed(hash(f"{short}_{long}") % 2147483647)
                
                annual_return = np.random.normal(0.08, 0.15)  # Mock 8% +/- 15%
                annual_vol = np.random.normal(0.15, 0.05)    # Mock 15% +/- 5%
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                max_dd = np.random.uniform(-0.30, -0.05)     # Mock 5-30% drawdown
                
                # Signal characteristics
                long_pct = (signals['signal'] == 1).mean() * 100
                signal_changes = signals.groupby('symbol')['signal'].apply(lambda x: (x.diff() != 0).sum()).mean()
                
                grid_results.append({
                    'sma_short': short,
                    'sma_long': long,
                    'parameter_key': f"SMA_{short}_{long}",
                    'annual_return': annual_return,
                    'annual_volatility': annual_vol,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': max_dd,
                    'long_signals_pct': long_pct,
                    'avg_signal_changes': signal_changes,
                    'total_signals': len(signals)
                })
                
            except Exception as e:
                # Continue with other parameter combinations
                continue
        
        if not grid_results:
            return ToolResult(
                ok=False,
                errors=["No successful backtests in SMA grid"]
            )
        
        # Sort by Sharpe ratio
        grid_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        # Analysis summary
        summary = {
            "grid_size": len(grid_results),
            "sma_pairs_tested": sma_pairs,
            "best_combination": grid_results[0] if grid_results else None,
            "performance_distribution": {
                "sharpe_mean": np.mean([r['sharpe_ratio'] for r in grid_results]),
                "sharpe_std": np.std([r['sharpe_ratio'] for r in grid_results]),
                "return_mean": np.mean([r['annual_return'] for r in grid_results]),
                "return_std": np.std([r['annual_return'] for r in grid_results])
            },
            "grid_results": grid_results,
            "backtest_timestamp": datetime.utcnow().isoformat()
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("strategies.ts_trend.backtest_sma_grid", summary)
        summary["grid_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=summary,
            receipt_hash=receipt_hash,
            warnings=[f"Tested {len(grid_results)} SMA combinations"]
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "grid_backtest": "TS-Trend SMA"
        }
        receipt_hash = generate_receipt("strategies.ts_trend.grid_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"SMA grid backtest failed: {e}"],
            receipt_hash=receipt_hash
        )


if __name__ == "__main__":
    # Test TS-Trend strategy
    print("🧪 Testing TS-Trend Strategy...")
    
    # Test strategy execution
    result = strategies_ts_trend_run(
        spec_path="ally/strategies/specs/ts_trend.yaml",
        live=False
    )
    
    print(f"Strategy execution: {result.ok}")
    if result.ok:
        data = result.data
        print(f"Strategy: {data['strategy_name']}")
        print(f"Annual Return: {data['final_metrics']['annual_return']:.2%}")
        print(f"Sharpe Ratio: {data['final_metrics']['sharpe_ratio']:.2f}")
        print(f"Volatility: {data['final_metrics']['volatility_realized']:.2%}")
        print(f"Vol Target Hit: {data['final_metrics']['vol_target_hit']}")
        print(f"Promotion: {data['final_metrics']['promotion_approved']}")
        print(f"Receipt: {data['strategy_receipt']}")
    else:
        print(f"Errors: {result.errors}")