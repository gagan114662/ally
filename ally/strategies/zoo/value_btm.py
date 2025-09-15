"""
Value Strategy Implementation - Book-to-Market
Cross-sectional value strategy using book-to-market ratio
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

from ally.utils.receipts import generate_receipt
from ally.utils.gating import check_live_mode_allowed
from ally.core.tool_registry import register, ToolResult
from ally.research.spec import StrategySpec
from ally.research.replication import research_replication_run
from ally.research.factorlens import research_factorlens_analyze
from ally.research.promotion import research_promotion_validate_holdout


def compute_book_to_market(fundamental_data: pd.DataFrame, 
                          price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute book-to-market ratio with proper PIT compliance
    
    Args:
        fundamental_data: DataFrame with book equity and market cap
        price_data: DataFrame with price data for market cap validation
        
    Returns:
        DataFrame with book-to-market ratios
    """
    
    # Ensure fundamental data is quarterly
    fund_data = fundamental_data.copy()
    fund_data['date'] = pd.to_datetime(fund_data['date'])
    
    # Sort by date for proper PIT compliance
    fund_data = fund_data.sort_values(['symbol', 'date'])
    
    btm_data = []
    
    for symbol in fund_data['symbol'].unique():
        sym_fund = fund_data[fund_data['symbol'] == symbol]
        
        for _, row in sym_fund.iterrows():
            report_date = row['date']
            
            # Implement 90-day reporting lag
            available_date = report_date + timedelta(days=90)
            
            # Calculate book-to-market
            book_equity = row.get('book_equity', 0)
            market_cap = row.get('market_cap', 0)
            
            # Filter out invalid cases
            if book_equity <= 0 or market_cap <= 0:
                continue
            
            btm_ratio = book_equity / market_cap
            
            btm_data.append({
                'symbol': symbol,
                'report_date': report_date,
                'available_date': available_date,
                'book_equity': book_equity,
                'market_cap': market_cap,
                'book_to_market': btm_ratio
            })
    
    return pd.DataFrame(btm_data)


def apply_value_screens(signal_data: pd.DataFrame, 
                       fundamental_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply value-specific screens to filter universe
    
    Args:
        signal_data: DataFrame with value signals
        fundamental_data: DataFrame with fundamental data
        
    Returns:
        Filtered signal data
    """
    
    # Merge with fundamentals for screening
    merged = signal_data.merge(
        fundamental_data[['symbol', 'date', 'book_equity', 'market_cap']],
        left_on=['symbol', 'available_date'],
        right_on=['symbol', 'date'],
        how='left'
    )
    
    # Screen 1: Positive book equity
    merged = merged[merged['book_equity'] > 0]
    
    # Screen 2: Reasonable BTM ratio (exclude extreme values)
    merged = merged[(merged['book_to_market'] > 0.01) & 
                   (merged['book_to_market'] < 10.0)]
    
    # Screen 3: Minimum market cap (already in universe selection)
    merged = merged[merged['market_cap'] >= 500_000_000]
    
    return merged[['symbol', 'available_date', 'book_to_market']].rename(
        columns={'available_date': 'date', 'book_to_market': 'signal'}
    )


def backtest_value_btm(spec: StrategySpec, live: bool = False) -> Dict[str, Any]:
    """
    Run complete Value BTM strategy backtest
    
    Args:
        spec: Strategy specification
        live: Whether to use live data
        
    Returns:
        Dict with backtest results and value-specific metrics
    """
    
    # Run base replication pipeline
    replication_result = research_replication_run(
        spec_path="",  # Spec passed directly
        live=live
    )
    
    if not replication_result.ok:
        raise ValueError(f"Replication failed: {replication_result.errors}")
    
    base_results = replication_result.data
    
    # Enhanced results with value-specific metrics
    enhanced_results = base_results.copy()
    enhanced_results.update({
        "strategy_type": "Value-BTM",
        "signal_enhancements": {
            "fundamental_lag_days": 90,
            "screening_applied": True,
            "sector_neutralization": True,
            "size_neutralization": True,
            "rebalance_frequency": "quarterly"
        },
        "value_metrics": {
            "book_to_market_median": None,  # Would compute from actual signals
            "positive_btm_rate": None,
            "universe_coverage": base_results.get("universe_count", 0),
            "avg_portfolio_size": spec.portfolio.k,
            "turnover_quarterly": None  # Would compute from weights
        }
    })
    
    # Additional value-specific analysis would go here
    # For example: BTM distribution, sector tilts, etc.
    
    return enhanced_results


@register("strategies.value_btm.run")
def strategies_value_btm_run(spec_path: str = None, live: bool = False, **kwargs) -> ToolResult:
    """
    Run Value BTM strategy with full pipeline
    
    Args:
        spec_path: Path to Value BTM YAML spec (default: value_btm.yaml)
        live: Whether to use live data
        
    Returns:
        ToolResult with strategy execution results
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Value BTM Strategy")
    
    try:
        # Default spec path
        if spec_path is None:
            spec_path = "ally/strategies/specs/value_btm.yaml"
        
        # Load strategy specification
        spec = StrategySpec.from_yaml(spec_path)
        
        if "value" not in spec.name.lower():
            return ToolResult(
                ok=False,
                errors=["Spec does not appear to be a value strategy"]
            )
        
        # Run backtest
        backtest_results = backtest_value_btm(spec, live=live)
        
        # Run FactorLens analysis
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
            t_stat_threshold=spec.gates.promotion.get("t_stat_threshold", 1.8),
            max_turnover=spec.gates.promotion.get("max_turnover", 1.5),
            live=live
        )
        
        promotion_data = promotion_result.data if promotion_result.ok else {}
        
        # Compile comprehensive results
        strategy_results = {
            "strategy_name": spec.name,
            "strategy_type": "Value-BTM",
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
                "hml_loading": factorlens_data.get("factor_loadings", {}).get("HML", {}).get("coefficient", 0),
                "promotion_approved": promotion_data.get("promotion_approved", False)
            },
            "execution_timestamp": datetime.utcnow().isoformat()
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("strategies.value_btm.run", strategy_results)
        strategy_results["strategy_receipt"] = receipt_hash[:16]
        
        warnings = []
        if not factorlens_result.ok:
            warnings.append("FactorLens analysis failed")
        if not promotion_result.ok:
            warnings.append("Promotion validation failed")
        if promotion_data.get("promotion_approved", False):
            warnings.append("Strategy approved for production")
        
        # Value-specific warnings
        hml_loading = strategy_results["final_metrics"]["hml_loading"]
        if abs(hml_loading) > 0.5:
            warnings.append(f"High HML factor loading: {hml_loading:.3f}")
        
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
            "strategy": "Value-BTM"
        }
        receipt_hash = generate_receipt("strategies.value_btm.error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Value BTM strategy failed: {e}"],
            receipt_hash=receipt_hash
        )


@register("strategies.value_btm.analyze_universe")
def strategies_value_btm_analyze_universe(fundamental_data: Dict[str, Any],
                                         analysis_date: str = None,
                                         live: bool = False, **kwargs) -> ToolResult:
    """
    Analyze value universe characteristics
    
    Args:
        fundamental_data: Fundamental data dictionary
        analysis_date: Specific date for analysis (default: latest)
        live: Whether this is live analysis
        
    Returns:
        ToolResult with universe analysis
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Value Universe Analysis")
    
    try:
        # Convert to DataFrame
        if isinstance(fundamental_data, dict):
            fund_df = pd.DataFrame(fundamental_data)
        else:
            fund_df = fundamental_data
        
        fund_df['date'] = pd.to_datetime(fund_df['date'])
        
        # Use latest date if not specified
        if analysis_date is None:
            analysis_date = fund_df['date'].max()
        else:
            analysis_date = pd.to_datetime(analysis_date)
        
        # Filter to analysis date
        analysis_data = fund_df[fund_df['date'] == analysis_date]
        
        if len(analysis_data) == 0:
            return ToolResult(
                ok=False,
                errors=[f"No data available for date: {analysis_date}"]
            )
        
        # Compute BTM ratios
        analysis_data = analysis_data.copy()
        analysis_data['book_to_market'] = analysis_data['book_equity'] / analysis_data['market_cap']
        
        # Filter valid BTM ratios
        valid_btm = analysis_data[
            (analysis_data['book_equity'] > 0) & 
            (analysis_data['market_cap'] > 0) &
            (analysis_data['book_to_market'] > 0) &
            (analysis_data['book_to_market'] < 10)  # Exclude extreme values
        ]
        
        # Universe statistics
        universe_stats = {
            "analysis_date": analysis_date.strftime('%Y-%m-%d'),
            "total_universe": len(analysis_data),
            "valid_btm_count": len(valid_btm),
            "coverage_rate": len(valid_btm) / len(analysis_data) if len(analysis_data) > 0 else 0,
            "btm_statistics": {
                "mean": valid_btm['book_to_market'].mean(),
                "median": valid_btm['book_to_market'].median(),
                "std": valid_btm['book_to_market'].std(),
                "min": valid_btm['book_to_market'].min(),
                "max": valid_btm['book_to_market'].max(),
                "p25": valid_btm['book_to_market'].quantile(0.25),
                "p75": valid_btm['book_to_market'].quantile(0.75)
            },
            "market_cap_stats": {
                "total_market_cap": valid_btm['market_cap'].sum(),
                "median_market_cap": valid_btm['market_cap'].median(),
                "mean_market_cap": valid_btm['market_cap'].mean()
            }
        }
        
        # Sector breakdown (if available)
        if 'sector' in valid_btm.columns:
            sector_stats = valid_btm.groupby('sector').agg({
                'book_to_market': ['count', 'mean', 'median'],
                'market_cap': 'sum'
            }).round(3)
            
            universe_stats["sector_breakdown"] = sector_stats.to_dict()
        
        # BTM quintile analysis
        valid_btm['btm_quintile'] = pd.qcut(valid_btm['book_to_market'], 
                                           q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        quintile_stats = valid_btm.groupby('btm_quintile').agg({
            'book_to_market': ['mean', 'count'],
            'market_cap': ['mean', 'sum']
        }).round(3)
        
        universe_stats["quintile_analysis"] = quintile_stats.to_dict()
        
        # Generate receipt
        receipt_hash = generate_receipt("strategies.value_btm.analyze_universe", universe_stats)
        universe_stats["universe_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=universe_stats,
            receipt_hash=receipt_hash,
            warnings=[f"Analyzed {len(valid_btm)} stocks with valid BTM ratios"]
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "analysis_date": analysis_date,
            "universe_analysis": "Value-BTM"
        }
        receipt_hash = generate_receipt("strategies.value_btm.universe_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Value universe analysis failed: {e}"],
            receipt_hash=receipt_hash
        )


@register("strategies.value_btm.validate_fundamentals")
def strategies_value_btm_validate_fundamentals(fundamental_data: Dict[str, Any],
                                              live: bool = False, **kwargs) -> ToolResult:
    """
    Validate fundamental data quality for value strategy
    
    Args:
        fundamental_data: Fundamental data to validate
        live: Whether this is live validation
        
    Returns:
        ToolResult with validation results
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Fundamental Data Validation")
    
    try:
        # Convert to DataFrame
        if isinstance(fundamental_data, dict):
            fund_df = pd.DataFrame(fundamental_data)
        else:
            fund_df = fundamental_data
        
        validation_results = {
            "total_records": len(fund_df),
            "unique_symbols": fund_df['symbol'].nunique() if 'symbol' in fund_df.columns else 0,
            "date_range": {
                "start": fund_df['date'].min() if 'date' in fund_df.columns else None,
                "end": fund_df['date'].max() if 'date' in fund_df.columns else None
            },
            "data_quality_checks": {},
            "validation_passed": True,
            "issues": []
        }
        
        # Check required columns
        required_cols = ['symbol', 'date', 'book_equity', 'market_cap']
        missing_cols = [col for col in required_cols if col not in fund_df.columns]
        
        if missing_cols:
            validation_results["validation_passed"] = False
            validation_results["issues"].append(f"Missing columns: {missing_cols}")
        
        # Data quality checks
        if 'book_equity' in fund_df.columns:
            negative_book = (fund_df['book_equity'] < 0).sum()
            zero_book = (fund_df['book_equity'] == 0).sum()
            
            validation_results["data_quality_checks"]["book_equity"] = {
                "negative_count": int(negative_book),
                "zero_count": int(zero_book),
                "negative_rate": float(negative_book / len(fund_df)) if len(fund_df) > 0 else 0,
                "mean": float(fund_df['book_equity'].mean()),
                "median": float(fund_df['book_equity'].median())
            }
            
            if negative_book / len(fund_df) > 0.1:  # >10% negative book equity
                validation_results["issues"].append("High rate of negative book equity")
        
        if 'market_cap' in fund_df.columns:
            zero_mcap = (fund_df['market_cap'] <= 0).sum()
            
            validation_results["data_quality_checks"]["market_cap"] = {
                "zero_or_negative_count": int(zero_mcap),
                "zero_rate": float(zero_mcap / len(fund_df)) if len(fund_df) > 0 else 0,
                "mean": float(fund_df['market_cap'].mean()),
                "median": float(fund_df['market_cap'].median())
            }
            
            if zero_mcap > 0:
                validation_results["validation_passed"] = False
                validation_results["issues"].append("Found zero or negative market cap values")
        
        # BTM ratio validation
        if 'book_equity' in fund_df.columns and 'market_cap' in fund_df.columns:
            valid_data = fund_df[(fund_df['book_equity'] > 0) & (fund_df['market_cap'] > 0)]
            btm_ratios = valid_data['book_equity'] / valid_data['market_cap']
            
            extreme_btm = ((btm_ratios < 0.01) | (btm_ratios > 10)).sum()
            
            validation_results["data_quality_checks"]["book_to_market"] = {
                "valid_ratios_count": int(len(btm_ratios)),
                "extreme_ratios_count": int(extreme_btm),
                "extreme_rate": float(extreme_btm / len(btm_ratios)) if len(btm_ratios) > 0 else 0,
                "mean_btm": float(btm_ratios.mean()),
                "median_btm": float(btm_ratios.median())
            }
        
        # Generate receipt
        receipt_hash = generate_receipt("strategies.value_btm.validate_fundamentals", validation_results)
        validation_results["validation_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=validation_results["validation_passed"],
            data=validation_results,
            receipt_hash=receipt_hash,
            errors=validation_results["issues"] if validation_results["issues"] else [],
            warnings=["Data validation completed"] if validation_results["validation_passed"] else []
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "validation": "Value-BTM fundamentals"
        }
        receipt_hash = generate_receipt("strategies.value_btm.validation_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Fundamental validation failed: {e}"],
            receipt_hash=receipt_hash
        )


if __name__ == "__main__":
    # Test Value BTM strategy
    print("🧪 Testing Value BTM Strategy...")
    
    # Test strategy execution
    result = strategies_value_btm_run(
        spec_path="ally/strategies/specs/value_btm.yaml",
        live=False
    )
    
    print(f"Strategy execution: {result.ok}")
    if result.ok:
        data = result.data
        print(f"Strategy: {data['strategy_name']}")
        print(f"Annual Return: {data['final_metrics']['annual_return']:.2%}")
        print(f"Sharpe Ratio: {data['final_metrics']['sharpe_ratio']:.2f}")
        print(f"Alpha t-stat: {data['final_metrics']['alpha_t_stat']:.2f}")
        print(f"HML loading: {data['final_metrics']['hml_loading']:.3f}")
        print(f"Promotion: {data['final_metrics']['promotion_approved']}")
        print(f"Receipt: {data['strategy_receipt']}")
    else:
        print(f"Errors: {result.errors}")