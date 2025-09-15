"""
FactorLens gate - Fama-French 5-factor + Momentum regression analysis
Regresses strategy returns on FF5+Mom factors with HAC errors
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats
import warnings

from ally.utils.receipts import generate_receipt
from ally.utils.gating import check_live_mode_allowed
from ally.core.tool_registry import register, ToolResult


def generate_mock_ff5mom_factors(start_date: str, end_date: str, frequency: str = 'daily') -> pd.DataFrame:
    """Generate mock Fama-French 5-factor + Momentum data for CI"""
    # Create date range
    dates = pd.date_range(start_date, end_date, freq='D' if frequency == 'daily' else 'M')
    
    # Mock factor loadings based on historical patterns
    np.random.seed(42)  # Deterministic for CI
    
    factors_data = []
    for date in dates:
        # Generate realistic factor returns
        factors_data.append({
            'date': date,
            'MKT_RF': np.random.normal(0.0004, 0.012),  # ~10% annual excess return, 19% vol
            'SMB': np.random.normal(0.0001, 0.008),     # Small-minus-big
            'HML': np.random.normal(0.0002, 0.009),     # High-minus-low book-to-market
            'RMW': np.random.normal(0.0001, 0.007),     # Robust-minus-weak profitability
            'CMA': np.random.normal(-0.0001, 0.006),    # Conservative-minus-aggressive investment
            'MOM': np.random.normal(0.0003, 0.014),     # Momentum factor
            'RF': 0.000015  # Risk-free rate (~4% annual)
        })
    
    return pd.DataFrame(factors_data)


def compute_hac_standard_errors(residuals: np.ndarray, X: np.ndarray, lags: int = 4) -> np.ndarray:
    """Compute Heteroskedasticity and Autocorrelation Consistent (HAC) standard errors"""
    n, k = X.shape
    
    # Compute meat of sandwich estimator
    S = np.zeros((k, k))
    
    for h in range(-lags, lags + 1):
        gamma_h = 0.0
        
        if h == 0:
            # Variance term
            for t in range(n):
                u_t = residuals[t]
                x_t = X[t].reshape(-1, 1)
                S += u_t**2 * np.dot(x_t, x_t.T)
        else:
            # Covariance terms with Bartlett kernel
            weight = 1 - abs(h) / (lags + 1)
            
            for t in range(abs(h), n):
                if h > 0:
                    u_t, u_s = residuals[t], residuals[t-h]
                    x_t, x_s = X[t].reshape(-1, 1), X[t-h].reshape(-1, 1)
                else:
                    u_t, u_s = residuals[t+h], residuals[t]
                    x_t, x_s = X[t+h].reshape(-1, 1), X[t].reshape(-1, 1)
                
                S += weight * (u_t * u_s * np.dot(x_t, x_s.T) + u_s * u_t * np.dot(x_s, x_t.T))
    
    S = S / n
    
    # Bread of sandwich estimator
    XTX_inv = np.linalg.inv(np.dot(X.T, X) / n)
    
    # HAC variance-covariance matrix
    V_hac = np.dot(np.dot(XTX_inv, S), XTX_inv) / n
    
    return np.sqrt(np.diag(V_hac))


def run_ff5mom_regression(strategy_returns: pd.DataFrame, factor_data: pd.DataFrame, 
                         hac_lags: int = 4) -> Dict[str, Any]:
    """Run FF5+Mom regression with HAC standard errors"""
    
    # Merge strategy returns with factors
    merged = strategy_returns.merge(factor_data, on='date', how='inner')
    merged = merged.dropna()
    
    if len(merged) < 50:
        raise ValueError(f"Insufficient data for regression: {len(merged)} observations")
    
    # Calculate excess returns
    merged['excess_return'] = merged['portfolio_return'] - merged['RF']
    
    # Set up regression
    y = merged['excess_return'].values
    factor_names = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    X_factors = merged[factor_names].values
    
    # Add constant for alpha
    X = np.column_stack([np.ones(len(X_factors)), X_factors])
    
    # OLS regression
    XTX_inv = np.linalg.inv(np.dot(X.T, X))
    coefficients = np.dot(np.dot(XTX_inv, X.T), y)
    
    # Calculate fitted values and residuals
    fitted_values = np.dot(X, coefficients)
    residuals = y - fitted_values
    
    # Standard OLS standard errors
    mse = np.sum(residuals**2) / (len(y) - len(coefficients))
    ols_se = np.sqrt(np.diag(XTX_inv * mse))
    
    # HAC standard errors
    hac_se = compute_hac_standard_errors(residuals, X, lags=hac_lags)
    
    # t-statistics using HAC standard errors
    t_stats = coefficients / hac_se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=len(y) - len(coefficients)))
    
    # R-squared
    tss = np.sum((y - np.mean(y))**2)
    rss = np.sum(residuals**2)
    r_squared = 1 - rss / tss
    
    # Prepare results
    results = {
        'alpha': {
            'coefficient': coefficients[0],
            'ols_se': ols_se[0],
            'hac_se': hac_se[0],
            't_stat': t_stats[0],
            'p_value': p_values[0],
            'annualized': coefficients[0] * 252,  # Daily to annual
            'annualized_se': hac_se[0] * np.sqrt(252)
        },
        'factor_loadings': {},
        'regression_stats': {
            'r_squared': r_squared,
            'observations': len(y),
            'residual_std': np.std(residuals),
            'hac_lags': hac_lags
        },
        'residuals': residuals.tolist(),
        'factor_names': factor_names
    }
    
    # Factor loadings
    for i, factor in enumerate(factor_names):
        results['factor_loadings'][factor] = {
            'coefficient': coefficients[i+1],
            'ols_se': ols_se[i+1], 
            'hac_se': hac_se[i+1],
            't_stat': t_stats[i+1],
            'p_value': p_values[i+1]
        }
    
    return results


@register("research.factorlens.analyze")
def research_factorlens_analyze(backtest_results: Dict[str, Any], spec_name: str,
                               live: bool = False, hac_lags: int = 4, **kwargs) -> ToolResult:
    """
    Run FactorLens analysis on strategy backtest results
    
    Args:
        backtest_results: Results from replication pipeline
        spec_name: Strategy specification name
        live: Whether to use live factor data
        hac_lags: Number of lags for HAC standard errors
        
    Returns:
        ToolResult with alpha decomposition and factor loadings
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "FactorLens Analysis")
    
    try:
        # Extract portfolio returns
        portfolio_returns = backtest_results.get("portfolio_returns", {})
        if not portfolio_returns:
            return ToolResult(
                ok=False,
                errors=["No portfolio returns found in backtest results"]
            )
        
        # Convert to DataFrame
        returns_df = pd.DataFrame([
            {"date": pd.to_datetime(date), "portfolio_return": ret}
            for date, ret in portfolio_returns.items()
        ]).sort_values('date')
        
        if len(returns_df) < 50:
            return ToolResult(
                ok=False,
                errors=[f"Insufficient return data: {len(returns_df)} observations (minimum: 50)"]
            )
        
        # Get factor data
        if live:
            raise NotImplementedError("Live factor data integration not implemented")
        else:
            # Generate mock factor data for CI
            start_date = returns_df['date'].min().strftime('%Y-%m-%d')
            end_date = returns_df['date'].max().strftime('%Y-%m-%d')
            factor_data = generate_mock_ff5mom_factors(start_date, end_date)
        
        # Run regression analysis
        regression_results = run_ff5mom_regression(returns_df, factor_data, hac_lags)
        
        # Alpha significance test
        alpha_t_stat = regression_results['alpha']['t_stat']
        alpha_significant = abs(alpha_t_stat) >= 2.0  # |t| >= 2 threshold
        
        # Factor exposure analysis
        significant_factors = []
        for factor, stats in regression_results['factor_loadings'].items():
            if abs(stats['t_stat']) >= 2.0:
                significant_factors.append({
                    'factor': factor,
                    'loading': stats['coefficient'],
                    't_stat': stats['t_stat'],
                    'p_value': stats['p_value']
                })
        
        # Compile analysis results
        analysis_results = {
            "spec_name": spec_name,
            "alpha_annual": regression_results['alpha']['annualized'],
            "alpha_se_annual": regression_results['alpha']['annualized_se'],
            "alpha_t_stat": alpha_t_stat,
            "alpha_p_value": regression_results['alpha']['p_value'],
            "alpha_significant": alpha_significant,
            "r_squared": regression_results['regression_stats']['r_squared'],
            "observations": regression_results['regression_stats']['observations'],
            "significant_factors": significant_factors,
            "factor_loadings": regression_results['factor_loadings'],
            "residual_alpha": regression_results['alpha']['annualized'],  # For promotion gate
            "hac_adjusted": True,
            "hac_lags": hac_lags,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("research.factorlens.analyze", analysis_results)
        
        # Add receipt to results
        analysis_results["factorlens_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=analysis_results,
            receipt_hash=receipt_hash,
            warnings=["Using mock factor data for CI"] if not live else []
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e), 
            "spec_name": spec_name,
            "live": live
        }
        receipt_hash = generate_receipt("research.factorlens.error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"FactorLens analysis failed: {e}"],
            receipt_hash=receipt_hash
        )


@register("research.factorlens.create_factor_data")
def research_factorlens_create_factor_data(start_date: str, end_date: str,
                                          frequency: str = "daily", 
                                          live: bool = False, **kwargs) -> ToolResult:
    """
    Create or fetch factor data for analysis
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)  
        frequency: Data frequency (daily/monthly)
        live: Whether to fetch live data
        
    Returns:
        ToolResult with factor data
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Factor Data Creation")
    
    try:
        if live:
            raise NotImplementedError("Live factor data fetching not implemented")
        
        # Generate mock factor data
        factor_data = generate_mock_ff5mom_factors(start_date, end_date, frequency)
        
        # Summary statistics
        summary_stats = {}
        factor_cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
        
        for factor in factor_cols:
            summary_stats[factor] = {
                'mean': factor_data[factor].mean(),
                'std': factor_data[factor].std(),
                'min': factor_data[factor].min(),
                'max': factor_data[factor].max(),
                'observations': len(factor_data)
            }
        
        result = {
            "factor_data": factor_data.to_dict(),
            "summary_stats": summary_stats,
            "date_range": {
                "start": start_date,
                "end": end_date
            },
            "frequency": frequency,
            "factors": factor_cols,
            "observations": len(factor_data),
            "creation_timestamp": datetime.utcnow().isoformat()
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("research.factorlens.create_factor_data", result)
        result["factor_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=result,
            receipt_hash=receipt_hash,
            warnings=["Generated mock factor data for CI"] if not live else []
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "start_date": start_date,
            "end_date": end_date
        }
        receipt_hash = generate_receipt("research.factorlens.factor_data_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Factor data creation failed: {e}"],
            receipt_hash=receipt_hash
        )


if __name__ == "__main__":
    # Test FactorLens functionality
    print("🧪 Testing FactorLens...")
    
    # Generate mock backtest results
    mock_backtest = {
        "portfolio_returns": {
            f"2024-01-{i:02d}": np.random.normal(0.001, 0.02) 
            for i in range(1, 32)
        }
    }
    
    # Test analysis
    result = research_factorlens_analyze(
        backtest_results=mock_backtest,
        spec_name="test_strategy",
        live=False
    )
    
    print(f"Analysis success: {result.ok}")
    if result.ok:
        data = result.data
        print(f"Alpha (annual): {data['alpha_annual']:.4f}")
        print(f"Alpha t-stat: {data['alpha_t_stat']:.2f}")
        print(f"R-squared: {data['r_squared']:.4f}")
        print(f"Significant factors: {len(data['significant_factors'])}")
        print(f"Receipt: {data['factorlens_receipt']}")
    else:
        print(f"Errors: {result.errors}")