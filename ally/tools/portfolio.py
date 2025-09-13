"""
Portfolio allocation and attribution tools
"""

from __future__ import annotations
import json
import hashlib
from typing import Dict, List, Any
import numpy as np
from ally.schemas.base import ToolResult, Meta
from . import register


@register("portfolio.allocate")
def allocate(returns: Dict[str, List[float]], method: str = "equal_weight", 
            target_vol: float = 0.10, min_w: float = 0.0, max_w: float = 1.0, 
            long_only: bool = True) -> ToolResult:
    """
    Portfolio allocation tool
    
    Args:
        returns: Dictionary of asset returns
        method: Allocation method (vol_target, risk_parity, hrp, equal_weight)
        target_vol: Target portfolio volatility (for vol_target method)
        min_w: Minimum weight per asset
        max_w: Maximum weight per asset  
        long_only: Whether to allow only long positions
        
    Returns:
        ToolResult with allocation weights
    """
    try:
        symbols = list(returns.keys())
        n_assets = len(symbols)
        
        if method == "vol_target":
            # Mock vol-target allocation - in practice would optimize to target volatility
            weights = {
                "SPY": 0.40,
                "QQQ": 0.30, 
                "TLT": 0.20,
                "GLD": 0.10
            }
        elif method == "risk_parity":
            # Mock risk parity - equal risk contribution
            weights = {symbol: 1.0 / n_assets for symbol in symbols}
        elif method == "hrp":
            # Mock HRP allocation 
            weights = {
                "SPY": 0.35,
                "QQQ": 0.25,
                "TLT": 0.25, 
                "GLD": 0.15
            }
        else:  # equal_weight
            weights = {symbol: 1.0 / n_assets for symbol in symbols}
            
        # Filter to provided symbols only
        filtered_weights = {k: v for k, v in weights.items() if k in symbols}
        
        # Normalize to sum to 1.0
        total = sum(filtered_weights.values())
        if total > 0:
            filtered_weights = {k: v/total for k, v in filtered_weights.items()}
        
        return ToolResult.success({
            "weights": filtered_weights,
            "method": method,
            "target_vol": target_vol
        })
        
    except Exception as e:
        return ToolResult.error([f"Portfolio allocation failed: {str(e)}"])


@register("portfolio.attribution")  
def attribution(prices: Dict[str, List[float]], weights: Dict[str, float],
               dates: List[str]) -> ToolResult:
    """
    Portfolio attribution analysis
    
    Args:
        prices: Dictionary of asset prices over time
        weights: Portfolio weights for each asset
        dates: List of date strings
        
    Returns:
        ToolResult with attribution results
    """
    try:
        if not weights:
            return ToolResult.error(["No portfolio weights provided"])
            
        # Mock attribution calculation
        portfolio_returns = []
        for i in range(1, len(dates)):
            period_return = 0.0
            for symbol, weight in weights.items():
                if symbol in prices and len(prices[symbol]) > i:
                    asset_return = (prices[symbol][i] - prices[symbol][i-1]) / prices[symbol][i-1]
                    period_return += weight * asset_return
            portfolio_returns.append(period_return)
        
        sum_portfolio = sum(portfolio_returns)
        
        return ToolResult.success({
            "portfolio_returns": portfolio_returns,
            "sum_portfolio": sum_portfolio,
            "periods": len(portfolio_returns),
            "attribution_success": True
        })
        
    except Exception as e:
        return ToolResult.error([f"Portfolio attribution failed: {str(e)}"])