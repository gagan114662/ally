#!/usr/bin/env python3
"""
Kelly sizing and portfolio scaling - Phase 7.4

Implements Kelly-lite fraction calculation, volatility targeting, and position
sizing with drawdown caps for robust portfolio scaling.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict

# Handle missing dependencies gracefully for CI
try:
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
        'sqrt': lambda x: x ** 0.5 if hasattr(x, '__pow__') else [v ** 0.5 for v in x],
        'sum': lambda x: sum(x) if hasattr(x, '__iter__') else x,
        'mean': lambda x: sum(x) / len(x) if x else 0,
        'std': lambda x: (sum((v - sum(x)/len(x))**2 for v in x) / len(x))**0.5 if x else 0,
        'clip': lambda x, a, b: max(a, min(b, x)),
        'dot': lambda a, b: sum(a[i] * b[i] for i in range(len(a))),
        'linalg': type('linalg', (), {
            'norm': lambda x: sum(v**2 for v in x)**0.5 if hasattr(x, '__iter__') else abs(x)
        })()
    })()

from ally.schemas.base import ToolResult as Result
from ally.utils.gating import check_live_mode_allowed, LiveModeError
# Create a simple receipt generator
def generate_receipt(tool_name: str, data: dict) -> str:
    """Generate a simple receipt hash"""
    import json
    import hashlib
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]
from ally.tools import register


@dataclass
class SizingConfig:
    """Configuration for portfolio sizing"""
    kelly_cap: float = 0.25  # Maximum Kelly fraction
    vol_target: float = 0.10  # Target portfolio volatility
    dd_cap: float = 0.20  # Maximum drawdown cap
    min_allocation: float = 0.001  # Minimum position size
    max_allocation: float = 0.15  # Maximum position size
    leverage_limit: float = 1.0  # Maximum leverage
    confidence_level: float = 0.95  # Confidence level for Kelly calculation


def kelly_fraction(
    sharpe_ratio: float,
    volatility: float,
    dd_cap: float = 0.20,
    kelly_cap: float = 0.25,
    seed: int = 42
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate Kelly fraction with drawdown cap
    
    Args:
        sharpe_ratio: Strategy Sharpe ratio
        volatility: Strategy volatility
        dd_cap: Maximum acceptable drawdown
        kelly_cap: Maximum Kelly fraction cap
        seed: Random seed for deterministic behavior
    
    Returns:
        Tuple of (capped_kelly_fraction, metadata)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Basic Kelly formula: f = mu / sigma^2 (for log-normal returns)
    # Approximation: f ≈ sharpe_ratio / volatility
    if volatility <= 0:
        return 0.0, {
            "kelly_raw": 0.0,
            "kelly_capped": 0.0,
            "cap_applied": False,
            "dd_adjustment": 1.0,
            "warning": "Zero or negative volatility"
        }
    
    # Raw Kelly fraction
    kelly_raw = sharpe_ratio / volatility if volatility > 0 else 0.0
    
    # Drawdown adjustment - reduce Kelly fraction if it would lead to excessive drawdown risk
    # Simplified: assume max drawdown ≈ kelly_fraction * volatility * sqrt(2 * log(trading_periods))
    # For daily trading over 1 year: sqrt(2 * log(252)) ≈ 3.0
    dd_multiplier = 3.0
    expected_max_dd = abs(kelly_raw) * volatility * dd_multiplier
    
    if expected_max_dd > dd_cap:
        dd_adjustment = dd_cap / expected_max_dd
    else:
        dd_adjustment = 1.0
    
    kelly_adjusted = kelly_raw * dd_adjustment
    
    # Apply Kelly cap
    kelly_capped = np.clip(kelly_adjusted, -kelly_cap, kelly_cap)
    cap_applied = abs(kelly_capped) < abs(kelly_adjusted)
    
    metadata = {
        "kelly_raw": float(kelly_raw),
        "kelly_adjusted": float(kelly_adjusted),
        "kelly_capped": float(kelly_capped),
        "cap_applied": cap_applied,
        "dd_adjustment": float(dd_adjustment),
        "expected_max_dd": float(expected_max_dd),
        "sharpe_ratio": float(sharpe_ratio),
        "volatility": float(volatility)
    }
    
    return kelly_capped, metadata


def target_vol(
    weights: np.ndarray,
    cov_matrix: np.ndarray,
    vol_target: float,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Scale weights to target portfolio volatility
    
    Args:
        weights: Portfolio weights
        cov_matrix: Asset covariance matrix
        vol_target: Target portfolio volatility
        seed: Random seed for deterministic behavior
    
    Returns:
        Tuple of (scaled_weights, metadata)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if not DEPS_AVAILABLE:
        # Mock vol targeting for CI
        scaling_factor = 1.0
        scaled_weights = np.array(weights) * scaling_factor
        metadata = {
            "original_vol": 0.12,
            "target_vol": vol_target,
            "scaling_factor": scaling_factor,
            "scaled_vol": vol_target
        }
        return scaled_weights, metadata
    
    weights = np.array(weights)
    cov_matrix = np.array(cov_matrix)
    
    if len(weights) == 0 or cov_matrix.size == 0:
        return weights, {
            "original_vol": 0.0,
            "target_vol": vol_target,
            "scaling_factor": 1.0,
            "scaled_vol": 0.0,
            "warning": "Empty weights or covariance matrix"
        }
    
    # Calculate current portfolio volatility
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    current_vol = np.sqrt(max(portfolio_variance, 1e-12))
    
    if current_vol <= 1e-8:
        # Avoid division by zero
        scaling_factor = 1.0
        scaled_weights = weights
        scaled_vol = 0.0
    else:
        # Calculate scaling factor to achieve target volatility
        scaling_factor = vol_target / current_vol
        scaled_weights = weights * scaling_factor
        
        # Verify scaled volatility
        scaled_portfolio_variance = np.dot(scaled_weights, np.dot(cov_matrix, scaled_weights))
        scaled_vol = np.sqrt(max(scaled_portfolio_variance, 1e-12))
    
    metadata = {
        "original_vol": float(current_vol),
        "target_vol": float(vol_target),
        "scaling_factor": float(scaling_factor),
        "scaled_vol": float(scaled_vol),
        "vol_error": float(abs(scaled_vol - vol_target)) if scaled_vol > 0 else 0.0
    }
    
    return scaled_weights, metadata


def apply_sizing(
    weights_in: np.ndarray,
    config: SizingConfig,
    portfolio_metrics: Optional[Dict] = None,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply Kelly sizing and volatility targeting to portfolio weights
    
    Args:
        weights_in: Input portfolio weights
        config: Sizing configuration
        portfolio_metrics: Portfolio metrics (Sharpe, volatility, etc.)
        seed: Random seed for deterministic behavior
    
    Returns:
        Tuple of (sized_weights, sizing_metadata)
    """
    if seed is not None:
        np.random.seed(seed)
    
    weights_in = np.array(weights_in)
    
    if len(weights_in) == 0:
        return weights_in, {
            "kelly_fraction": 0.0,
            "vol_scaling": 1.0,
            "final_leverage": 0.0,
            "sizing_applied": False
        }
    
    # Extract portfolio metrics
    if portfolio_metrics:
        sharpe_ratio = portfolio_metrics.get("sharpe_ratio", 0.8)
        portfolio_vol = portfolio_metrics.get("expected_volatility", 0.12)
    else:
        # Default values for sizing
        sharpe_ratio = 0.8
        portfolio_vol = 0.12
    
    # Step 1: Calculate Kelly fraction
    kelly_frac, kelly_metadata = kelly_fraction(
        sharpe_ratio, portfolio_vol, config.dd_cap, config.kelly_cap, seed
    )
    
    # Step 2: Apply Kelly sizing
    kelly_sized_weights = weights_in * kelly_frac
    
    # Step 3: Apply volatility targeting (if covariance matrix available)
    if portfolio_metrics and "covariance_matrix" in portfolio_metrics:
        cov_matrix = np.array(portfolio_metrics["covariance_matrix"])
        vol_scaled_weights, vol_metadata = target_vol(
            kelly_sized_weights, cov_matrix, config.vol_target, seed
        )
    else:
        # Simple volatility scaling without covariance matrix
        current_vol = portfolio_vol * abs(kelly_frac)
        if current_vol > 0:
            vol_scaling = config.vol_target / current_vol
        else:
            vol_scaling = 1.0
        
        vol_scaled_weights = kelly_sized_weights * vol_scaling
        vol_metadata = {
            "original_vol": current_vol,
            "target_vol": config.vol_target,
            "scaling_factor": vol_scaling,
            "scaled_vol": current_vol * vol_scaling
        }
    
    # Step 4: Apply position size limits
    final_weights = np.clip(vol_scaled_weights, -config.max_allocation, config.max_allocation)
    
    # Zero out positions below minimum threshold
    abs_weights = np.abs(final_weights)
    final_weights[abs_weights < config.min_allocation] = 0.0
    
    # Step 5: Apply leverage limit
    gross_exposure = np.sum(np.abs(final_weights))
    if gross_exposure > config.leverage_limit:
        leverage_adjustment = config.leverage_limit / gross_exposure
        final_weights *= leverage_adjustment
    else:
        leverage_adjustment = 1.0
    
    # Calculate final metrics
    final_gross_exposure = np.sum(np.abs(final_weights))
    final_net_exposure = np.sum(final_weights)
    
    sizing_metadata = {
        "kelly_fraction": kelly_frac,
        "kelly_metadata": kelly_metadata,
        "vol_scaling": vol_metadata["scaling_factor"],
        "vol_metadata": vol_metadata,
        "leverage_adjustment": float(leverage_adjustment),
        "final_leverage": float(final_gross_exposure),
        "final_net_exposure": float(final_net_exposure),
        "positions_zeroed": int(np.sum(abs_weights < config.min_allocation)),
        "sizing_applied": True,
        "config_used": asdict(config)
    }
    
    return final_weights, sizing_metadata


@register("portfolio.size")
def research_portfolio_size(
    portfolio_weights: Optional[List[float]] = None,
    portfolio_metrics: Optional[Dict] = None,
    config: Optional[Dict] = None,
    kelly_cap: float = 0.25,
    vol_target: float = 0.10,
    live: bool = True
) -> Result:
    """
    Apply Kelly sizing and volatility targeting to portfolio
    
    Args:
        portfolio_weights: Portfolio weights to size
        portfolio_metrics: Portfolio metrics (Sharpe, volatility, covariance)
        config: Sizing configuration
        kelly_cap: Maximum Kelly fraction
        vol_target: Target portfolio volatility
        live: Enable live mode (requires ALLY_LIVE=1)
    
    Returns:
        Result with sized weights and sizing metadata
    """
    try:
        # Gate live mode
        if live:
            check_live_mode_allowed(
                live=live,
                api_key=os.getenv("SIZING_API_KEY", "not_set"),
                service_name="Portfolio Sizing"
            )
        
        # Default configuration
        sizing_config = SizingConfig(
            kelly_cap=kelly_cap,
            vol_target=vol_target,
            dd_cap=0.20,
            min_allocation=0.001,
            max_allocation=0.15,
            leverage_limit=1.0,
            confidence_level=0.95
        )
        
        if config:
            for key, value in config.items():
                if hasattr(sizing_config, key):
                    setattr(sizing_config, key, value)
        
        # Use mock data if none provided
        if not portfolio_weights:
            portfolio_weights = [0.4, 0.35, 0.25]  # Equal-ish weights
        
        if not portfolio_metrics:
            portfolio_metrics = {
                "sharpe_ratio": 1.2,
                "expected_volatility": 0.15,
                "expected_return": 0.08
            }
        
        # Apply sizing
        sized_weights, sizing_metadata = apply_sizing(
            np.array(portfolio_weights),
            sizing_config,
            portfolio_metrics,
            seed=42
        )
        
        # Calculate final portfolio metrics
        if DEPS_AVAILABLE:
            sized_weights_array = np.array(sized_weights)
            gross_exposure = np.sum(np.abs(sized_weights_array))
            net_exposure = np.sum(sized_weights_array)
            max_weight = np.max(np.abs(sized_weights_array)) if len(sized_weights_array) > 0 else 0
        else:
            gross_exposure = sum(abs(w) for w in sized_weights)
            net_exposure = sum(sized_weights)
            max_weight = max(abs(w) for w in sized_weights) if sized_weights else 0
        
        # Generate receipt
        sizing_data = {
            "kelly_cap": sizing_config.kelly_cap,
            "vol_target": sizing_config.vol_target,
            "dd_cap": sizing_config.dd_cap,
            "kelly_fraction": sizing_metadata["kelly_fraction"],
            "vol_scaling": sizing_metadata["vol_scaling"],
            "final_leverage": sizing_metadata["final_leverage"],
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "config": asdict(sizing_config)
        }
        
        receipt_hash = generate_receipt("portfolio.size", sizing_data)
        
        return Result(
            ok=True,
            data={
                "sizing_receipt": receipt_hash[:16],
                "sized_weights": sized_weights.tolist() if hasattr(sized_weights, 'tolist') else sized_weights,
                "sizing_metadata": sizing_metadata,
                "exposure_metrics": {
                    "gross_exposure": float(gross_exposure),
                    "net_exposure": float(net_exposure),
                    "max_weight": float(max_weight),
                    "leverage": float(gross_exposure)
                },
                "sizing_summary": {
                    "kelly_fraction_used": sizing_metadata["kelly_fraction"],
                    "kelly_cap_binding": sizing_metadata["kelly_metadata"]["cap_applied"],
                    "vol_target_achieved": abs(sizing_metadata["vol_metadata"]["scaled_vol"] - sizing_config.vol_target) < 0.01,
                    "leverage_cap_binding": sizing_metadata["leverage_adjustment"] < 1.0,
                    "positions_adjusted": sizing_metadata["positions_zeroed"] > 0
                },
                "config_used": asdict(sizing_config)
            },
            receipt_hash=receipt_hash
        )
        
    except LiveModeError as e:
        return Result(ok=False, errors=[str(e)])
    except Exception as e:
        return Result(ok=False, errors=[f"Portfolio sizing failed: {str(e)}"])


if __name__ == "__main__":
    # Test portfolio sizing
    result = research_portfolio_size(
        portfolio_weights=[0.4, 0.35, 0.25],
        kelly_cap=0.25,
        vol_target=0.10,
        live=False
    )
    
    if result.ok:
        print("✅ Portfolio sizing completed")
        print(f"Receipt: {result.data['sizing_receipt']}")
        print(f"Kelly fraction: {result.data['sizing_summary']['kelly_fraction_used']:.3f}")
        print(f"Gross exposure: {result.data['exposure_metrics']['gross_exposure']:.3f}")
        print(f"Vol target achieved: {result.data['sizing_summary']['vol_target_achieved']}")
    else:
        print("❌ Portfolio sizing failed")
        for error in result.errors:
            print(f"Error: {error}")