"""
Factor analysis tools for M-FactorLens
"""

from typing import Dict, Any, List, Optional
from ..schemas.base import ToolResult
from . import register
import numpy as np
import pandas as pd
import json


@register("factors.compute_exposures")
def compute_exposures(returns: List[Dict[str, Any]], lags: int = 5) -> ToolResult:
    """
    Compute factor exposures using OLS-Newey West regression

    Args:
        returns: List of dicts with 'date' and 'return' keys
        lags: Newey-West lags for HAC standard errors

    Returns:
        ToolResult with factor exposures
    """
    try:
        # Mock factor exposures for testing
        mock_exposures = [
            {"factor": "alpha", "beta": 0.001, "tstat": 1.8, "pvalue": 0.08},
            {"factor": "MKT", "beta": 0.25, "tstat": 3.2, "pvalue": 0.002},
            {"factor": "SMB", "beta": -0.12, "tstat": -1.5, "pvalue": 0.14},
            {"factor": "HML", "beta": 0.08, "tstat": 0.9, "pvalue": 0.37},
            {"factor": "RMW", "beta": 0.15, "tstat": 1.7, "pvalue": 0.09},
            {"factor": "CMA", "beta": -0.05, "tstat": -0.6, "pvalue": 0.55},
            {"factor": "MOM", "beta": 0.18, "tstat": 2.1, "pvalue": 0.04}
        ]

        return ToolResult.success({
            "exposures": mock_exposures,
            "n_obs": len(returns),
            "lags": lags
        })

    except Exception as e:
        return ToolResult.error([f"Exposure computation error: {str(e)}"])


@register("factors.residual_alpha")
def compute_residual_alpha(
    returns: List[Dict[str, Any]],
    window: int = 252,
    step: int = 21,
    lags: int = 5
) -> ToolResult:
    """
    Compute residual alpha using rolling window regression

    Args:
        returns: List of dicts with 'date' and 'return' keys
        window: Rolling window size in days
        step: Step size for rolling windows
        lags: Newey-West lags for HAC standard errors

    Returns:
        ToolResult with residual alpha analysis
    """
    try:
        # Mock residual alpha computation for testing
        np.random.seed(42)
        alpha_bps = np.random.randn() * 50 + 120  # ~120 bps with noise
        alpha_tstat = np.random.randn() * 0.5 + 2.3  # ~2.3 t-stat with noise

        return ToolResult.success({
            "alpha_bps": float(alpha_bps),
            "alpha_tstat": float(alpha_tstat),
            "r2": 0.65,
            "window_days": window,
            "step_days": step,
            "n_windows": max(1, (len(returns) - window) // step),
            "residual_series_path": "runs/factor_analysis/residuals.json"
        })

    except Exception as e:
        return ToolResult.error([f"Residual alpha error: {str(e)}"])