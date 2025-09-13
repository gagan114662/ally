"""
Orchestrator with M-FactorLens Gate
"""

from typing import Dict, Any, List, Optional
from ..schemas.base import ToolResult
from ..utils.hashing import hash_inputs
from . import TOOL_REGISTRY, register
import json


@register("orchestrator.factor_gate")
def factor_gate(
    returns: List[Dict[str, Any]],
    max_beta: float = 0.30,
    min_alpha_tstat: float = 2.0,
    window: int = 252,
    step: int = 21,
    lags: int = 5
) -> ToolResult:
    """
    M-FactorLens Gate: Enforce factor exposure limits and alpha significance

    Args:
        returns: List of dicts with 'date' and 'return' keys
        max_beta: Maximum absolute factor beta allowed (default 0.30)
        min_alpha_tstat: Minimum residual alpha t-stat required (default 2.0)
        window: Rolling window size in days
        step: Step size for rolling windows
        lags: Newey-West lags for HAC standard errors

    Returns:
        ToolResult with gate decision and proofs
    """
    try:
        # Step 1: Compute factor exposures
        exposures_result = TOOL_REGISTRY["factors.compute_exposures"](
            returns=returns,
            lags=lags
        )

        if not exposures_result.ok:
            return ToolResult.error(["Failed to compute factor exposures"])

        exposures = exposures_result.data["exposures"]

        # Step 2: Check beta constraints
        betas_ok = True
        violations = []

        for exp in exposures:
            if exp["factor"] == "alpha":
                continue  # Skip intercept

            if abs(exp["beta"]) > max_beta:
                betas_ok = False
                violations.append(f"{exp['factor']}: β={exp['beta']:.3f}")

        # Step 3: Compute residual alpha
        alpha_result = TOOL_REGISTRY["factors.residual_alpha"](
            returns=returns,
            window=window,
            step=step,
            lags=lags
        )

        if not alpha_result.ok:
            return ToolResult.error(["Failed to compute residual alpha"])

        alpha_data = alpha_result.data
        alpha_tstat = alpha_data["alpha_tstat"]
        alpha_bps = alpha_data["alpha_bps"]

        # Step 4: Gate decision
        alpha_pass = alpha_tstat >= min_alpha_tstat
        gate_pass = betas_ok and alpha_pass

        # Step 5: Generate deterministic hash
        gate_input = {
            "n_returns": len(returns),
            "max_beta": max_beta,
            "min_alpha_tstat": min_alpha_tstat,
            "window": window,
            "step": step,
            "lags": lags
        }
        gate_hash = hash_inputs(gate_input)[:16]

        # Prepare output
        output = {
            "gate_pass": gate_pass,
            "betas_ok": betas_ok,
            "alpha_pass": alpha_pass,
            "alpha_tstat": round(alpha_tstat, 3),
            "alpha_bps": round(alpha_bps, 1),
            "violations": violations,
            "gate_hash": gate_hash,
            "exposures": exposures,
            "proofs": {
                "FACTLENS_GATE": "PASS" if gate_pass else "FAIL",
                "RES_ALPHA_T": round(alpha_tstat, 3),
                "BETAS_OK": str(betas_ok).lower(),
                "FACTORLENS_HASH": gate_hash
            }
        }

        return ToolResult.success(output)

    except Exception as e:
        return ToolResult.error([f"Factor gate error: {str(e)}"])


@register("orchestrator.run_pipeline")
def run_pipeline(
    strategy_returns: Optional[List[Dict[str, Any]]] = None,
    enforce_gate: bool = True,
    **kwargs
) -> ToolResult:
    """
    Run full orchestration pipeline with factor gate

    Args:
        strategy_returns: Optional strategy returns to analyze
        enforce_gate: Whether to enforce the factor gate (default True)
        **kwargs: Additional parameters for gate

    Returns:
        ToolResult with pipeline results
    """
    try:
        # Generate synthetic returns if not provided
        if strategy_returns is None:
            # Use a simple synthetic generator for demo
            import numpy as np
            import pandas as pd
            from datetime import datetime, timedelta

            np.random.seed(42)
            dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
            returns = []
            for i, date in enumerate(dates):
                daily_return = np.random.randn() * 0.02 + 0.0003  # 3bps daily drift
                returns.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "return": float(daily_return)
                })
            strategy_returns = returns

        # Run factor gate
        gate_result = factor_gate(
            returns=strategy_returns,
            max_beta=kwargs.get("max_beta", 0.30),
            min_alpha_tstat=kwargs.get("min_alpha_tstat", 2.0),
            window=kwargs.get("window", 252),
            step=kwargs.get("step", 21),
            lags=kwargs.get("lags", 5)
        )

        if not gate_result.is_success:
            return ToolResult.error(["Pipeline failed at factor gate"])

        gate_data = gate_result.data

        # Check gate enforcement
        if enforce_gate and not gate_data["gate_pass"]:
            return ToolResult.success({
                "pipeline_status": "BLOCKED",
                "reason": "Factor gate failed",
                "gate_result": gate_data,
                "message": f"Strategy blocked: alpha t-stat={gate_data['alpha_tstat']:.3f}, "
                          f"violations={gate_data['violations']}"
            })

        # Pipeline continues if gate passed or not enforced
        output = {
            "pipeline_status": "APPROVED" if gate_data["gate_pass"] else "WARNED",
            "gate_result": gate_data,
            "strategy_metrics": {
                "n_returns": len(strategy_returns),
                "alpha_bps": gate_data["alpha_bps"],
                "alpha_tstat": gate_data["alpha_tstat"],
                "factor_exposures": gate_data["exposures"]
            }
        }

        return ToolResult.success(output)

    except Exception as e:
        return ToolResult.error([f"Pipeline error: {str(e)}"])