"""
M9 Orchestrator - Portfolio Integration
Demonstrates portfolio allocation and attribution in multi-asset backtests
"""

from __future__ import annotations
import json
import hashlib
from datetime import datetime
from typing import Dict, List
from ally.schemas.orch import OrchSummary
from ally.tools import TOOL_REGISTRY

def orchestrator_run(symbols: List[str] = None, 
                    start_date: str = "2020-01-01",
                    end_date: str = "2020-01-05") -> OrchSummary:
    """
    Run orchestrated multi-asset backtest with portfolio proofs
    
    Args:
        symbols: List of symbols for portfolio (default: SPY, QQQ, TLT, GLD)
        start_date: Backtest start date
        end_date: Backtest end date
        
    Returns:
        OrchSummary with portfolio allocation and attribution results
    """
    if symbols is None:
        symbols = ["SPY", "QQQ", "TLT", "GLD"]
    
    summary = OrchSummary(
        symbols=symbols,
        backtest_start=start_date,
        backtest_end=end_date,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
    
    # Mock returns data for demonstration (deterministic)
    # In real implementation, this would come from backtesting results
    returns_dict = {
        "SPY": [0.001, -0.002, 0.0005, 0.0015, -0.0007, 0.0009, 0.0012, -0.0011],
        "QQQ": [0.0015, -0.0025, 0.0007, 0.0018, -0.0009, 0.0011, 0.0015, -0.0012],
        "TLT": [0.0004, 0.0002, -0.0001, 0.0003, 0.0001, -0.0002, 0.0001, 0.0002],
        "GLD": [0.0003, 0.0000, 0.0002, 0.0001, 0.0001, 0.0000, 0.0001, 0.0002]
    }
    
    # Mock prices data aligned with returns
    prices_dict = {
        "SPY": [320, 321, 320, 321, 320, 321, 322, 321, 322],
        "QQQ": [210, 211, 210, 212, 211, 212, 213, 212, 213],
        "TLT": [140, 140.1, 140.05, 140.1, 140.15, 140.12, 140.13, 140.11, 140.12],
        "GLD": [145, 145.05, 145.1, 145.12, 145.13, 145.14, 145.15, 145.16, 145.17]
    }
    
    dates_list = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", 
                  "2020-01-07", "2020-01-08", "2020-01-09", "2020-01-10", "2020-01-13"]
    
    # --- PORTFOLIO ALLOCATION (vol-target) ---
    try:
        alloc_res = TOOL_REGISTRY["portfolio.allocate"](
            returns=returns_dict, method="vol_target", target_vol=0.10,
            min_w=0.0, max_w=0.60, long_only=True
        )
        port_weights = alloc_res.data["weights"] if alloc_res.ok else {}
        port_weights_sum = round(sum(port_weights.values()), 6)
    except Exception as e:
        port_weights, port_weights_sum = {}, 0.0
    
    # --- ATTRIBUTION (constant weights MVP) ---
    try:
        attr_res = TOOL_REGISTRY["portfolio.attribution"](
            prices=prices_dict, weights=port_weights, dates=dates_list
        )
        attribution_ok = bool(attr_res.ok)
        # Deterministic hash for reproducibility  
        port_det_hash = hashlib.sha1(json.dumps({
            "w": port_weights, "sum_port": attr_res.data.get("sum_portfolio", 0.0)
        }, sort_keys=True).encode()).hexdigest()
    except Exception:
        attribution_ok = False
        port_det_hash = "na"
    
    # Attach to orchestrator summary
    summary.port_weights = port_weights
    summary.port_weights_sum = port_weights_sum
    summary.attribution_ok = attribution_ok
    summary.port_det_hash = port_det_hash
    summary.execution_success = bool(port_weights_sum > 0.99)
    
    # Print CI proofs (stdout) so the M9 job picks them up
    print(f"PROOF:ORCH_PORT_WEIGHTS_SUM: {port_weights_sum}")
    print(f"PROOF:ORCH_ATTRIBUTION_OK: {attribution_ok}")
    print(f"PROOF:ORCH_PORT_HASH: {port_det_hash}")
    
    return summary