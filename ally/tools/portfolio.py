from __future__ import annotations
import numpy as np, hashlib, json
from typing import Dict, List
from ally.schemas.base import ToolResult, Meta
from ally.schemas.portfolio import AllocateIn, AllocateOut, AttributionOut, AttributionRow
from ally.utils.portfolio_math import vol_target, risk_parity, hrp

def allocate(**kwargs) -> ToolResult:
    inp = AllocateIn(**kwargs)
    if inp.method == "vol_target":
        w = vol_target(inp.returns, target_vol=inp.target_vol,
                       bounds=(inp.min_w, inp.max_w), long_only=inp.long_only)
    elif inp.method == "risk_parity":
        w = risk_parity(inp.returns)
    else:
        w = hrp(inp.returns)
    # realized vol from portfolio weights
    syms = sorted(inp.returns.keys())
    X = np.column_stack([np.asarray(inp.returns[s], float) for s in syms])
    C = np.cov(X, rowvar=False)
    wv = np.array([w[s] for s in syms])
    realized = float(np.sqrt(wv @ C @ wv) * np.sqrt(252.0))
    out = AllocateOut(weights=w, method=inp.method, target_vol=inp.target_vol,
                      realized_vol=realized, ok=abs(sum(w.values()) - 1.0) < 1e-6)
    return ToolResult(ok=out.ok, data=out.model_dump(), errors=[], meta=Meta(ts=None, duration_ms=0))

def attribution(prices: Dict[str, List[float]], weights: Dict[str, float], dates: List[str]) -> ToolResult:
    # simple log returns; contribution = w_{t-1} * r_t
    syms = sorted(prices.keys())
    P = {s: np.asarray(prices[s], float) for s in syms}
    R = {s: np.diff(np.log(P[s])) for s in syms}
    rows = []; sum_port = 0.0; sum_contrib = 0.0
    # constant weights across period (MVP); rolling weights can come later
    for t in range(len(next(iter(R.values())))):
        contrib = {s: float(weights[s] * R[s][t]) for s in syms}
        port = float(sum(contrib.values()))
        rows.append(AttributionRow(date=dates[t+1], portfolio_ret=port, contrib=contrib))
        sum_port += port; sum_contrib += port
    out = AttributionOut(rows=rows, sum_contrib=sum_contrib, sum_portfolio=sum_port,
                         ok=abs(sum_contrib - sum_port) < 1e-9)
    return ToolResult(ok=out.ok, data=out.model_dump(), errors=[], meta=Meta(ts=None, duration_ms=0))