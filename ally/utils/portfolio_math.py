from __future__ import annotations
import math, numpy as np
from typing import Dict, List

def _ann_vol(series: np.ndarray) -> float:
    # daily → annualized
    return float(np.std(series, ddof=1) * math.sqrt(252.0))

def _cov(returns: Dict[str, List[float]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    syms = sorted(returns.keys())
    X = np.column_stack([np.asarray(returns[s], dtype=float) for s in syms])
    mu = np.nanmean(X, axis=0)
    C = np.cov(X, rowvar=False)
    return syms, mu, C

def vol_target(returns: Dict[str, List[float]], target_vol: float,
               bounds=(0.0, 0.60), long_only=True) -> dict[str, float]:
    syms, mu, C = _cov(returns)
    diag = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    inv_vol = 1.0 / diag
    w = inv_vol / inv_vol.sum()     # inverse-vol weights
    # bounds/long-only
    lo, hi = bounds
    w = np.clip(w, lo, hi)
    if long_only: w = np.clip(w, 0, None)
    w = w / w.sum()
    port_vol = float(np.sqrt(w @ C @ w) * math.sqrt(252.0))
    scale = (target_vol / port_vol) if port_vol > 1e-12 else 1.0
    weights = dict(zip(syms, (w * scale) / (w * scale).sum()))
    return weights

def risk_parity(returns: Dict[str, List[float]], iters=1000) -> dict[str, float]:
    # simple iterative RP (equal risk contribution) — deterministic init
    syms, _, C = _cov(returns)
    n = len(syms)
    w = np.full(n, 1.0 / n)
    for _ in range(iters):
        rc = w * (C @ w)             # risk contributions
        targ = np.full(n, (w @ C @ w) / n)
        grad = rc - targ
        w = np.clip(w - 0.01 * grad, 1e-6, 1.0)
        w = w / w.sum()
    return dict(zip(syms, w))

def hrp(returns: Dict[str, List[float]]) -> dict[str, float]:
    # minimal, deterministic HRP (distance by correlation, recursive bisection)
    syms, _, C = _cov(returns)
    D = np.diag(1.0 / np.sqrt(np.diag(C)))
    corr = D @ C @ D
    dist = np.sqrt(0.5 * (1 - corr))
    order = np.argsort(np.mean(dist, axis=1))  # simple seriation
    w = np.ones(len(syms))
    # recursive split
    def _bisect(idx):
        if len(idx) <= 1: return
        mid = len(idx)//2
        A, B = idx[:mid], idx[mid:]
        Ca = C[np.ix_(A, A)]; Cb = C[np.ix_(B, B)]
        varA = np.mean(np.diag(Ca)); varB = np.mean(np.diag(Cb))
        alpha = 1.0 - varA/(varA+varB)
        w[A] *= alpha; w[B] *= (1.0 - alpha)
        _bisect(A); _bisect(B)
    idx = order.tolist()
    _bisect(idx)
    w = w / w.sum()
    return dict(zip([syms[i] for i in range(len(syms))], w))