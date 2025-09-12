from __future__ import annotations
import numpy as np

def sharpe(returns, rf=0.0):
    r = np.asarray(returns)
    if r.size == 0 or np.std(r) == 0: return 0.0
    return (np.mean(r) - rf) / np.std(r)

def deflated_sharpe(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    # Loosely following Bailey et al.: penalize SR for sample size & non-normality
    if n <= 1: return 0.0
    # small-sample adjustment (approx)
    adj = np.sqrt((n - 1) / (n - 2))
    sr_adj = sr * adj
    # simple skew/kurt penalty
    penalty = 0.5 * (abs(skew) + max(0.0, kurt - 3.0)) / np.sqrt(n)
    return max(0.0, sr_adj - penalty)

def reality_check_pvalue(oos_returns, n_boot=200, seed=42) -> float:
    # Stationary bootstrap-ish simple p-value: compare SR to shuffled SRs
    rng = np.random.default_rng(seed)
    obs = sharpe(oos_returns)
    sims = []
    arr = np.asarray(oos_returns)
    for _ in range(n_boot):
        perm = rng.permutation(arr)
        sims.append(sharpe(perm))
    sims = np.asarray(sims)
    # one-sided: how often sim >= observed
    return float((sims >= obs).mean())