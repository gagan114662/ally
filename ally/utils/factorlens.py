import hashlib, json
import pandas as pd
import numpy as np
from typing import Tuple, Dict

NW_LAGS_DEFAULT = 5

def normalize_ts(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure index is UTC ISO-Z strings at the edges; internal = Timestamp UTC
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
    return df.sort_index()

def pit_align(returns: pd.DataFrame, factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    r = normalize_ts(returns)
    f = normalize_ts(factors)
    # PIT: factor date must be <= return date. Standard is same-day alignment.
    # We enforce INNER JOIN on intersection only; no ffill.
    common = r.index.intersection(f.index)
    return r.loc[common], f.loc[common]

def ols_neweywest(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS_DEFAULT) -> Dict:
    # X includes intercept in first column
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    T, k = X.shape
    # Newey-West variance
    S = np.zeros((k,k))
    # lag 0
    S0 = (X * resid[:,None]).T @ (X * resid[:,None]) / T
    S += S0
    for L in range(1, min(lags, T-1)+1):
        w = 1.0 - L/(lags+1.0)
        Gamma = (X[L:,:] * resid[L:,None]).T @ (X[:-L,:] * resid[:-L,None]) / T
        S += w * (Gamma + Gamma.T)
    var = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.clip(np.diag(var), 1e-18, None))
    tstats = beta / se
    r2 = 1.0 - (resid @ resid) / ( (y - y.mean()) @ (y - y.mean()) + 1e-18 )
    return {"beta": beta, "t": tstats, "r2": float(r2), "resid": resid}

def exposures(returns_df: pd.DataFrame, factors_df: pd.DataFrame, lags: int = NW_LAGS_DEFAULT):
    r, f = pit_align(returns_df, factors_df)
    y = r.iloc[:,0].to_numpy()  # single series (portfolio or asset)
    X = np.column_stack([np.ones(len(f)), f.to_numpy()])  # intercept + factors
    res = ols_neweywest(y, X, lags=lags)
    # return betas excluding intercept
    return res

def rolling_alpha(returns_df: pd.DataFrame, factors_df: pd.DataFrame, window: int = 252, step: int = 21, lags: int = NW_LAGS_DEFAULT):
    r, f = pit_align(returns_df, factors_df)
    dates = r.index
    alphas, tstats, r2s = [], [], []
    for i in range(window, len(dates), step):
        y = r.iloc[i-window:i, 0].to_numpy()
        X = np.column_stack([np.ones(window), f.iloc[i-window:i,:].to_numpy()])
        res = ols_neweywest(y, X, lags=lags)
        # alpha in daily return units → annualize to bps
        alpha_daily = res["beta"][0]
        alpha_bps = float(alpha_daily * 252.0 * 1e4)
        alphas.append(alpha_bps)
        tstats.append(float(res["t"][0]))
        r2s.append(float(res["r2"]))
    # summary = last window metrics
    return {
        "alpha_bps": float(alphas[-1]) if alphas else 0.0,
        "alpha_t": float(tstats[-1]) if tstats else 0.0,
        "r2": float(r2s[-1]) if r2s else 0.0
    }

def det_hash(obj: Dict) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",",":")).encode()
    return hashlib.sha1(s).hexdigest()