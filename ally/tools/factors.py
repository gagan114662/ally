from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timezone
from ..schemas.base import ToolResult, Meta
from ..schemas.factors import FactorSetMeta, ExposureRow, ExposuresOut, ResidualAlphaOut, FactorLensSummary
from ..utils.factorlens import exposures as fn_exposures, rolling_alpha, det_hash, pit_align
from ..utils.determinism import set_global_determinism
from ..utils.serialization import convert_timestamps
from ..tools import register

FACTOR_COLS = ["MKT","SMB","HML","RMW","CMA","MOM"]

def _load_fixture_factors() -> pd.DataFrame:
    df = pd.read_csv("data/fixtures/factors/ff5_mom_daily.csv")
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date")[FACTOR_COLS].astype(float)
    return df

@register("factors.load_ff")
def load_ff(set: str = "FF5+Mom", frequency: str = "D") -> ToolResult:
    # CI: fixtures; Live (optional): use RealData Gate with receipts
    df = _load_fixture_factors()
    meta = Meta(ts=datetime.now(timezone.utc))
    return ToolResult(ok=True, data={"meta": FactorSetMeta().model_dump(), "frame": df.reset_index().to_dict("records")}, errors=[], meta=meta)

@register("factors.exposures")
def compute_exposures(returns: List[Dict[str, Any]], lags: int = 5) -> ToolResult:
    set_global_determinism(1337)
    fac = _load_fixture_factors()
    r = pd.DataFrame(returns).copy()
    r["date"] = pd.to_datetime(r["date"], utc=True)
    r = r.set_index("date")[["ret"]].astype(float)
    res = fn_exposures(r, fac, lags=lags)
    rows = [ExposureRow(factor=f, beta=float(res["beta"][i+1]), tstat=float(res["t"][i+1])).model_dump()
            for i, f in enumerate(FACTOR_COLS)]
    out = ExposuresOut(r2=res["r2"], exposures=rows, lags=lags).model_dump()
    meta = Meta(ts=datetime.now(timezone.utc))
    return ToolResult(ok=True, data=out, errors=[], meta=meta)

@register("factors.residual_alpha")
def compute_residual_alpha(returns: List[Dict[str, Any]], window: int = 252, step: int = 21, lags: int = 5) -> ToolResult:
    set_global_determinism(1337)
    fac = _load_fixture_factors()
    r = pd.DataFrame(returns).copy()
    r["date"] = pd.to_datetime(r["date"], utc=True)
    r = r.set_index("date")[["ret"]].astype(float)
    summ = rolling_alpha(r, fac, window=window, step=step, lags=lags)
    res = ResidualAlphaOut(
        alpha_bps=summ["alpha_bps"],
        alpha_tstat=summ["alpha_t"],
        r2=summ["r2"],
        window_days=window,
        residual_series_path=""  # optional later
    ).model_dump()
    meta_info = {
      "meta": FactorSetMeta().model_dump(),
      "exposures_method": "OLS-NeweyWest",
      "lags": lags
    }
    det = det_hash({"res": res, "meta": meta_info})
    summary = FactorLensSummary(
        meta=FactorSetMeta(),
        exposures=ExposuresOut(r2=0.0, exposures=[], lags=lags),
        residual=ResidualAlphaOut(**res),
        det_hash=det,
        ts_utc=datetime.now(timezone.utc)
    ).model_dump()
    meta = Meta(ts=datetime.now(timezone.utc))
    return ToolResult(ok=True, data=summary, errors=[], meta=meta)