from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional

Method = Literal["vol_target", "risk_parity", "hrp"]

class AllocateIn(BaseModel):
    returns: Dict[str, List[float]]  # per-symbol daily returns
    method: Method = "vol_target"
    target_vol: float = 0.10         # 10% annualized
    min_w: float = 0.0
    max_w: float = 0.60
    long_only: bool = True

class AllocateOut(BaseModel):
    weights: Dict[str, float]
    method: Method
    target_vol: float
    realized_vol: float
    ok: bool

class AttributionRow(BaseModel):
    date: str
    portfolio_ret: float
    contrib: Dict[str, float]  # weight_{t-1} * ret_t per symbol

class AttributionOut(BaseModel):
    rows: List[AttributionRow]
    sum_contrib: float
    sum_portfolio: float
    ok: bool