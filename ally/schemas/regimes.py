from pydantic import BaseModel
from typing import Dict, List

class RegimeInput(BaseModel):
    dates: List[str]           # ISO-Z or YYYY-MM-DD
    realized_vol_bps: List[float]
    illiq_score_bps: List[float]

class RegimeResult(BaseModel):
    labels: List[str]          # per-date regime labels
    res_alpha_t_per_regime: Dict[str, float]
    stable_ok: bool
    det_hash: str