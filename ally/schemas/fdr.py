from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class FDRConfig(BaseModel):
    alpha: float = 0.05
    method: str = "BH"                 # "BH" (Benjamini–Hochberg) or "BY"
    group_key: Optional[str] = None    # stratified FDR by group (e.g., asset class)
    require_positive_alpha: bool = True
    min_oos_obs: int = 60              # minimum OOS observations to consider

class Candidate(BaseModel):
    id: str
    t_oos: float
    oos_obs: int
    alpha_oos: float                   # mean OOS residual alpha (bps or pct, doc it)
    meta: Dict[str, str] = {}

class FDRResult(BaseModel):
    n_tested: int
    n_promoted: int
    promoted_ids: List[str]
    q_values: Dict[str, float] = Field(default_factory=dict)
    method: str
    alpha: float
    groups: List[str] = Field(default_factory=list)
    mean_t_promoted: float = 0.0
    pos_alpha_enforced: bool = True
    det_hash: str