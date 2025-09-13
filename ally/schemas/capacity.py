from pydantic import BaseModel
from typing import List, Dict

class CapacityInput(BaseModel):
    symbol: str
    adv_usd: float
    daily_vol_bps: float
    impact_alpha: float = 0.6
    beta: float = 0.5  # square-root
    commission_bps: float = 1.0
    decay_target_bps: float = 25.0

class CapacityPoint(BaseModel):
    notional_usd: float
    total_cost_bps: float

class CapacityResult(BaseModel):
    capacity_usd: float
    impact_decay_bps: float
    curve: List[CapacityPoint]
    det_hash: str