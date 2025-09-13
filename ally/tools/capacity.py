import json, hashlib
from typing import List
from ally.schemas.capacity import CapacityInput, CapacityResult, CapacityPoint
from ally.schemas.base import ToolResult

def _sha1(o): return hashlib.sha1(json.dumps(o, sort_keys=True).encode()).hexdigest()

def estimate(inp: CapacityInput) -> CapacityResult:
    # Generate deterministic curve: 20 evenly spaced points up to 5x ADV
    curve = []
    for k in range(1,21):
        notional = k * 0.25 * inp.adv_usd  # 0.25x to 5x ADV
        vol_frac = notional / inp.adv_usd
        impact = inp.impact_alpha * (vol_frac ** inp.beta) * max(inp.daily_vol_bps, 1e-9)
        total_bps = inp.commission_bps + impact
        curve.append(CapacityPoint(notional_usd=notional, total_cost_bps=round(total_bps,4)))
    # capacity = first point where total <= decay_target_bps (highest notional under target)
    under = [p.notional_usd for p in curve if p.total_cost_bps <= inp.decay_target_bps]
    cap = max(under) if under else 0.0
    res = {"cap": cap, "curve":[p.__dict__ for p in curve]}
    return CapacityResult(
        capacity_usd=cap,
        impact_decay_bps=inp.decay_target_bps,
        curve=curve,
        det_hash=_sha1(res)
    )

def capacity_curve(**kw) -> ToolResult:
    res = estimate(CapacityInput(**kw))
    return ToolResult(ok=True, data=res.dict())