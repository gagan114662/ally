import json
from pathlib import Path
from ally.tools.capacity import capacity_curve

out = capacity_curve(symbol="SPY", adv_usd=5_000_000_000, daily_vol_bps=150.0,
                     impact_alpha=0.6, beta=0.5, commission_bps=1.0, decay_target_bps=25.0).data
print(f"PROOF:CAPACITY_USD:{int(out['capacity_usd'])}")
print(f"PROOF:IMPACT_DECAY_BPS:{out['impact_decay_bps']}")
print(f"PROOF:CURVE_HASH:{out['det_hash']}")

Path("capacity-proof-bundle").mkdir(exist_ok=True)
with open("capacity-proof-bundle/capacity.json","w") as f:
    json.dump(out, f, sort_keys=True)