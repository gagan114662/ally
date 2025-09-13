import json
from pathlib import Path
from ally.tools.regimes import regime_gate

fx = json.load(open("data/fixtures/regimes/small.json"))
out = regime_gate(**fx).data
print(f"PROOF:RES_ALPHA_T_PER_REGIME:{json.dumps(out['res_alpha_t_per_regime'], sort_keys=True)}")
print(f"PROOF:REGIME_STABILITY:{str(out['stable_ok']).lower()}")
print(f"PROOF:REGIME_HASH:{out['det_hash']}")

Path("regimes-proof-bundle").mkdir(exist_ok=True)
with open("regimes-proof-bundle/regimes.json","w") as f:
    json.dump(out, f, sort_keys=True)