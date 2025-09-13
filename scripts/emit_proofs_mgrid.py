from ally.tools.grid import run
from pathlib import Path
import json
ART = Path("mgrid-proof-bundle"); ART.mkdir(parents=True, exist_ok=True)
param_grid = [
  {"lookback":50,"thresh":1.2},
  {"lookback":50,"thresh":1.2},  # duplicate → dedup hit
  {"lookback":100,"thresh":0.8},
  {"lookback":150,"thresh":1.1},
]
out = run(param_grid, max_workers=4)
proofs = {
  "GRID_JOBS": out.data["submitted"],
  "DEDUP_HITS": out.data["dedup_hits"],
  "RESUMED": out.data["resumed"]
}
print(f"PROOF:GRID_JOBS:{proofs['GRID_JOBS']}")
print(f"PROOF:DEDUP_HITS:{proofs['DEDUP_HITS']}")
print(f"PROOF:RESUMED:{proofs['RESUMED']}")
(ART/"mgrid_proofs.json").write_text(json.dumps(proofs, indent=2))