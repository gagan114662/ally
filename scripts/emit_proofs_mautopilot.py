from ally.tools.autopilot import run
from pathlib import Path
import json
ART = Path("mautopilot-proof-bundle"); ART.mkdir(parents=True, exist_ok=True)
res = run("implement add", max_rounds=2)
proofs = {"AUTO_PR":"ok","SELF_REPAIR_ROUNDS":res.data["rounds"],"GREEN_CI":res.ok,"DET_HASH":res.data["det_hash"]}
print(f"PROOF:AUTO_PR:{proofs['AUTO_PR']}")
print(f"PROOF:SELF_REPAIR_ROUNDS:{proofs['SELF_REPAIR_ROUNDS']}")
print(f"PROOF:GREEN_CI:{proofs['GREEN_CI']}")
(ART/"mautopilot_proofs.json").write_text(json.dumps(proofs, indent=2))