import json, hashlib
from pathlib import Path
from ally.tools.fdr import fdr_gate

CANDS = [
  {"sid":"A","spa_pvalue":0.008,"resid_alpha_t":2.6},
  {"sid":"B","spa_pvalue":0.012,"resid_alpha_t":2.2},
  {"sid":"C","spa_pvalue":0.051,"resid_alpha_t":2.1},
  {"sid":"D","spa_pvalue":0.20,"resid_alpha_t":1.0},
  {"sid":"E","spa_pvalue":0.03,"resid_alpha_t":-0.2},
  {"sid":"F","spa_pvalue":0.049,"resid_alpha_t":2.05},
  {"sid":"G","spa_pvalue":0.5,"resid_alpha_t":0.1},
]
Q=0.05
out = fdr_gate(q=Q, candidates=CANDS, promotion_budget=3).data
passed = out["passed"]
print(f"PROOF:FDR_ALPHA:{Q}")
print(f"PROOF:FDR_PASS:{len(passed)}/{out['total']}")
print(f"PROOF:PSI_OK:{str(out['psi_ok']).lower() if isinstance(out['psi_ok'], bool) else out['psi_ok']}")
print(f"PROOF:FDR_DET_HASH:{out['det_hash']}")

Path("fdr-proof-bundle").mkdir(exist_ok=True)
with open("fdr-proof-bundle/fdr_proofs.json","w") as f:
    json.dump({"q":Q,"passed":passed,"det_hash":out["det_hash"]}, f, sort_keys=True)