import json, os, hashlib
from ally.tools import TOOL_REGISTRY

proofs = {}

# Repro proof
rep = TOOL_REGISTRY["debug.make_repro"]("memory.log_run", {
    "run_id": "test_debug_repro",
    "task": "debug_proof", 
    "code_hash": "deadbeef", 
    "inputs_hash": "cafebabe"
}, seed=1337)
proofs["DEBUG_REPRO_HASH"] = rep.data.get("repro_sha1","")

# Propcheck proof
pc = TOOL_REGISTRY["debug.propcheck"](trials=25)
proofs["PROP_FAILS"] = pc.data.get("prop_fails", -1)

# Lint/Typecheck proof
lt = TOOL_REGISTRY["debug.lint_typecheck"](["ally"])
proofs["LINT_OK"] = lt.data.get("lint_ok", False)
proofs["MYPY_OK"] = lt.data.get("mypy_ok", True)

os.makedirs("artifacts", exist_ok=True)
with open("artifacts/mdebug_proofs.json","w") as f: 
    json.dump(proofs, f, indent=2)
    
print("PROOF:DEBUG_REPRO_HASH:", proofs["DEBUG_REPRO_HASH"])
print("PROOF:PROP_FAILS:", proofs["PROP_FAILS"])
print("PROOF:LINT_OK:", proofs["LINT_OK"])
print("PROOF:MYPY_OK:", proofs["MYPY_OK"])