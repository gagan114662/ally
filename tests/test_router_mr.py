import json, hashlib
from ally.tools import TOOL_REGISTRY

def test_router_matrix_and_proofs():
    r = TOOL_REGISTRY["router.build_matrix"]()
    assert r.ok
    mx = r.data["matrix"]
    assert set(mx.keys()) == {"codegen","nlp","math","cv"}
    # Fallback ok & eval hash present
    assert r.data["fallback_ok"] is True
    assert isinstance(r.data["eval_det_hash"], str) and len(r.data["eval_det_hash"]) >= 8

    # Deterministic proof hash over matrix + eval hash
    h = hashlib.sha1(json.dumps({"mx":mx, "h":r.data["eval_det_hash"]}, sort_keys=True).encode()).hexdigest()

    print("PROOF:ROUTER_MATRIX:", json.dumps(mx, sort_keys=True))
    print("PROOF:ROUTER_FALLBACK:", "ok" if r.data["fallback_ok"] else "fail")
    print("PROOF:EVAL_DET_HASH:", r.data["eval_det_hash"])
    print("PROOF:ROUTER_DET:", h)

    # Basic invariants
    assert all(isinstance(v, str) and v for v in mx.values())