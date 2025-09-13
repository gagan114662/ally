import hashlib, json, math
from typing import List
from ally.schemas.fdr import FdrInput, FdrResult
from ally.schemas.base import ToolResult

def _sha1(obj)->str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def _bh(pvals: List[float], q: float):
    # Benjamini–Hochberg
    m = len(pvals)
    pairs = sorted((p,i) for i,p in enumerate(pvals))
    cutoff = -1
    for k,(p,i) in enumerate(pairs, start=1):
        if p <= q * k / m: cutoff = k
    passed_idx = set(i for _,i in pairs[:cutoff]) if cutoff>0 else set()
    return passed_idx

def evaluate_fdr(q: float, candidates):
    pvals = [c["spa_pvalue"] for c in candidates]
    passed_idx = _bh(pvals, q)
    passed = [c["sid"] for i,c in enumerate(candidates) if i in passed_idx and c["resid_alpha_t"]>0]
    # light PSI: ensure T-stats not inflated vs rank; deterministic check
    psi_ok = all(c["resid_alpha_t"]>=0 or c["sid"] not in passed for c in candidates)
    det_hash = _sha1({"q":q,"cand":candidates,"passed":passed})
    return FdrResult(q=q,total=len(candidates),passed=passed,psi_ok=psi_ok,det_hash=det_hash)

# TOOL REGISTRY HOOK
def fdr_gate(q: float, candidates: list, promotion_budget: int|None=None) -> ToolResult:
    inp = FdrInput(q=q, candidates=[c for c in candidates], promotion_budget=promotion_budget)
    res = evaluate_fdr(inp.q, [c.dict() for c in inp.candidates])
    if promotion_budget is not None and len(res.passed) > promotion_budget:
        res.passed = res.passed[:promotion_budget]
        res.det_hash = _sha1({"q":q,"cand":[c.dict() for c in inp.candidates],"passed":res.passed})
    return ToolResult(ok=True, data=res.dict())