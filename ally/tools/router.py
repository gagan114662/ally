from __future__ import annotations
import json, hashlib
from typing import Dict, List
from pathlib import Path
from datetime import datetime
from ally.schemas.base import ToolResult, Meta
from ally.schemas.router import RouterConfig, RouterMatrix
from ally.utils.router_eval import score_dataset
from . import register

# Offline engine outputs to keep CI deterministic (no network)
_FIXTURES_DIR = Path("data/fixtures/router")

def _load_eval(eval_path: str) -> Dict:
    return json.loads(Path(eval_path).read_text())

def _load_engine_outputs() -> Dict[str, Dict[str, List[str]]]:
    # engines/<engine_id>.json : {"codegen":[...], "nlp":[...], ...}
    out = {}
    p = _FIXTURES_DIR / "engines"
    for f in p.glob("*.json"):
        out[f.stem] = json.loads(f.read_text())
    return out

def _choose_winners(candidates: Dict[str, List[str]], scores_by_task: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    winners = {}
    for task, cands in candidates.items():
        best = None; best_s = -1.0
        for eng in cands:
            s = scores_by_task.get(task, {}).get(eng, -1.0)
            if s > best_s or (s == best_s and eng < (best or "")):  # deterministic tie-break by id
                best, best_s = eng, s
        winners[task] = best or (cands[0] if cands else "")
    return winners

def _fallback_ok(config: RouterConfig, winners: Dict[str, str]) -> bool:
    # Verify fallback chains are valid and have alternatives
    for task, winner in winners.items():
        chain = config.fallback.get(task, [])
        if not chain:
            # No fallback defined - acceptable
            continue
        # Ensure winner has valid fallback alternatives
        if winner not in chain:
            # Winner not in fallback chain - ensure chain is non-empty
            if len(chain) == 0:
                return False
        else:
            # Winner in chain - ensure other alternatives exist
            if len(chain) < 2:
                return False
    return True

@register("router.build_matrix")
def router_build_matrix(**kwargs) -> ToolResult:
    cfg = RouterConfig(**kwargs)
    eval_set = _load_eval(cfg.eval_path)
    engines = _load_engine_outputs()

    # filter engines to those in candidates list (more efficient)
    all_candidates = set()
    for cands in cfg.candidates.values():
        all_candidates.update(cands)
    engines = {k:v for k,v in engines.items() if k in all_candidates}
    scores_by_task, det_hash = score_dataset(eval_set, engines)
    winners = _choose_winners(cfg.candidates, scores_by_task)
    fb_ok = _fallback_ok(cfg, winners)

    data = RouterMatrix(matrix=winners, scores=scores_by_task, fallback_ok=fb_ok, eval_det_hash=det_hash).model_dump()
    return ToolResult(ok=True, data=data, errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))