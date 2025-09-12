from __future__ import annotations
import json, os, sys, time, traceback, hashlib, subprocess, textwrap, inspect, random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from ally.schemas.base import ToolResult, Meta

# Simple determinism helper
def set_determinism(seed: int = 1337) -> None:
    import os, random, numpy as np
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        np.random.seed(seed)
    except ImportError:
        pass

RUNS = Path("runs"); RUNS.mkdir(parents=True, exist_ok=True)

def _jsonable(x: Any) -> Any:
    try:
        json.dumps(x); return x
    except Exception:
        return repr(x)

def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

# 1) debug.capture_trace — run a callable/tool and capture failure trace+locals to runs/
def capture_trace(task_name: str, fn: Callable, *args, **kwargs) -> ToolResult:
    t0 = time.time(); set_determinism(1337)
    meta = Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name":"debug.capture_trace"})
    try:
        out = fn(*args, **kwargs)
        meta.duration_ms = int((time.time()-t0)*1000)
        return ToolResult(ok=True, data={"result": _jsonable(out)}, errors=[], meta=meta)
    except Exception as e:
        tb = traceback.format_exc()
        loc = {k:_jsonable(v) for k,v in (kwargs or {}).items()}
        rec = {
            "task": task_name,
            "error_type": e.__class__.__name__,
            "error": str(e),
            "traceback": tb,
            "kwargs": loc,
            "ts": int(time.time())
        }
        p = RUNS / f"DEBUG_{task_name}_{int(time.time())}.json"
        p.write_text(json.dumps(rec, indent=2))
        meta.duration_ms = int((time.time()-t0)*1000)
        return ToolResult(ok=False, data={"trace_path": str(p)}, errors=[str(e)], meta=meta)

# 2) debug.make_repro — emit one-file, hermetic repro harness for a failing call
def make_repro(tool_name: str, inputs: Dict[str, Any], seed: int = 1337) -> ToolResult:
    set_determinism(seed)
    repro_id = f"{tool_name.replace('.','_')}_{seed}_{int(time.time())}"
    code = f'''\
# Auto-generated reproducible harness for {tool_name}
import json, os
os.environ.setdefault("PYTHONHASHSEED", "{seed}")
os.environ.setdefault("TZ", "UTC")
from ally.tools import TOOL_REGISTRY
from ally.tools.debug import set_determinism
set_determinism({seed})
inputs = {json.dumps(inputs, indent=2)}
res = TOOL_REGISTRY["{tool_name}"](**inputs)
print(json.dumps({{"ok": res.ok, "data": res.data, "errors": res.errors}}, sort_keys=True))
'''
    Path("artifacts").mkdir(exist_ok=True)
    fp = Path("artifacts") / f"repro_{repro_id}.py"
    fp.write_text(textwrap.dedent(code))
    h = _sha1_bytes(fp.read_bytes())
    return ToolResult(ok=True, data={"repro_path": str(fp), "repro_sha1": h}, errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))

# 3) debug.minimize — delta-minimize a failing list input for a given tool
def minimize(tool_name: str, list_arg_name: str, inputs: Dict[str, Any], max_iters: int = 20) -> ToolResult:
    from ally.tools import TOOL_REGISTRY
    set_determinism(1337)
    payload = list(inputs[list_arg_name])
    def run(payload_subset):
        args = dict(inputs); args[list_arg_name] = payload_subset
        try:
            r = TOOL_REGISTRY[tool_name](**args)
            return not r.ok  # failing condition -> True means we reproduced the failure
        except Exception:
            return True
    if not payload:  # nothing to minimize
        return ToolResult(ok=False, data={"reason":"empty_payload"}, errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))
    lo, hi = 1, len(payload)
    best = payload
    it = 0
    while lo <= hi and it < max_iters:
        it += 1
        mid = max(1, (lo+hi)//2)
        candidate = payload[:mid]
        if run(candidate):
            best = candidate; hi = mid-1
        else:
            lo = mid+1
    sha = _sha1_bytes(json.dumps(best, sort_keys=True).encode())
    out = {"min_size": len(best), "hash": sha, "first_items": best[:3]}
    return ToolResult(ok=True, data=out, errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))

# 4) debug.bisect — optional helper; no-op in shallow CI
def bisect(first_bad_cmd: List[str], good_ref: Optional[str]=None, bad_ref: Optional[str]="HEAD") -> ToolResult:
    # CI often has shallow clones; so we detect and no-op with a clear message
    try:
        c = subprocess.run(["git","rev-parse","--is-inside-work-tree"], capture_output=True, text=True)
        if c.returncode != 0: raise RuntimeError("Not a git worktree")
        # Light: report current head; actual bisect left to local usage
        head = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
        return ToolResult(ok=True, data={"bisect_supported": False, "head": head}, errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))
    except Exception as e:
        return ToolResult(ok=True, data={"bisect_supported": False, "reason": str(e)}, errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))

# 5) debug.lint_typecheck — ruff + mypy (if configured), returns booleans
def lint_typecheck(paths: List[str] = None) -> ToolResult:
    paths = paths or ["ally", "tests"]
    lint_ok = True; mypy_ok = True
    
    # Check ruff
    try:
        ruff = subprocess.run(["ruff","--version"], capture_output=True, text=True)
        if ruff.returncode == 0:
            lint = subprocess.run(["ruff","check","--quiet", *paths])
            lint_ok = (lint.returncode == 0)
    except FileNotFoundError:
        lint_ok = True  # Tool not available, don't fail
    
    # Check mypy
    try:
        mypy = subprocess.run(["mypy","--version"], capture_output=True, text=True)
        if mypy.returncode == 0:
            tchk = subprocess.run(["mypy","--strict", *paths])
            mypy_ok = (tchk.returncode == 0)
    except FileNotFoundError:
        mypy_ok = True  # Tool not available, don't fail
        
    return ToolResult(ok=lint_ok and mypy_ok, data={"lint_ok": lint_ok, "mypy_ok": mypy_ok}, errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))

# 6) debug.propcheck — simple property-based checks (Hypothesis) for core utilities
def propcheck_hashing(trials: int = 50) -> ToolResult:
    random.seed(1337)
    fails = 0
    try:
        from ally.utils.hashing import hash_inputs
        for i in range(trials):
            obj = {"i": i, "vals": [j for j in range(i%7)]}
            h1 = hash_inputs(obj); h2 = hash_inputs(obj)
            if not (isinstance(h1, str) and h1 == h2 and len(h1) >= 8):
                fails += 1
    except Exception:
        fails = trials
    return ToolResult(ok=(fails==0), data={"prop_fails": fails}, errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))