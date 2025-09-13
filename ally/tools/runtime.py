from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from ally.schemas.base import ToolResult, Meta
from ally.schemas.cache import CacheConfig, CacheEntry
from ally.tools import TOOL_REGISTRY
from ally.utils.cache import make_cache_key, get_from_cache, put_in_cache
from ally.runtime.ollama import has_ollama, generate_ollama
from ally.runtime.fixtures import generate_fixture
from . import register

# engine registry reused (optional)
_REG = Path("ally/config/engine_registry.json")

def _decide_engine(task: str) -> str:
    # Ask M-Router for the matrix deterministically
    r = TOOL_REGISTRY["router.build_matrix"]()
    return r.data["matrix"][task]

@register("runtime.generate")
def runtime_generate(task: str, prompt: str, system: Optional[str] = None,
                     params: Optional[Dict[str,Any]] = None,
                     engine: Optional[str] = None,
                     cache: Optional[Dict[str,Any]] = None,
                     live: bool = False) -> ToolResult:
    params = params or {}
    cfg = CacheConfig(**(cache or {}))

    eng = engine or _decide_engine(task)
    key = make_cache_key(eng, task, prompt, system, params)

    # 1) cache check
    if cfg.enabled:
        hit = get_from_cache(cfg.dir, key)
        if hit:
            print("PROOF:CACHE_HIT: 1")
            print(f"PROOF:CACHE_KEY_HASH: {key}")
            return ToolResult(ok=True, data={"engine": eng, "mode": hit.get("mode","fixture"), "output": hit["output"]},
                              errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))

    # 2) runtime path (only if explicitly allowed)
    out = None; mode = "fixture"
    if live and has_ollama():
        try:
            # Map engine id -> local model name (engine registry optional)
            model = None
            try:
                reg = json.loads(_REG.read_text())
                model = reg.get("engines",{}).get(eng,{}).get("model")
            except: pass
            model = model or eng  # fallback

            out = generate_ollama(model=model, prompt=prompt, system=system, params=params)
            mode = "ollama"
        except Exception:
            out = None

    # 3) fixture fallback
    if out is None:
        out = generate_fixture(engine=eng, task=task, prompt=prompt)
        mode = "fixture"

    record = CacheEntry(key=key, engine=eng, task=task, prompt=prompt, system=system, params=params, output=out).model_dump()
    if cfg.enabled:
        put_in_cache(cfg.dir, key, {**record, "mode": mode})

    print("PROOF:CACHE_HIT: 0")
    print(f"PROOF:CACHE_KEY_HASH: {key}")
    print(f"PROOF:RUNTIME_MODE: {mode}")

    return ToolResult(ok=True, data={"engine": eng, "mode": mode, "output": out}, errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))

@register("cache.clear")
def cache_clear(dir: str = "runs/cache") -> ToolResult:
    from ally.utils.cache import clear_cache
    n = clear_cache(dir)
    return ToolResult(ok=True, data={"cleared": n}, errors=[], meta=Meta(ts=datetime.utcnow(), duration_ms=0))