from __future__ import annotations
import json, hashlib, os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

def _canonical(obj: Any) -> str:
    # deterministic JSON (no spaces, sorted keys)
    return json.dumps(obj, sort_keys=True, separators=(",",":"))

def make_cache_key(engine: str, task: str, prompt: str, system: Optional[str], params: Dict[str,Any]) -> str:
    payload = _canonical({"engine":engine,"task":task,"prompt":prompt,"system":system or "", "params":params})
    return hashlib.sha1(payload.encode()).hexdigest()

def cache_path(base_dir: str, key: str) -> Path:
    p = Path(base_dir).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{key}.json"

def get_from_cache(base_dir: str, key: str) -> Optional[Dict[str,Any]]:
    fp = cache_path(base_dir, key)
    if not fp.exists(): return None
    try:
        return json.loads(fp.read_text())
    except Exception:
        return None

def put_in_cache(base_dir: str, key: str, record: Dict[str,Any]) -> None:
    fp = cache_path(base_dir, key)
    record = dict(record)
    record["_ts"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fp.write_text(_canonical(record))

def clear_cache(base_dir: str) -> int:
    p = Path(base_dir)
    if not p.exists(): return 0
    n=0
    for f in p.glob("*.json"):
        try: f.unlink(); n+=1
        except: pass
    return n