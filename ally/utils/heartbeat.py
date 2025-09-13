from pathlib import Path
import json, time
from datetime import datetime, timezone

def write_heartbeat(path: str, run_id: str, status: str = "ok") -> str:
    """
    Write a tiny JSON heartbeat file atomically.
    Returns the absolute path string.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    tmp.replace(p)
    return str(p)

def rotate_heartbeat(path: str, run_id: str, delay_sec: float = 1.0) -> dict:
    """
    Write heartbeat twice; verify mtime increased.
    Deterministic enough for CI with small delay & rounded proofs.
    """
    p = Path(path)
    write_heartbeat(path, run_id, status="start")
    t0 = p.stat().st_mtime
    time.sleep(delay_sec)
    write_heartbeat(path, run_id, status="tick")
    t1 = p.stat().st_mtime
    return {
        "path": str(p),
        "rotated": bool(t1 > t0),
        "delay_sec": round(delay_sec, 3),
    }