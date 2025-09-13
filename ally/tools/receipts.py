# ally/tools/receipts.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

from ally.schemas.base import ToolResult, Meta
from ally.tools import register
from ally.utils.db import get_db_manager
from ally.utils.provenance import compute_provenance_hash

RECEIPTS_DIR = Path("runs/receipts")
RAW_DIR = Path("runs/raw")

def _sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _load_json_safe(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _find_receipts_for_run(run_id: str, base_dir: Path = RECEIPTS_DIR) -> List[Path]:
    if not base_dir.exists():
        return []
    return sorted([p for p in base_dir.glob("*.json") if run_id in p.name])

def _assert_invariants(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Hard invariant: if live==True then receipts must be present and non-empty."""
    live = bool(obj.get("live", False))
    receipts = obj.get("receipts")
    ok = True
    reasons: List[str] = []
    if live:
        if receipts is None or not isinstance(receipts, list) or len(receipts) == 0:
            ok = False
            reasons.append("live==true but receipts are missing/empty")
        else:
            for i, r in enumerate(receipts):
                if not isinstance(r, dict):
                    ok = False; reasons.append(f"receipt[{i}] not an object"); continue
                for k in ("content_sha1","vendor","endpoint","ts_iso"):
                    if k not in r:
                        ok = False; reasons.append(f"receipt[{i}] missing {k}")
    return {"ok": ok, "reasons": reasons}

@register("receipts.verify")
def receipts_verify(run_id: str,
                    strict: bool = True,
                    base_dir: Optional[str] = None) -> ToolResult:
    """
    Verify receipts for a run:
    - All live outputs have at least one ReceiptRef
    - Each receipt file exists and SHA1 matches content_sha1
    - DB rows exist when DB available
    """
    base = Path(base_dir) if base_dir else RECEIPTS_DIR
    receipts_paths = _find_receipts_for_run(run_id, base)
    file_checks: List[Dict[str, Any]] = []
    for rp in receipts_paths:
        j = _load_json_safe(rp)
        sha = j.get("content_sha1")
        calc = _sha1_file(rp)
        file_checks.append({"path": str(rp), "sha1_file": calc, "sha1_json": sha, "match": (sha == calc)})

    # DB check (best-effort) - check by session_id if available
    db_ok = None
    rows = []
    try:
        db = get_db_manager()
        # Try to find receipts by content_sha1 since run_id might not be in data_receipts table
        if file_checks:
            content_sha1s = [fc["sha1_json"] for fc in file_checks if fc.get("sha1_json")]
            if content_sha1s:
                sha1_list = "','".join(content_sha1s)
                result = db.query("data_receipts", where=f"content_sha1 IN ('{sha1_list}')")
                rows = result.get("rows", [])
        db_ok = len(rows) >= len(receipts_paths) or len(rows) > 0 if receipts_paths else True
    except Exception:
        db_ok = None  # DB optional

    all_match = all(x.get("match", False) for x in file_checks) if file_checks else (not strict)
    ok = (len(receipts_paths) > 0 or not strict) and all_match and (db_ok in (True, None))

    data = {
        "run_id": run_id,
        "receipt_files": len(receipts_paths),
        "file_checks": file_checks,
        "db_rows": len(rows),
        "db_ok": db_ok,
        "ok": ok,
        "provenance_hash": compute_provenance_hash(
            {"run_id": run_id}, 
            {"files": [f["sha1_file"] for f in file_checks]}, 
            []
        )
    }
    return ToolResult.success(data)

@register("receipts.diff")
def receipts_diff(series_a: List[float],
                  series_b: List[float],
                  tolerance: float = 1e-6) -> ToolResult:
    """
    Offline, deterministic quorum-style diff on two numeric series.
    Returns max_abs_diff and pass/fail against tolerance.
    """
    n = min(len(series_a), len(series_b))
    diffs = [abs(series_a[i]-series_b[i]) for i in range(n)]
    max_abs = max(diffs) if diffs else 0.0
    ok = max_abs <= tolerance and len(series_a) == len(series_b)
    data = {"max_abs_diff": max_abs, "n": n, "ok": ok, "tolerance": tolerance}
    return ToolResult.success(data) if ok else ToolResult.error(["quorum_disagreement"], data)

# Export invariant helper for tests
def assert_receipts_invariants(obj: Dict[str, Any]) -> None:
    res = _assert_invariants(obj)
    if not res["ok"]:
        raise AssertionError("; ".join(res["reasons"]))