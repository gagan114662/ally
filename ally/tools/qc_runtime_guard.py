"""
QC runtime error classification for auto-repair
"""

from __future__ import annotations
import re
import yaml
from datetime import datetime
from pathlib import Path
from ally.schemas.base import ToolResult, Meta


def _load_error_db() -> dict:
    """Load error patterns database"""
    db_path = Path(__file__).parent.parent / "qc" / "qc_error_db.yaml"
    try:
        return yaml.safe_load(db_path.read_text())
    except FileNotFoundError:
        return {"patterns": []}


def qc_classify_error(stderr_text: str) -> ToolResult:
    """
    Classify QC engine errors and suggest fixes
    
    Args:
        stderr_text: Error output from LEAN engine
        
    Returns:
        ToolResult with classified fixes
    """
    try:
        db = _load_error_db()
        fixes = []
        
        for pattern in db.get("patterns", []):
            if re.search(pattern["regex"], stderr_text, re.I | re.M):
                fixes.append(pattern["fix"])
        
        return ToolResult(
            ok=bool(fixes),
            data={
                "fixes": fixes,
                "count": len(fixes),
                "error_text": stderr_text[:500]  # Truncated for logging
            },
            errors=[],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.classify_error"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.classify_error"})
        )