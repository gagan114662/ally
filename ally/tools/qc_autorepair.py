"""
QC Auto-Repair: Automatic fixing and re-running of QC algorithms
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime
from pathlib import Path
from ally.schemas.base import ToolResult, Meta
from ally.tools.qc_lint import qc_lint
from ally.tools.qc_lean import qc_smoke_run
from ally.tools.qc_runtime_guard import qc_classify_error
from ally.qc import qc_fixers as F


# Map fix IDs to fixer functions
FIXER_FNS = {
    "add_algorithmimports": F.add_algorithmimports,
    "fix_ondata_signature": F.fix_ondata_signature,
    "replace_now_with_self_time": F.replace_now_with_self_time,
    "replace_transactions_orders": F.replace_transactions_orders,
    "schedule_everyday": F.schedule_everyday,
    "normalize_bnb_to_usdt_binance": F.normalize_bnb_to_usdt_binance,
}


def _sha1(p: Path) -> str:
    """Calculate SHA1 hash of file"""
    return hashlib.sha1(p.read_bytes()).hexdigest()


def qc_autorepair(
    algo_path: str,
    max_rounds: int = 3,
    minutes_per_round: int = 2
) -> ToolResult:
    """
    Auto-repair QC algorithm: lint → smoke → classify errors → apply fixes → repeat
    
    Args:
        algo_path: Path to QC algorithm file
        max_rounds: Maximum repair attempts
        minutes_per_round: Time limit per smoke test
        
    Returns:
        ToolResult with repair results and proof data
    """
    try:
        p = Path(algo_path)
        if not p.exists():
            raise FileNotFoundError(f"Algorithm file not found: {algo_path}")
            
        before = _sha1(p)
        fixes_applied = []
        
        # Round 0: lint + autofix
        lint_result = qc_lint(algo_path, autofix=True)
        if not lint_result.ok:
            return ToolResult(
                ok=False,
                data={"error": "Initial lint failed", "lint_errors": lint_result.errors},
                errors=lint_result.errors,
                meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.autorepair"})
            )
        
        # Repair attempts
        for attempt in range(1, max_rounds + 1):
            run = qc_smoke_run(algo_path, max_minutes=minutes_per_round)
            
            if run.ok:
                # Success! Return proof data
                return ToolResult(
                    ok=True,
                    data={
                        "attempts": attempt,
                        "fixes_applied": fixes_applied,
                        "result_hash": run.data.get("result_hash", ""),
                        "algo_sha1_before": before,
                        "algo_sha1_after": _sha1(p),
                        "smoke_result": run.data
                    },
                    errors=[],
                    meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.autorepair"})
                )
            
            # Extract error information for classification
            stderr = ""
            if run.errors:
                stderr = " ".join(run.errors)
            elif run.data and "stderr_preview" in run.data:
                stderr = run.data["stderr_preview"]
            
            # Classify errors and get suggested fixes
            cls = qc_classify_error(stderr)
            if not cls.data.get("fixes"):
                # No fixable errors found, give up
                break
            
            # Apply all suggested fixes
            txt = p.read_text()
            mutated = False
            
            for fix_id in cls.data["fixes"]:
                fn = FIXER_FNS.get(fix_id)
                if fn:
                    new_txt = fn(txt)
                    if new_txt != txt:
                        mutated = True
                        txt = new_txt
                        if fix_id not in fixes_applied:
                            fixes_applied.append(fix_id)
            
            if mutated:
                p.write_text(txt)
                # Re-lint after applying fixes
                qc_lint(algo_path, autofix=True)
            else:
                # No fixes could be applied
                break
        
        # All attempts exhausted
        return ToolResult(
            ok=False,
            data={
                "attempts": attempt,
                "fixes_applied": fixes_applied,
                "algo_sha1_before": before,
                "algo_sha1_after": _sha1(p)
            },
            errors=["QC Auto-Repair could not achieve a clean smoke run"],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.autorepair"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.autorepair"})
        )