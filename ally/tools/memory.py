import hashlib
import json
from typing import Dict, List, Any, Optional

from ally.schemas.memory import LogRunIn, QueryIn, QueryOut
from ally.schemas.base import ToolResult
from ally.utils.db import get_db_manager
from . import register


@register("memory.log_run")
def memory_log_run(
    run_id: str,
    task: str, 
    code_hash: str,
    inputs_hash: str,
    ts: str,
    metrics: Dict[str, float] = None,
    events: List[Dict[str, Any]] = None,
    trades: List[Dict[str, Any]] = None,
    notes: Optional[str] = None
) -> ToolResult:
    """Log a run to memory database. Idempotent on same (run_id, code_hash, inputs_hash)."""
    
    # Validate inputs
    input_data = LogRunIn(
        run_id=run_id,
        task=task,
        code_hash=code_hash,
        inputs_hash=inputs_hash,
        ts=ts,
        metrics=metrics or {},
        events=events or [],
        trades=trades or [],
        notes=notes
    )
    
    db = get_db_manager()
    success = db.log_run(
        run_id=input_data.run_id,
        task=input_data.task,
        code_hash=input_data.code_hash,
        inputs_hash=input_data.inputs_hash,
        ts=input_data.ts,
        metrics=input_data.metrics,
        events=input_data.events,
        trades=input_data.trades,
        notes=input_data.notes
    )
    
    if success:
        return ToolResult.success({"run_id": run_id, "logged": True})
    else:
        return ToolResult.error(["Failed to log run to database"])


@register("memory.query")
def memory_query(
    table: str,
    where: Optional[str] = None,
    limit: int = 100
) -> ToolResult:
    """Query memory database with SQL-like filters."""
    
    # Validate inputs
    input_data = QueryIn(
        table=table,
        where=where,
        limit=limit
    )
    
    # Validate table name for security
    allowed_tables = {"runs", "metrics", "events", "trades"}
    if input_data.table not in allowed_tables:
        return ToolResult.error([f"Invalid table '{input_data.table}'. Allowed: {allowed_tables}"])
    
    db = get_db_manager()
    result = db.query(
        table=input_data.table,
        where=input_data.where,
        limit=input_data.limit
    )
    
    output = QueryOut(
        rows=result["rows"],
        count=result["count"]
    )
    
    return ToolResult.success(output.model_dump())