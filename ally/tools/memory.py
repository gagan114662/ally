"""
Memory tools for Ally - logging and querying experimental runs
"""

from ..tools import register
from ..schemas.base import ToolResult
from ..schemas.memory import LogRunIn, QueryIn, QueryOut
from ..utils.db import get_db_manager


@register("memory.log_run")
def memory_log_run(**kwargs) -> ToolResult:
    """
    Log a completed experimental run to persistent memory
    
    Stores run metadata, metrics, events, and trades in DuckDB for later analysis
    and reporting. This creates a permanent record of experimental results.
    """
    try:
        inputs = LogRunIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        # Get database manager and log the run
        db = get_db_manager()
        
        success = db.log_run(
            run_id=inputs.run_id,
            task=inputs.task,
            code_hash=inputs.code_hash,
            inputs_hash=inputs.inputs_hash,
            ts=inputs.ts,
            metrics=inputs.metrics,
            events=inputs.events,
            trades=inputs.trades,
            notes=inputs.notes
        )
        
        if not success:
            return ToolResult.error(["Failed to log run to database"])
        
        return ToolResult.success({
            "run_id": inputs.run_id,
            "logged": True,
            "metrics_count": len(inputs.metrics),
            "events_count": len(inputs.events),
            "trades_count": len(inputs.trades)
        })
        
    except Exception as e:
        return ToolResult.error([f"Memory logging failed: {e}"])


@register("memory.query")
def memory_query(**kwargs) -> ToolResult:
    """
    Query experimental runs from persistent memory
    
    Supports SQL queries or simple table queries to retrieve stored run data
    for analysis, comparison, and reporting purposes.
    """
    try:
        inputs = QueryIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        # Get database manager and execute query
        db = get_db_manager()
        
        result = db.query_runs(
            query=inputs.query,
            table=inputs.table,
            params=inputs.params,
            limit=inputs.limit
        )
        
        # Create output with proper schema
        query_out = QueryOut(
            rows=result["rows"],
            count=result["count"],
            columns=result["columns"],
            execution_time_ms=result.get("execution_time_ms")
        )
        
        return ToolResult.success(query_out.model_dump())
        
    except Exception as e:
        return ToolResult.error([f"Memory query failed: {e}"])