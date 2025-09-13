from __future__ import annotations
from datetime import datetime
from ally.schemas.base import ToolResult, Meta
from ally.orchestrator.run import orchestrator_run
from . import register

@register("orchestrator.run")
def orchestrate(**kwargs) -> ToolResult:
    """
    Run orchestrated multi-asset backtest with portfolio integration
    
    Args:
        symbols: List of symbols for portfolio
        start_date: Backtest start date  
        end_date: Backtest end date
        
    Returns:
        ToolResult with orchestrator summary
    """
    try:
        summary = orchestrator_run(**kwargs)
        
        return ToolResult(
            ok=summary.execution_success,
            data=summary.model_dump(),
            errors=[],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0)
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0)
        )