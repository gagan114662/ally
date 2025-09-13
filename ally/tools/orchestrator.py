"""Orchestrator tools with M-Receipts-Everywhere provenance tracking."""

import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from ally.schemas.base import ToolResult
from ally.tools import register
from ally.utils.provenance import store_run_with_provenance, link_receipts_to_run
from ally.schemas.report import ReceiptRef, ReportSummary


@register("orchestrator.demo")
def orchestrator_demo(
    symbols: Optional[List[str]] = None,
    use_live_data: bool = False,
    live_quorum: Optional[Dict[str, Any]] = None,
    live_budget_cents: int = 100,
    use_runtime: bool = False,
    runtime_live: bool = False
) -> ToolResult:
    """Demo orchestrator with receipt tracking."""
    try:
        run_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4()) if use_live_data else None
        ts_iso = datetime.utcnow().isoformat() + "Z"
        
        # Record inputs for provenance 
        inputs = {
            "symbols": symbols or ["AAPL", "GOOGL"],
            "use_live_data": use_live_data,
            "live_quorum": live_quorum,
            "live_budget_cents": live_budget_cents,
            "use_runtime": use_runtime,
            "runtime_live": runtime_live
        }
        
        # Mock orchestrator execution
        result_data = {
            "run_id": run_id,
            "status": "demo_completed",
            "symbols_processed": inputs["symbols"],
            "data_mode": "live" if use_live_data else "fixtures",
            "runtime_mode": "enabled" if use_runtime else "disabled"
        }
        
        # Link receipts for provenance tracking
        receipts = []
        if use_live_data and session_id:
            receipts = link_receipts_to_run(run_id, session_id)
        
        # Store run with provenance
        store_run_with_provenance(
            run_id=run_id,
            task="orchestrator.demo",
            ts=ts_iso,
            inputs=inputs,
            outputs=result_data,
            session_id=session_id
        )
        
        # Create report summary with receipt references
        receipt_cost = sum(r.cost_cents or 0 for r in receipts)
        receipt_vendors = list(set(r.vendor for r in receipts))
        
        audit_hash = hashlib.sha256(
            json.dumps({**inputs, **result_data}, sort_keys=True).encode()
        ).hexdigest()
        
        inputs_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()
        
        report = ReportSummary(
            run_id=run_id,
            task="orchestrator.demo",
            ts_iso=ts_iso,
            html_path=f"reports/{run_id}_demo.html",
            receipts=receipts,
            receipt_cost_cents=receipt_cost,
            receipt_vendors=receipt_vendors,
            audit_hash=audit_hash,
            inputs_hash=inputs_hash,
            code_hash=hashlib.sha256(orchestrator_demo.__code__.co_code).hexdigest()
        )
        
        return ToolResult(
            ok=True,
            data={
                **result_data,
                "provenance": {
                    "receipts_linked": len(receipts),
                    "receipt_cost_cents": receipt_cost,
                    "receipt_vendors": receipt_vendors,
                    "audit_hash": audit_hash
                },
                "report_summary": report.model_dump()
            }
        )
        
    except Exception as e:
        return ToolResult(ok=False, data={"error": str(e)})


@register("orchestrator.run")
def orchestrator_run(
    experiment_id: str,
    symbols: Optional[List[str]] = None,
    use_live_data: bool = False,
    live_quorum: Optional[Dict[str, Any]] = None,
    live_budget_cents: int = 100,
    use_runtime: bool = False,
    runtime_live: bool = False
) -> ToolResult:
    """Run orchestrator experiment with receipt tracking."""
    try:
        run_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4()) if use_live_data else None
        ts_iso = datetime.utcnow().isoformat() + "Z"
        
        # Record inputs for provenance
        inputs = {
            "experiment_id": experiment_id,
            "symbols": symbols or ["SPY", "QQQ", "IWM"],
            "use_live_data": use_live_data,
            "live_quorum": live_quorum,
            "live_budget_cents": live_budget_cents,
            "use_runtime": use_runtime,
            "runtime_live": runtime_live
        }
        
        # Mock orchestrator execution with runtime handling
        if use_runtime:
            runtime_status = _maybe_runtime(
                experiment_id=experiment_id,
                symbols=inputs["symbols"],
                live=runtime_live
            )
        else:
            runtime_status = {"runtime_used": False, "reason": "use_runtime=False"}
        
        # Generate synthetic portfolio returns for factor analysis
        import pandas as pd
        import numpy as np
        from ally.tools.factors import compute_residual_alpha

        # Create mock portfolio returns for factor lens analysis
        np.random.seed(hash(run_id) % 2**32)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        synthetic_returns = []
        for i, date in enumerate(dates):
            # Mock portfolio return with some alpha
            ret = 0.0003 + 0.0001 * np.sin(i * 0.05) + np.random.normal(0, 0.008)
            synthetic_returns.append({
                "date": date.strftime("%Y-%m-%d"),
                "ret": ret
            })

        # Compute factor lens metrics
        factorlens_result = None
        try:
            from ally.tools import TOOL_REGISTRY
            factorlens_result = TOOL_REGISTRY["factors.residual_alpha"](
                returns=synthetic_returns,
                window=252,
                step=21,
                lags=5
            )
        except Exception as e:
            factorlens_result = {"error": str(e)}

        result_data = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "status": "experiment_completed",
            "symbols_processed": inputs["symbols"],
            "data_mode": "live" if use_live_data else "fixtures",
            "runtime_status": runtime_status,
            "metrics": {
                "processing_time_ms": 1500,
                "data_points": len(inputs["symbols"]) * 252,
                "memory_peak_mb": 45.7
            },
            "factorlens": factorlens_result.data if hasattr(factorlens_result, 'data') else factorlens_result
        }
        
        # Link receipts for provenance tracking
        receipts = []
        if use_live_data and session_id:
            receipts = link_receipts_to_run(run_id, session_id)
        
        # Store run with provenance
        store_run_with_provenance(
            run_id=run_id,
            task=f"orchestrator.run:{experiment_id}",
            ts=ts_iso,
            inputs=inputs,
            outputs=result_data,
            session_id=session_id
        )
        
        # Create report summary with receipt references
        receipt_cost = sum(r.cost_cents or 0 for r in receipts)
        receipt_vendors = list(set(r.vendor for r in receipts))
        
        audit_hash = hashlib.sha256(
            json.dumps({**inputs, **result_data}, sort_keys=True).encode()
        ).hexdigest()
        
        inputs_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True).encode()
        ).hexdigest()
        
        report = ReportSummary(
            run_id=run_id,
            task=f"orchestrator.run:{experiment_id}",
            ts_iso=ts_iso,
            html_path=f"reports/{run_id}_experiment.html",
            receipts=receipts,
            receipt_cost_cents=receipt_cost,
            receipt_vendors=receipt_vendors,
            audit_hash=audit_hash,
            inputs_hash=inputs_hash,
            code_hash=hashlib.sha256(orchestrator_run.__code__.co_code).hexdigest()
        )
        
        return ToolResult(
            ok=True,
            data={
                **result_data,
                "provenance": {
                    "receipts_linked": len(receipts),
                    "receipt_cost_cents": receipt_cost,
                    "receipt_vendors": receipt_vendors,
                    "audit_hash": audit_hash
                },
                "report_summary": report.model_dump()
            }
        )
        
    except Exception as e:
        return ToolResult(ok=False, data={"error": str(e)})


def _maybe_runtime(experiment_id: str, symbols: List[str], live: bool = False) -> Dict[str, Any]:
    """Helper to conditionally run runtime environment."""
    if not live:
        return {
            "runtime_used": False,
            "reason": "runtime_live=False",
            "mode": "fixture"
        }
    
    # Check environment
    if os.getenv("ALLY_LIVE") != "1":
        return {
            "runtime_used": False, 
            "reason": "ALLY_LIVE!=1",
            "mode": "fixture"
        }
    
    # Mock runtime execution
    return {
        "runtime_used": True,
        "experiment_id": experiment_id,
        "symbols": symbols,
        "mode": "live",
        "execution_time_ms": 2300,
        "containers_spawned": len(symbols)
    }