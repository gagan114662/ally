"""Provenance utilities for end-to-end receipt tracking."""

import hashlib
import json
from typing import List, Dict, Any, Optional
from ally.utils.db import get_db_manager
from ally.schemas.report import ReceiptRef


def link_receipts_to_run(run_id: str, session_id: Optional[str] = None) -> List[ReceiptRef]:
    """Link all receipts from a session to a run for provenance tracking."""
    db = get_db_manager()
    
    # Query receipts by session_id if provided, otherwise by run_id timeframe
    if session_id:
        where_clause = f"session_id = '{session_id}'"
    else:
        where_clause = f"ts_iso >= (SELECT ts FROM runs WHERE run_id = '{run_id}')"
    
    result = db.query("data_receipts", where=where_clause, limit=1000)
    
    receipt_refs = []
    for row in result.get("rows", []):
        receipt_ref = ReceiptRef(
            content_sha1=row["content_sha1"],
            vendor=row["vendor"],
            endpoint=row["endpoint"], 
            ts_iso=row["ts_iso"],
            cost_cents=row.get("cost_cents", 0)
        )
        receipt_refs.append(receipt_ref)
    
    return receipt_refs


def compute_provenance_hash(inputs: Dict[str, Any], outputs: Dict[str, Any], 
                           receipts: List[ReceiptRef]) -> str:
    """Compute SHA256 hash of inputs + outputs + receipts for audit trail."""
    # Sort for deterministic hashing
    sorted_inputs = json.dumps(inputs, sort_keys=True)
    sorted_outputs = json.dumps(outputs, sort_keys=True) 
    
    # Convert receipts to dict and sort
    receipts_data = [r.model_dump() for r in receipts]
    sorted_receipts = json.dumps(receipts_data, sort_keys=True)
    
    combined = f"{sorted_inputs}|{sorted_outputs}|{sorted_receipts}"
    return hashlib.sha256(combined.encode()).hexdigest()


def get_receipt_summary(receipts: List[ReceiptRef]) -> Dict[str, Any]:
    """Get summary statistics from receipt references."""
    if not receipts:
        return {
            "total_cost_cents": 0,
            "unique_vendors": [],
            "receipt_count": 0,
            "earliest_fetch": None,
            "latest_fetch": None
        }
    
    total_cost = sum(r.cost_cents or 0 for r in receipts)
    unique_vendors = list(set(r.vendor for r in receipts))
    timestamps = [r.ts_iso for r in receipts]
    
    return {
        "total_cost_cents": total_cost,
        "unique_vendors": sorted(unique_vendors),
        "receipt_count": len(receipts),
        "earliest_fetch": min(timestamps) if timestamps else None,
        "latest_fetch": max(timestamps) if timestamps else None
    }


def store_run_with_provenance(run_id: str, task: str, ts: str, inputs: Dict[str, Any],
                             outputs: Dict[str, Any], session_id: Optional[str] = None) -> bool:
    """Store run with linked receipt provenance in database."""
    try:
        db = get_db_manager()
        
        # Link receipts to this run
        receipts = link_receipts_to_run(run_id, session_id)
        
        # Compute provenance hash
        audit_hash = compute_provenance_hash(inputs, outputs, receipts)
        inputs_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        
        # Get receipt summary
        receipt_summary = get_receipt_summary(receipts)
        
        # Store enhanced run record
        success = db.log_run(
            run_id=run_id,
            task=task,
            code_hash="",  # TODO: compute from function code
            inputs_hash=inputs_hash,
            ts=ts,
            metrics={"receipt_cost_cents": receipt_summary["total_cost_cents"]},
            events=[{
                "type": "provenance_linked",
                "receipt_count": receipt_summary["receipt_count"],
                "vendors": receipt_summary["unique_vendors"],
                "audit_hash": audit_hash
            }],
            trades=[],  # TODO: extract trades from outputs if present
            notes=f"Provenance: {receipt_summary['receipt_count']} receipts, {len(receipt_summary['unique_vendors'])} vendors"
        )
        
        return success
        
    except Exception as e:
        print(f"Error storing run with provenance: {e}")
        return False