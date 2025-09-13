#!/usr/bin/env python3
"""M-Receipts-Everywhere proof emission for CI validation."""

import json
import hashlib
import os
from datetime import datetime
from typing import Dict, Any

def emit_proofs() -> Dict[str, Any]:
    """Generate M-Receipts-Everywhere proofs for CI validation."""
    proofs = {}
    
    try:
        # 1) Import and test orchestrator tools
        from ally.tools import TOOL_REGISTRY
        
        orchestrator_tools = [k for k in TOOL_REGISTRY.keys() if k.startswith("orchestrator.")]
        proofs["ORCHESTRATOR_TOOLS"] = orchestrator_tools
        proofs["ORCHESTRATOR_COUNT"] = len(orchestrator_tools)
        
        # 2) Test orchestrator.demo with receipt tracking
        demo_result = TOOL_REGISTRY["orchestrator.demo"](
            symbols=["SPY", "QQQ"],
            use_live_data=False,  # Dry mode for CI
            use_runtime=False
        )
        
        proofs["DEMO_OK"] = demo_result.ok
        if demo_result.ok:
            demo_data = demo_result.data
            provenance = demo_data.get("provenance", {})
            proofs["DEMO_RECEIPTS_LINKED"] = provenance.get("receipts_linked", 0)
            proofs["DEMO_AUDIT_HASH"] = provenance.get("audit_hash", "")[:16] + "..."
            
            report_summary = demo_data.get("report_summary", {})
            proofs["DEMO_REPORT_RUN_ID"] = report_summary.get("run_id", "")[:8] + "..."
            proofs["DEMO_RECEIPTS_COUNT"] = len(report_summary.get("receipts", []))
        
        # 3) Test orchestrator.run with receipt tracking
        run_result = TOOL_REGISTRY["orchestrator.run"](
            experiment_id="EXP_RECEIPTS_TEST",
            symbols=["AAPL", "GOOGL", "MSFT"],
            use_live_data=False,  # Dry mode for CI
            use_runtime=False
        )
        
        proofs["RUN_OK"] = run_result.ok
        if run_result.ok:
            run_data = run_result.data
            provenance = run_data.get("provenance", {})
            proofs["RUN_RECEIPTS_LINKED"] = provenance.get("receipts_linked", 0)
            proofs["RUN_AUDIT_HASH"] = provenance.get("audit_hash", "")[:16] + "..."
            
            report_summary = run_data.get("report_summary", {})
            proofs["RUN_REPORT_TASK"] = report_summary.get("task", "")
            proofs["RUN_RECEIPTS_COUNT"] = len(report_summary.get("receipts", []))
        
        # 4) Test provenance utilities
        from ally.utils.provenance import compute_provenance_hash, get_receipt_summary
        from ally.schemas.report import ReceiptRef
        
        # Create test receipt references
        test_receipts = [
            ReceiptRef(
                content_sha1="abc123def456",
                vendor="test_provider_1",
                endpoint="/api/quotes",
                ts_iso="2025-01-15T12:00:00Z",
                cost_cents=50
            ),
            ReceiptRef(
                content_sha1="def456ghi789",
                vendor="test_provider_2", 
                endpoint="/api/history",
                ts_iso="2025-01-15T12:05:00Z",
                cost_cents=75
            )
        ]
        
        test_inputs = {"symbols": ["TEST"], "mode": "dry"}
        test_outputs = {"status": "completed", "count": 2}
        
        prov_hash = compute_provenance_hash(test_inputs, test_outputs, test_receipts)
        proofs["PROVENANCE_HASH"] = prov_hash[:16] + "..."
        
        receipt_summary = get_receipt_summary(test_receipts)
        proofs["RECEIPT_SUMMARY_COST"] = receipt_summary["total_cost_cents"]
        proofs["RECEIPT_SUMMARY_VENDORS"] = receipt_summary["unique_vendors"]
        
        # 5) Schema validation
        from ally.schemas.report import ReportSummary
        
        test_report = ReportSummary(
            run_id="TEST_RUN_123",
            task="test_task",
            ts_iso="2025-01-15T12:00:00Z",
            html_path="test_report.html",
            receipts=test_receipts,
            receipt_cost_cents=125,
            receipt_vendors=["test_provider_1", "test_provider_2"],
            audit_hash=prov_hash,
            inputs_hash="test_inputs_hash",
            code_hash="test_code_hash"
        )
        
        proofs["SCHEMA_VALIDATION_OK"] = True
        proofs["SCHEMA_RECEIPTS_COUNT"] = len(test_report.receipts)
        proofs["SCHEMA_VENDORS_COUNT"] = len(test_report.receipt_vendors)
        
        # 6) Database integration test
        from ally.utils.db import get_db_manager
        
        db = get_db_manager()
        
        # Test receipt insertion capability
        from ally.schemas.receipts import Receipt
        test_receipt = Receipt(
            vendor="test_db_vendor",
            endpoint="/test/endpoint",
            params={"symbol": "TEST"},
            ts_iso=datetime.utcnow().isoformat() + "Z",
            content_sha1="test_content_hash",
            bytes=1024,
            cost_cents=25
        )
        
        db_insert_ok = db.insert_receipt(test_receipt)
        proofs["DB_INSERT_OK"] = db_insert_ok
        
        # Test receipt stats
        stats = db.get_receipt_stats()
        proofs["DB_STATS_OK"] = stats is not None
        if stats:
            proofs["DB_TOTAL_RECEIPTS"] = stats.get("total_receipts", 0)
        
        # 7) Overall M-Receipts-Everywhere status
        proofs["MRECEIPTS_STATUS"] = "operational"
        proofs["MRECEIPTS_FEATURES"] = [
            "receipt_refs",
            "provenance_hash", 
            "orchestrator_wiring",
            "db_integration",
            "schema_validation"
        ]
        
    except Exception as e:
        proofs["ERROR"] = str(e)
        proofs["MRECEIPTS_STATUS"] = "error"
    
    return proofs


def main():
    """Main proof emission with CI output."""
    proofs = emit_proofs()
    
    # Emit to GitHub Step Summary
    for key, value in proofs.items():
        if isinstance(value, (str, int, bool)):
            print(f"PROOF:{key}: {value}")
        else:
            print(f"PROOF:{key}: {json.dumps(value)}")
    
    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/mreceipts_proofs.json", "w") as f:
        json.dump(proofs, f, indent=2)
    
    # Deterministic hash for CI validation
    proof_content = json.dumps(proofs, sort_keys=True)
    det_hash = hashlib.sha1(proof_content.encode()).hexdigest()
    print(f"PROOF:MRECEIPTS_DET_HASH: {det_hash}")
    
    return proofs


if __name__ == "__main__":
    main()