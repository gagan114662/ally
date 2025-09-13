#!/usr/bin/env python3
"""
M-RealData Gate proof emission script
Generates verification proofs for live data and receipt system
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ally.utils.db import get_db_manager
from ally.utils.receipts import list_receipts_by_vendor
from ally.schemas.receipts import LiveAccessError


def emit_dry_proofs() -> Dict[str, Any]:
    """Emit proofs for dry mode (CI environment)"""
    proofs = {
        "REALDATA_MODE": "dry",
        "RECEIPTS_N": 0,
        "QUORUM_OK": "n/a", 
        "COST_CENTS": 0,
        "LIVE_ACCESS": "denied"
    }
    
    # Verify that live access is properly blocked
    try:
        import ally.tools.data_live  # Ensure tools are registered
        from ally.utils.receipts import enforce_live_mode_or_die
        enforce_live_mode_or_die("test", live=True)
        proofs["LIVE_GATE"] = "broken"  # Should not reach here
    except LiveAccessError:
        proofs["LIVE_GATE"] = "working"  # Expected
    
    # Check for any existing receipts (shouldn't be any in CI)
    try:
        db_manager = get_db_manager()
        stats = db_manager.get_receipt_stats()
        proofs["RECEIPTS_N"] = stats["total_receipts"]
        proofs["COST_CENTS"] = stats["total_cost_cents"]
        
        if stats["total_receipts"] > 0:
            proofs["WARNING"] = "Found receipts in CI environment"
            
    except Exception as e:
        proofs["DB_ERROR"] = str(e)
    
    return proofs


def emit_live_proofs() -> Dict[str, Any]:
    """Emit proofs for live mode (local environment with ALLY_LIVE=1)"""
    proofs = {
        "REALDATA_MODE": "live",
        "RECEIPTS_N": 0,
        "QUORUM_OK": "n/a",
        "COST_CENTS": 0
    }
    
    # Check live access capability
    ally_live = os.getenv("ALLY_LIVE")
    if ally_live != "1":
        proofs["REALDATA_MODE"] = "blocked"
        proofs["REASON"] = f"ALLY_LIVE={ally_live}, expected '1'"
        return proofs
    
    try:
        # Get receipt statistics from database
        db_manager = get_db_manager()
        stats = db_manager.get_receipt_stats()
        
        proofs["RECEIPTS_N"] = stats["total_receipts"]
        proofs["COST_CENTS"] = stats["total_cost_cents"]
        
        # Get sample receipt SHA1 if available
        if stats["total_receipts"] > 0:
            # Find a recent receipt
            for vendor in stats["by_vendor"].keys():
                receipts = db_manager.get_receipts_by_vendor(vendor, limit=1)
                if receipts:
                    proofs["RECEIPT_SHA1"] = receipts[0]["content_sha1"]
                    proofs["SAMPLE_VENDOR"] = vendor
                    break
        
        # Check quorum capability (basic verification)
        if stats["total_receipts"] >= 2:
            # If we have receipts from multiple vendors, quorum is possible
            vendor_count = len([v for v, count in stats["by_vendor"].items() if count > 0])
            if vendor_count >= 2:
                proofs["QUORUM_OK"] = True
            else:
                proofs["QUORUM_OK"] = False
                proofs["QUORUM_REASON"] = f"Only {vendor_count} vendor(s) with receipts"
        else:
            proofs["QUORUM_OK"] = "insufficient_data"
        
        # Verify receipt files exist on disk
        receipts_dir = Path("runs/receipts")
        if receipts_dir.exists():
            receipt_files = list(receipts_dir.glob("*.json"))
            proofs["RECEIPT_FILES"] = len(receipt_files)
        else:
            proofs["RECEIPT_FILES"] = 0
        
        # Verify raw data files exist
        raw_dir = Path("runs/raw")
        if raw_dir.exists():
            raw_files = sum(1 for _ in raw_dir.rglob("*.json"))
            proofs["RAW_FILES"] = raw_files
        else:
            proofs["RAW_FILES"] = 0
            
    except Exception as e:
        proofs["ERROR"] = str(e)
        proofs["REALDATA_MODE"] = "error"
    
    return proofs


def verify_tools_exist() -> Dict[str, bool]:
    """Verify required tools are registered"""
    verification = {}
    
    try:
        # Import tools to ensure registration
        import ally.tools.data_live
        from ally.tools import TOOL_REGISTRY
        
        required_tools = [
            "data.live_fetch",
            "data.live_history"
        ]
        
        for tool in required_tools:
            verification[tool] = tool in TOOL_REGISTRY
            
    except Exception as e:
        verification["error"] = str(e)
    
    return verification


def main():
    parser = argparse.ArgumentParser(description="Emit M-RealData Gate proofs")
    parser.add_argument("--dry", action="store_true", 
                       help="Emit dry mode proofs (CI environment)")
    parser.add_argument("--live", action="store_true",
                       help="Emit live mode proofs (local environment)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify tool registration only")
    
    args = parser.parse_args()
    
    if args.verify:
        verification = verify_tools_exist()
        print(json.dumps(verification, indent=2))
        return
    
    # Default to dry mode if neither specified
    if not args.live:
        args.dry = True
    
    if args.dry:
        proofs = emit_dry_proofs()
    else:
        proofs = emit_live_proofs()
    
    # Add tool verification to proofs
    verification = verify_tools_exist()
    proofs["TOOLS_REGISTERED"] = all(verification.values())
    
    # Create proof bundle directory and files
    from pathlib import Path
    bundle_dir = Path("mrealdata-proof-bundle")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    # Write JSON format for programmatic access
    with open(bundle_dir / "proofs.json", "w") as f:
        json.dump(proofs, f, indent=2)
    
    # Write plain text format for comments
    proof_lines = []
    for key, value in proofs.items():
        if not key.startswith("TOOLS_") and key != "ERROR":
            proof_lines.append(f"PROOF:{key}: {json.dumps(value) if isinstance(value, (dict, list)) else value}")
    
    with open(bundle_dir / "proofs.txt", "w") as f:
        f.write("\n".join(proof_lines))
    
    # Print proofs in CI-friendly format
    print(json.dumps(proofs, indent=2))
    
    # Also print individual PROOF lines for easy copying
    print("\n# Individual PROOF lines:")
    for line in proof_lines:
        print(line)


if __name__ == "__main__":
    main()