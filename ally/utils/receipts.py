"""
Receipt utilities for M-RealData Gate system
Handles payload storage, attestation, and quorum verification
"""

from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from ally.schemas.receipts import Receipt, QuorumVerdict, LiveAccessError, BudgetExceededError, QuorumFailureError
from ally.utils.db import get_db_manager


def enforce_live_mode_or_die(tool_name: str, live: bool = False) -> None:
    """
    Enforce double gate: live=True AND ALLY_LIVE=1 required for network access
    Raises LiveAccessError if either gate is missing
    """
    if not live:
        raise LiveAccessError(f"{tool_name}: live=False, network access denied")
    
    if os.getenv("ALLY_LIVE") != "1":
        raise LiveAccessError(f"{tool_name}: ALLY_LIVE!=1, network access denied")


def write_payload_and_receipt(vendor: str, endpoint: str, params: Dict[str, Any], 
                            payload_bytes: bytes, cost_cents: Optional[int] = None) -> Receipt:
    """
    Write raw payload and generate attestation receipt
    
    Args:
        vendor: Provider name (polygon, alphavantage, etc.)
        endpoint: API endpoint called
        params: Request parameters (secrets removed)
        payload_bytes: Raw response bytes
        cost_cents: Estimated cost in cents
    
    Returns:
        Receipt with content hash and metadata
    """
    # Generate content hash
    content_sha1 = hashlib.sha1(payload_bytes).hexdigest()
    ts_iso = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    # Create directories
    raw_dir = Path("runs/raw") / vendor
    receipts_dir = Path("runs/receipts")
    raw_dir.mkdir(parents=True, exist_ok=True)
    receipts_dir.mkdir(parents=True, exist_ok=True)
    
    # Write raw payload
    # Use timestamp + endpoint for filename to avoid collisions
    safe_endpoint = endpoint.replace('/', '_').replace('?', '_')
    raw_filename = f"{ts_iso.replace(':', '-')}_{safe_endpoint}.json"
    raw_path = raw_dir / raw_filename
    
    with open(raw_path, 'wb') as f:
        f.write(payload_bytes)
    
    # Create receipt
    receipt = Receipt(
        vendor=vendor,
        endpoint=endpoint,
        params=_sanitize_params(params),
        ts_iso=ts_iso,
        content_sha1=content_sha1,
        bytes=len(payload_bytes),
        cost_cents=cost_cents
    )
    
    # Write receipt
    receipt_path = receipts_dir / f"{content_sha1}.json"
    with open(receipt_path, 'w') as f:
        json.dump(receipt.model_dump(), f, indent=2)
    
    # Store in DuckDB for querying
    try:
        db_manager = get_db_manager()
        db_manager.insert_receipt(receipt)
    except Exception as e:
        print(f"Warning: Failed to store receipt in database: {e}")
    
    return receipt


def quorum_check(measurements: List[float], tolerance_bps: float, 
                 members: List[str], metric: str) -> QuorumVerdict:
    """
    Check if multiple provider measurements agree within tolerance
    
    Args:
        measurements: Values from each provider
        tolerance_bps: Allowed variance in basis points (e.g., 5 = 0.05%)
        members: Provider names corresponding to measurements
        metric: What is being measured (close, volume, etc.)
    
    Returns:
        QuorumVerdict with agreement status
    """
    if len(measurements) < 2:
        return QuorumVerdict(
            members=members,
            metric=metric,
            tolerance_bps=tolerance_bps,
            ok=True,  # Single measurement always "agrees"
            measurements=measurements,
            variance_bps=0.0
        )
    
    # Calculate variance as max deviation from mean in basis points
    mean_val = sum(measurements) / len(measurements)
    max_deviation = max(abs(m - mean_val) for m in measurements)
    variance_bps = (max_deviation / mean_val) * 10000 if mean_val != 0 else 0.0
    
    agreement = variance_bps <= tolerance_bps
    
    return QuorumVerdict(
        members=members,
        metric=metric,
        tolerance_bps=tolerance_bps,
        ok=agreement,
        measurements=measurements,
        variance_bps=variance_bps
    )


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove secrets from parameters before storing in receipt"""
    sensitive_keys = {'apikey', 'api_key', 'token', 'secret', 'password', 'auth'}
    
    sanitized = {}
    for key, value in params.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        else:
            sanitized[key] = value
    
    return sanitized


def verify_receipt_exists(content_sha1: str) -> bool:
    """Check if receipt exists for given content hash"""
    receipt_path = Path("runs/receipts") / f"{content_sha1}.json"
    return receipt_path.exists()


def load_receipt(content_sha1: str) -> Optional[Receipt]:
    """Load receipt by content hash"""
    receipt_path = Path("runs/receipts") / f"{content_sha1}.json"
    if not receipt_path.exists():
        return None
    
    with open(receipt_path) as f:
        data = json.load(f)
    
    return Receipt(**data)


def list_receipts_by_vendor(vendor: str) -> List[Receipt]:
    """List all receipts for a specific vendor"""
    receipts_dir = Path("runs/receipts")
    if not receipts_dir.exists():
        return []
    
    receipts = []
    for receipt_file in receipts_dir.glob("*.json"):
        try:
            with open(receipt_file) as f:
                data = json.load(f)
            receipt = Receipt(**data)
            if receipt.vendor == vendor:
                receipts.append(receipt)
        except Exception:
            continue  # Skip malformed receipts
    
    return receipts