"""
Receipt utilities for storing and retrieving proof of tool execution
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

from .hashing import hash_payload, hash_inputs
from .db import get_db_manager


def store_tool_receipt(tool_name: str, inputs: Dict[str, Any], 
                      raw_payload: Any, timestamp: Optional[str] = None) -> str:
    """
    Store a receipt for tool execution
    
    Args:
        tool_name: Name of the tool that was executed
        inputs: Tool input parameters
        raw_payload: Raw response payload from tool
        timestamp: Optional timestamp (defaults to UTC now)
    
    Returns:
        Receipt hash (SHA-1)
    """
    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Create hashes
    args_hash = hash_inputs(inputs, algorithm="sha256")[:8]  # 8-char args hash
    receipt_hash = hash_payload(raw_payload)[:16]  # 16-char receipt hash
    
    # Convert payload to string for storage
    if isinstance(raw_payload, dict):
        payload_str = json.dumps(raw_payload, sort_keys=True, default=str)
    elif isinstance(raw_payload, bytes):
        payload_str = raw_payload.decode('utf-8', errors='replace')
    else:
        payload_str = str(raw_payload)
    
    # Store in database
    db = get_db_manager()
    success = db.store_receipt(
        tool_name=tool_name,
        args_hash=args_hash,
        receipt_hash=receipt_hash,
        payload_raw=payload_str,
        timestamp=timestamp
    )
    
    if success:
        return receipt_hash
    else:
        raise Exception("Failed to store receipt")


def get_tool_receipt(tool_name: str, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get stored receipt for tool execution
    
    Args:
        tool_name: Name of the tool
        inputs: Tool input parameters
    
    Returns:
        Receipt data or None if not found
    """
    args_hash = hash_inputs(inputs, algorithm="sha256")[:8]
    
    db = get_db_manager()
    return db.get_receipt(tool_name, args_hash)


def should_use_live_mode() -> bool:
    """
    Check if live mode is enabled via ALLY_LIVE environment variable
    
    Returns:
        True if live mode enabled (ALLY_LIVE=1), False otherwise
    """
    return os.getenv("ALLY_LIVE", "0") == "1"


def create_proof_line(tool_name: str, inputs: Dict[str, Any], receipt_hash: str) -> str:
    """
    Create PROOF line for CI validation
    
    Args:
        tool_name: Name of the tool
        inputs: Tool input parameters  
        receipt_hash: Receipt hash from store_tool_receipt
    
    Returns:
        PROOF line string
    """
    args_hash = hash_inputs(inputs, algorithm="sha256")[:8]
    return f"PROOF:run:{tool_name}@{args_hash}:{receipt_hash}"