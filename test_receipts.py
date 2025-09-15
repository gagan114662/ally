#!/usr/bin/env python3
"""
Tests for receipt system - hashing determinism, DB insert, offline proof
"""

import os
import tempfile
import json
from datetime import datetime

from ally.utils.hashing import hash_payload, hash_inputs, content_hash
from ally.utils.db import DatabaseManager
from ally.utils.receipts import store_tool_receipt, get_tool_receipt, create_proof_line
from ally.schemas.base import ToolResult


def test_hashing_determinism():
    """Test that hashing is deterministic across runs"""
    print("Testing hashing determinism...")
    
    # Test hash_payload determinism
    test_data = {
        "symbol": "AAPL",
        "data": [1, 2, 3, 4, 5],
        "timestamp": "2024-01-01T00:00:00Z"
    }
    
    hash1 = hash_payload(test_data)
    hash2 = hash_payload(test_data)
    assert hash1 == hash2, f"hash_payload not deterministic: {hash1} != {hash2}"
    
    # Test hash_inputs determinism
    inputs = {"symbol": "AAPL", "lookback": 100, "patterns": ["engulfing"]}
    
    inputs_hash1 = hash_inputs(inputs)
    inputs_hash2 = hash_inputs(inputs)
    assert inputs_hash1 == inputs_hash2, f"hash_inputs not deterministic: {inputs_hash1} != {inputs_hash2}"
    
    # Test order independence for dictionaries
    inputs_reordered = {"patterns": ["engulfing"], "symbol": "AAPL", "lookback": 100}
    inputs_hash3 = hash_inputs(inputs_reordered)
    assert inputs_hash1 == inputs_hash3, f"hash_inputs not order-independent: {inputs_hash1} != {inputs_hash3}"
    
    # Test content_hash determinism
    binary_data = b"test binary data for hashing"
    content_hash1 = content_hash(binary_data)
    content_hash2 = content_hash(binary_data)
    assert content_hash1 == content_hash2, f"content_hash not deterministic: {content_hash1} != {content_hash2}"
    
    print("✅ Hashing determinism tests passed")


def test_db_insert_and_retrieval():
    """Test database insert and retrieval operations"""
    print("Testing database operations...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        db = DatabaseManager(db_path)
        
        # Test receipt storage
        tool_name = "test.tool"
        inputs = {"symbol": "TEST", "param": "value"}
        raw_payload = {"result": "success", "data": [1, 2, 3]}
        timestamp = "2024-01-01T00:00:00Z"
        
        receipt_hash = store_tool_receipt(tool_name, inputs, raw_payload, timestamp)
        assert receipt_hash, "Failed to store receipt"
        assert len(receipt_hash) == 16, f"Receipt hash wrong length: {len(receipt_hash)} != 16"
        
        # Test receipt retrieval
        retrieved = get_tool_receipt(tool_name, inputs)
        assert retrieved is not None, "Failed to retrieve receipt"
        assert retrieved['tool_name'] == tool_name
        assert retrieved['receipt_hash'] == receipt_hash
        assert retrieved['timestamp'] == timestamp
        
        # Test idempotency - storing same receipt twice should work
        receipt_hash2 = store_tool_receipt(tool_name, inputs, raw_payload, timestamp)
        assert receipt_hash == receipt_hash2, "Receipt storage not idempotent"
        
        db.close()
        print("✅ Database operations tests passed")
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_offline_proof_generation():
    """Test offline proof line generation"""
    print("Testing offline proof generation...")
    
    # Test data
    tool_name = "cv.detect_patterns"
    inputs = {"symbol": "AAPL", "lookback": 100, "patterns": ["engulfing_bull"]}
    raw_payload = {
        "detections": [{"pattern": "engulfing_bull", "confidence": 0.85}],
        "chart_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    }
    
    # Create temporary database for this test
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Store receipt
        original_env = os.environ.copy()
        os.environ['ALLY_DB_PATH'] = db_path  # Point to our test DB
        
        receipt_hash = store_tool_receipt(tool_name, inputs, raw_payload)
        
        # Generate proof line
        proof_line = create_proof_line(tool_name, inputs, receipt_hash)
        
        # Verify proof line format
        expected_pattern = r"PROOF:run:cv\.detect_patterns@[a-f0-9]{8}:[a-f0-9]{16}"
        import re
        assert re.match(expected_pattern, proof_line), f"Invalid proof line format: {proof_line}"
        
        # Verify we can retrieve the receipt using the proof line components
        retrieved = get_tool_receipt(tool_name, inputs)
        assert retrieved is not None, "Receipt not found for proof line"
        assert retrieved['receipt_hash'] == receipt_hash, "Receipt hash mismatch in proof line"
        
        print(f"✅ Generated proof line: {proof_line}")
        print("✅ Offline proof generation tests passed")
        
    finally:
        # Restore environment and cleanup
        os.environ.clear()
        os.environ.update(original_env)
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_tool_result_integration():
    """Test ToolResult integration with receipt system"""
    print("Testing ToolResult integration...")
    
    # Create a ToolResult
    data = {"symbols": ["AAPL", "MSFT"], "total_rows": 100}
    result = ToolResult.success(data)
    
    # Test store_receipt method
    tool_name = "data.load_ohlcv"
    inputs = {"symbols": ["AAPL", "MSFT"], "interval": "1d", "lookback": 30}
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        original_env = os.environ.copy()
        os.environ['ALLY_DB_PATH'] = db_path
        
        receipt_hash = result.store_receipt(tool_name, inputs)
        
        # Verify receipt was stored in meta
        assert result.meta.receipt_hash == receipt_hash, "Receipt hash not stored in meta"
        
        # Verify we can retrieve the receipt
        retrieved = get_tool_receipt(tool_name, inputs)
        assert retrieved is not None, "Receipt not stored in database"
        assert retrieved['receipt_hash'] == receipt_hash, "Receipt hash mismatch"
        
        print("✅ ToolResult integration tests passed")
        
    finally:
        # Restore environment and cleanup
        os.environ.clear()
        os.environ.update(original_env)
        if os.path.exists(db_path):
            os.unlink(db_path)


def run_all_tests():
    """Run all receipt system tests"""
    print("🧪 Running receipt system tests...\n")
    
    try:
        test_hashing_determinism()
        print()
        
        test_db_insert_and_retrieval()
        print()
        
        test_offline_proof_generation()
        print()
        
        test_tool_result_integration()
        print()
        
        print("🎉 All receipt system tests PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)