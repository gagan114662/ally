"""
Regression tests for memory and reporting security/functionality
"""

import pytest
from ally.tools import TOOL_REGISTRY


def test_memory_query_injection_inert(tmp_path, monkeypatch):
    """Test that SQL injection attempts are neutralized by parameterized queries"""
    # Log a known run
    run_id = "RUN_M9_TEST_INJECTION"
    TOOL_REGISTRY["memory.log_run"](
        run_id=run_id, 
        task="demo", 
        code_hash="deadbeef", 
        inputs_hash="cafebabe",
        ts="2025-01-15T12:00:00Z", 
        metrics={"sharpe_ratio": 1.23}, 
        events=[]
    )
    
    # Legacy where string with attempted injection should NOT break or return extra rows
    where = "run_id='RUN_M9_TEST_INJECTION'; DROP TABLE runs; --"
    res = TOOL_REGISTRY["memory.query"](table="runs", where=where, limit=5)
    rows = res.data["rows"]
    
    # Should find our test run (injection neutralized, parsed safely)
    assert any(r.get("run_id") == run_id for r in rows)
    
    # Verify we can still query the table afterwards (i.e., not dropped by injection)
    res2 = TOOL_REGISTRY["memory.query"](table="runs", limit=1)
    assert isinstance(res2.data["rows"], list)


def test_memory_table_allowlist_enforced():
    """Test that only allowlisted tables can be queried"""
    # Valid table should work
    res = TOOL_REGISTRY["memory.query"](table="runs", limit=1)
    assert res.ok
    
    # Invalid table should be rejected
    res = TOOL_REGISTRY["memory.query"](table="malicious_table", limit=1)
    assert not res.ok
    assert "Invalid table" in str(res.error)


def test_report_path_is_relative():
    """Test that report paths are stored as repo-relative for portability"""
    res = TOOL_REGISTRY["orchestrator.run"](
        experiment_id="EXP_demo_path_test", 
        symbols=["BTCUSDT"], 
        interval="1h",
        lookback=600, 
        targets={"annual_return": 0.1, "sharpe_ratio": 1.0},
        risk_policy_yaml="max_leverage: 3.0\nmax_single_order_notional: 25000",
        save_run=True, 
        make_report=True
    )
    
    assert res.ok
    summary = res.data
    
    # Report path should be relative, not absolute
    assert summary["report_path"].startswith("reports/"), f"Path should be relative: {summary['report_path']}"
    assert not summary["report_path"].startswith("/"), f"Path should not be absolute: {summary['report_path']}"


def test_safe_query_parameterized():
    """Test that the new safe_query method works correctly with filters"""
    from ally.utils.db import get_db_manager
    
    db = get_db_manager()
    
    # Test with valid table and filters
    rows = db.safe_query(table="runs", filters={"task": "demo"}, limit=5)
    assert isinstance(rows, list)
    
    # Test with empty filters
    rows = db.safe_query(table="runs", filters={}, limit=1)
    assert isinstance(rows, list)
    
    # Test with invalid table should raise ValueError
    with pytest.raises(ValueError, match="Invalid table"):
        db.safe_query(table="invalid_table", filters={}, limit=1)