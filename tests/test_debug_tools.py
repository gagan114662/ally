import pytest, json, os
from ally.tools import TOOL_REGISTRY

pytestmark = pytest.mark.mdebug

def test_make_repro_and_propcheck(tmp_path, monkeypatch):
    """Test make_repro and propcheck functionality"""
    # Test make_repro with a simple tool call
    r = TOOL_REGISTRY["debug.make_repro"]("memory.log_run", {
        "run_id": "test_123", 
        "task": "test", 
        "code_hash": "deadbeef", 
        "inputs_hash": "cafebabe"
    }, seed=1337)
    
    assert r.ok 
    assert r.data["repro_path"].endswith(".py")
    assert len(r.data["repro_sha1"]) == 40
    
    # Verify repro file exists and has expected content
    repro_path = r.data["repro_path"]
    assert os.path.exists(repro_path)
    
    with open(repro_path, 'r') as f:
        content = f.read()
        assert "memory.log_run" in content
        assert "set_determinism(1337)" in content
        assert "TOOL_REGISTRY" in content

    # Test property checking
    p = TOOL_REGISTRY["debug.propcheck"](trials=10)
    assert p.ok
    assert p.data["prop_fails"] == 0

def test_lint_typecheck_smoke():
    """Test lint and typecheck functionality"""
    r = TOOL_REGISTRY["debug.lint_typecheck"](["ally"])
    assert "lint_ok" in r.data 
    assert "mypy_ok" in r.data
    # Don't enforce strict passing as tools might not be installed

def test_capture_trace_success():
    """Test trace capture for successful execution"""
    def dummy_success():
        return {"result": "success"}
    
    r = TOOL_REGISTRY["debug.capture_trace"](
        "test_success",
        dummy_success
    )
    
    assert r.ok
    assert "result" in r.data

def test_capture_trace_failure():
    """Test trace capture for failing execution"""
    def dummy_failure():
        raise ValueError("Test error")
    
    r = TOOL_REGISTRY["debug.capture_trace"](
        "test_failure", 
        dummy_failure
    )
    
    assert not r.ok
    assert "trace_path" in r.data
    assert os.path.exists(r.data["trace_path"])
    
    # Verify trace file contents
    with open(r.data["trace_path"], 'r') as f:
        trace_data = json.load(f)
        assert trace_data["task"] == "test_failure"
        assert trace_data["error_type"] == "ValueError"
        assert "Test error" in trace_data["error"]

def test_minimize_functionality():
    """Test input minimization"""
    # Mock a tool that fails when list has more than 2 items
    def mock_failing_tool(items):
        from ally.schemas.base import ToolResult, Meta
        if len(items) > 2:
            return ToolResult(ok=False, data={"error": "too many items"}, errors=[], meta=Meta())
        return ToolResult(ok=True, data={"items": items}, errors=[], meta=Meta())
    
    # Register mock tool temporarily
    TOOL_REGISTRY["test.mock_fail"] = mock_failing_tool
    
    try:
        r = TOOL_REGISTRY["debug.minimize"](
            tool_name="test.mock_fail",
            list_arg_name="items", 
            inputs={"items": ["a", "b", "c", "d", "e"]},
            max_iters=10
        )
        
        assert r.ok
        assert r.data["min_size"] <= 3  # Should minimize to 3 or fewer
        assert len(r.data["hash"]) == 40  # SHA1 length
        
    finally:
        # Cleanup
        if "test.mock_fail" in TOOL_REGISTRY:
            del TOOL_REGISTRY["test.mock_fail"]

def test_bisect_functionality():
    """Test git bisect helper"""
    r = TOOL_REGISTRY["debug.bisect"](["echo", "test"])
    
    # Should return safely whether git is available or not
    assert r.ok
    assert "bisect_supported" in r.data