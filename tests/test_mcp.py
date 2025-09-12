"""
Tests for Ally MCP tools
"""

import pytest
import sys
from pathlib import Path

# Add Ally to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "Ally"))

from Ally.tools.mcp import mcp_discover, mcp_call, mcp_list_servers, mcp_list_tools
from Ally.schemas.base import ToolResult
from Ally.adapters.mcp_client import MockMCPServer


def test_mock_mcp_server():
    """Test the mock MCP server functionality"""
    mock_server = MockMCPServer()
    
    # Test describe response
    describe_response = mock_server.get_describe_response()
    assert "capabilities" in describe_response
    assert "tools" in describe_response
    assert len(describe_response["tools"]) > 0
    
    # Test calculate tool
    result = mock_server.call_tool("calculate", {"expression": "2 + 3"})
    assert "result" in result
    assert result["result"] == 5
    
    # Test random_fact tool
    result = mock_server.call_tool("random_fact", {"category": "science"})
    assert "fact" in result
    assert "category" in result


def test_mcp_discover():
    """Test MCP server discovery with mock server"""
    result = mcp_discover(
        server_id="mock_test",
        endpoint="http://localhost:9000"
    )
    
    assert isinstance(result, ToolResult)
    assert result.ok
    assert "server_info" in result.data
    assert "tools_count" in result.data
    assert result.data["tools_count"] > 0
    assert result.data["status"] == "connected"
    
    # Should have warning about mock server
    assert len(result.meta.warnings) > 0
    assert "mock" in result.meta.warnings[0].lower()


def test_mcp_call_calculate():
    """Test MCP tool call with calculate tool"""
    # First discover the server
    discover_result = mcp_discover(
        server_id="mock_test",
        endpoint="http://localhost:9000"
    )
    assert discover_result.ok
    
    # Then call the calculate tool
    result = mcp_call(
        server_id="mock_test",
        tool="calculate",
        args={"expression": "10 + 5 * 2"}
    )
    
    assert isinstance(result, ToolResult)
    assert result.ok
    assert "result" in result.data
    assert result.data["success"]
    assert result.data["result"]["result"] == 20  # 10 + (5 * 2)


def test_mcp_call_random_fact():
    """Test MCP tool call with random_fact tool"""
    result = mcp_call(
        server_id="mock_test",
        tool="random_fact",
        args={"category": "math"}
    )
    
    assert isinstance(result, ToolResult)
    assert result.ok
    assert "result" in result.data
    assert result.data["success"]
    assert "fact" in result.data["result"]
    assert "category" in result.data["result"]


def test_mcp_call_invalid_tool():
    """Test MCP tool call with non-existent tool"""
    result = mcp_call(
        server_id="mock_test",
        tool="nonexistent_tool",
        args={}
    )
    
    assert isinstance(result, ToolResult)
    assert result.ok  # Mock server returns result with error field
    assert "result" in result.data
    assert not result.data["success"]
    assert "error" in result.data["result"]


def test_mcp_discover_invalid_inputs():
    """Test MCP discover with invalid inputs"""
    result = mcp_discover()  # Missing required fields
    
    assert isinstance(result, ToolResult)
    assert not result.ok
    assert len(result.errors) > 0
    assert "invalid inputs" in result.errors[0].lower()


def test_mcp_call_invalid_inputs():
    """Test MCP call with invalid inputs"""
    result = mcp_call()  # Missing required fields
    
    assert isinstance(result, ToolResult)
    assert not result.ok
    assert len(result.errors) > 0
    assert "invalid inputs" in result.errors[0].lower()


def test_mcp_list_servers():
    """Test listing MCP servers"""
    result = mcp_list_servers()
    
    assert isinstance(result, ToolResult)
    assert result.ok
    assert "servers" in result.data
    assert "total_servers" in result.data
    assert isinstance(result.data["servers"], list)


def test_mcp_list_tools():
    """Test listing MCP tools"""
    result = mcp_list_tools()
    
    assert isinstance(result, ToolResult)
    assert result.ok
    assert "tools" in result.data
    assert "total_tools" in result.data
    assert "tools_by_server" in result.data
    assert isinstance(result.data["tools"], list)


def test_mcp_integration_workflow():
    """Integration test: discover server, then call tools"""
    # Step 1: Discover server
    discover_result = mcp_discover(
        server_id="integration_test",
        endpoint="http://localhost:9000"
    )
    assert discover_result.ok
    
    # Step 2: Call multiple tools
    calc_result = mcp_call(
        server_id="integration_test",
        tool="calculate",
        args={"expression": "3 * 7"}
    )
    assert calc_result.ok
    assert calc_result.data["result"]["result"] == 21
    
    fact_result = mcp_call(
        server_id="integration_test",
        tool="random_fact",
        args={"category": "general"}
    )
    assert fact_result.ok
    assert "fact" in fact_result.data["result"]
    
    # Both calls should reference the same server
    assert calc_result.data["server_id"] == fact_result.data["server_id"]


if __name__ == "__main__":
    # Run tests manually
    print("Running MCP tools tests...")
    
    test_functions = [
        test_mock_mcp_server,
        test_mcp_discover,
        test_mcp_call_calculate,
        test_mcp_call_random_fact,
        test_mcp_call_invalid_tool,
        test_mcp_discover_invalid_inputs,
        test_mcp_call_invalid_inputs,
        test_mcp_list_servers,
        test_mcp_list_tools,
        test_mcp_integration_workflow
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__} passed")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All MCP tests passed!")
    else:
        print("❌ Some tests failed")