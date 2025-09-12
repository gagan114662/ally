"""
MCP (Model Context Protocol) tools for Ally
"""

from typing import Dict, Any, List
import time

from ..tools import register
from ..schemas.base import ToolResult
from ..schemas.mcp import MCPDiscoverIn, MCPCallIn
from ..adapters.mcp_client import MCPClient, MockMCPServer


# Global MCP client instance
mcp_client = MCPClient()


@register("mcp.discover")
def mcp_discover(**kwargs) -> ToolResult:
    """
    Discover MCP server capabilities and tools
    
    Connects to an MCP server and retrieves available tools and capabilities
    """
    try:
        inputs = MCPDiscoverIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    warnings = []
    
    try:
        # Special handling for mock server
        if "localhost:9000" in inputs.endpoint or "mock" in inputs.server_id.lower():
            # Use mock server for testing
            mock_server = MockMCPServer()
            
            # Create mock server info
            from ..schemas.mcp import MCPServerInfo
            server_info = MCPServerInfo(
                server_id=inputs.server_id,
                endpoint=inputs.endpoint,
                status="connected",
                capabilities=["tools", "resources"],
                tools=mock_server.get_describe_response()["tools"],
                last_ping=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            warnings.append("Using mock MCP server for demonstration")
            
            return ToolResult.success(
                data={
                    "server_info": server_info.model_dump(),
                    "tools_count": len(server_info.tools),
                    "capabilities": server_info.capabilities,
                    "status": "connected"
                },
                warnings=warnings
            )
        
        # Real server discovery
        server_info = mcp_client.discover_server(
            inputs.server_id, 
            inputs.endpoint, 
            inputs.timeout_s
        )
        
        if server_info.status == "connected":
            return ToolResult.success(
                data={
                    "server_info": server_info.model_dump(),
                    "tools_count": len(server_info.tools),
                    "capabilities": server_info.capabilities,
                    "status": "connected"
                },
                warnings=warnings
            )
        else:
            return ToolResult.error([
                f"Failed to connect to server: {server_info.error_message}"
            ])
            
    except Exception as e:
        return ToolResult.error([f"Discovery failed: {str(e)}"])


@register("mcp.call")
def mcp_call(**kwargs) -> ToolResult:
    """
    Call a tool on an MCP server
    
    Executes a tool call on a previously discovered MCP server
    """
    try:
        inputs = MCPCallIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    warnings = []
    
    try:
        # Special handling for mock server
        if "mock" in inputs.server_id.lower():
            mock_server = MockMCPServer()
            
            start_time = time.time()
            mock_result = mock_server.call_tool(inputs.tool, inputs.args)
            duration_ms = int((time.time() - start_time) * 1000)
            
            warnings.append("Using mock MCP server for demonstration")
            
            return ToolResult.success(
                data={
                    "server_id": inputs.server_id,
                    "tool_name": inputs.tool,
                    "args": inputs.args,
                    "result": mock_result,
                    "duration_ms": duration_ms,
                    "success": "error" not in mock_result
                },
                warnings=warnings
            )
        
        # Real server call
        call_result = mcp_client.call_tool(
            inputs.server_id,
            inputs.tool,
            inputs.args,
            inputs.timeout_s
        )
        
        if call_result.success:
            return ToolResult.success(
                data=call_result.model_dump(),
                warnings=warnings
            )
        else:
            return ToolResult.error([
                f"Tool call failed: {call_result.error}"
            ])
            
    except Exception as e:
        return ToolResult.error([f"Call failed: {str(e)}"])


@register("mcp.list_servers")
def mcp_list_servers(**kwargs) -> ToolResult:
    """
    List all registered MCP servers
    
    Returns information about all discovered MCP servers
    """
    try:
        servers = mcp_client.list_servers()
        
        return ToolResult.success(
            data={
                "servers": [server.model_dump() for server in servers],
                "total_servers": len(servers)
            }
        )
        
    except Exception as e:
        return ToolResult.error([f"Failed to list servers: {str(e)}"])


@register("mcp.list_tools")
def mcp_list_tools(**kwargs) -> ToolResult:
    """
    List all available MCP tools
    
    Returns tools from all registered servers or a specific server
    """
    server_id = kwargs.get("server_id")
    
    try:
        tools = mcp_client.list_tools(server_id)
        
        # Group tools by server
        tools_by_server = {}
        for tool in tools:
            if tool.server_id not in tools_by_server:
                tools_by_server[tool.server_id] = []
            tools_by_server[tool.server_id].append(tool.model_dump())
        
        return ToolResult.success(
            data={
                "tools": [tool.model_dump() for tool in tools],
                "tools_by_server": tools_by_server,
                "total_tools": len(tools)
            }
        )
        
    except Exception as e:
        return ToolResult.error([f"Failed to list tools: {str(e)}"])


# Helper function to create mock server fixtures
def create_mock_server_fixture():
    """Create a mock MCP server response for testing"""
    mock_server = MockMCPServer()
    return {
        "endpoint": "http://localhost:9000",
        "describe_response": mock_server.get_describe_response(),
        "sample_calls": {
            "calculate": {"expression": "2 + 3 * 4"},
            "random_fact": {"category": "science"}
        }
    }


if __name__ == "__main__":
    # Test MCP tools
    print("Testing MCP tools...")
    
    # Test discover with mock server
    result = mcp_discover(
        server_id="mock_server",
        endpoint="http://localhost:9000"
    )
    print(f"Discover result: {result.ok}")
    if result.ok:
        print(f"Tools found: {result.data['tools_count']}")
    
    # Test tool call
    if result.ok:
        call_result = mcp_call(
            server_id="mock_server",
            tool="calculate",
            args={"expression": "10 + 5"}
        )
        print(f"Call result: {call_result.ok}")
        if call_result.ok:
            print(f"Calculation result: {call_result.data['result']}")
    
    print("MCP tools test complete!")