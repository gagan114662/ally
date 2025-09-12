"""
MCP (Model Context Protocol) schemas for Ally
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Dict, Any, Optional
from ..schemas.base import ToolInput


class MCPDiscoverIn(ToolInput):
    """Input schema for mcp.discover tool"""
    server_id: str = Field(..., description="MCP server identifier")
    endpoint: str = Field(..., description="MCP server endpoint URL")
    timeout_s: int = Field(30, description="Connection timeout in seconds")


class MCPCallIn(ToolInput):
    """Input schema for mcp.call tool"""
    server_id: str = Field(..., description="MCP server identifier") 
    tool: str = Field(..., description="Tool name to call")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    timeout_s: int = Field(30, description="Call timeout in seconds")


class MCPServerInfo(BaseModel):
    """MCP server information"""
    server_id: str
    endpoint: str
    status: str  # connected, disconnected, error
    capabilities: List[str] = Field(default_factory=list)
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    last_ping: Optional[str] = None
    error_message: Optional[str] = None


class MCPTool(BaseModel):
    """MCP tool definition"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    server_id: str


class MCPCallResult(BaseModel):
    """MCP tool call result"""
    server_id: str
    tool_name: str
    success: bool
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: int = 0
    cached: bool = False