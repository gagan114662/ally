"""
MCP (Model Context Protocol) client implementation
"""

import json
import sqlite3
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..schemas.mcp import MCPServerInfo, MCPTool, MCPCallResult
from ..utils.io import ensure_dir


class MCPClient:
    """
    Minimal MCP HTTP/JSON client with schema introspection
    """
    
    def __init__(self, db_path: str = "data/mcp.sqlite"):
        """
        Initialize MCP client
        
        Args:
            db_path: Path to SQLite database for registry
        """
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._init_database()
        self._servers: Dict[str, MCPServerInfo] = {}
    
    def _init_database(self):
        """Initialize SQLite database for MCP registry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Servers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS servers (
                server_id TEXT PRIMARY KEY,
                endpoint TEXT NOT NULL,
                status TEXT NOT NULL,
                capabilities TEXT,  -- JSON
                last_ping TEXT,
                error_message TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tools table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tools (
                server_id TEXT,
                name TEXT,
                description TEXT,
                parameters TEXT,  -- JSON
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (server_id, name),
                FOREIGN KEY (server_id) REFERENCES servers(server_id)
            )
        ''')
        
        # Rate limits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS limits (
                server_id TEXT,
                tool_name TEXT,
                calls_per_minute INTEGER DEFAULT 60,
                calls_per_hour INTEGER DEFAULT 1000,
                last_reset TEXT,
                current_calls INTEGER DEFAULT 0,
                PRIMARY KEY (server_id, tool_name)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def discover_server(self, server_id: str, endpoint: str, timeout_s: int = 30) -> MCPServerInfo:
        """
        Discover MCP server capabilities and tools
        
        Args:
            server_id: Server identifier
            endpoint: Server endpoint URL
            timeout_s: Request timeout
            
        Returns:
            MCPServerInfo object
        """
        server_info = MCPServerInfo(
            server_id=server_id,
            endpoint=endpoint,
            status="disconnected",
            last_ping=datetime.now().isoformat()
        )
        
        try:
            # Try to connect and get server info
            describe_url = f"{endpoint.rstrip('/')}/describe"
            
            response = requests.get(
                describe_url,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Ally-MCP-Client/1.0"
                },
                timeout=timeout_s
            )
            
            if response.status_code == 200:
                data = response.json()
                
                server_info.status = "connected"
                server_info.capabilities = data.get("capabilities", [])
                server_info.tools = data.get("tools", [])
                
                # Store in database
                self._store_server_info(server_info)
                
            else:
                server_info.status = "error"
                server_info.error_message = f"HTTP {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.Timeout:
            server_info.status = "error"
            server_info.error_message = f"Connection timeout after {timeout_s}s"
        except requests.exceptions.ConnectionError:
            server_info.status = "error"
            server_info.error_message = "Connection failed - server unreachable"
        except Exception as e:
            server_info.status = "error"
            server_info.error_message = f"Discovery failed: {str(e)}"
        
        # Cache server info
        self._servers[server_id] = server_info
        return server_info
    
    def call_tool(self, server_id: str, tool: str, args: Dict[str, Any], 
                  timeout_s: int = 30) -> MCPCallResult:
        """
        Call a tool on an MCP server
        
        Args:
            server_id: Server identifier
            tool: Tool name
            args: Tool arguments
            timeout_s: Call timeout
            
        Returns:
            MCPCallResult object
        """
        start_time = time.time()
        
        result = MCPCallResult(
            server_id=server_id,
            tool_name=tool,
            success=False
        )
        
        try:
            # Get server info
            server_info = self._servers.get(server_id)
            if not server_info:
                # Try to load from database
                server_info = self._load_server_info(server_id)
                
            if not server_info or server_info.status != "connected":
                result.error = f"Server '{server_id}' not available"
                return result
            
            # Check rate limits
            if not self._check_rate_limit(server_id, tool):
                result.error = "Rate limit exceeded"
                return result
            
            # Make the call
            call_url = f"{server_info.endpoint.rstrip('/')}/call/{tool}"
            
            response = requests.post(
                call_url,
                json=args,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Ally-MCP-Client/1.0"
                },
                timeout=timeout_s
            )
            
            if response.status_code == 200:
                result.success = True
                result.result = response.json()
            else:
                result.error = f"HTTP {response.status_code}: {response.text[:200]}"
                
        except requests.exceptions.Timeout:
            result.error = f"Call timeout after {timeout_s}s"
        except requests.exceptions.RequestException as e:
            result.error = f"Request failed: {str(e)}"
        except Exception as e:
            result.error = f"Call failed: {str(e)}"
        
        result.duration_ms = int((time.time() - start_time) * 1000)
        
        # Update rate limit counters
        if result.success:
            self._update_rate_limit(server_id, tool)
        
        return result
    
    def list_servers(self) -> List[MCPServerInfo]:
        """List all registered servers"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT server_id, endpoint, status, capabilities, 
                   last_ping, error_message 
            FROM servers
        ''')
        
        servers = []
        for row in cursor.fetchall():
            server_info = MCPServerInfo(
                server_id=row[0],
                endpoint=row[1],
                status=row[2],
                capabilities=json.loads(row[3]) if row[3] else [],
                last_ping=row[4],
                error_message=row[5]
            )
            servers.append(server_info)
        
        conn.close()
        return servers
    
    def list_tools(self, server_id: str = None) -> List[MCPTool]:
        """List available tools"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if server_id:
            cursor.execute('''
                SELECT server_id, name, description, parameters
                FROM tools 
                WHERE server_id = ?
            ''', (server_id,))
        else:
            cursor.execute('''
                SELECT server_id, name, description, parameters
                FROM tools
            ''')
        
        tools = []
        for row in cursor.fetchall():
            tool = MCPTool(
                server_id=row[0],
                name=row[1],
                description=row[2],
                parameters=json.loads(row[3]) if row[3] else {}
            )
            tools.append(tool)
        
        conn.close()
        return tools
    
    def _store_server_info(self, server_info: MCPServerInfo):
        """Store server info in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store server
        cursor.execute('''
            INSERT OR REPLACE INTO servers 
            (server_id, endpoint, status, capabilities, last_ping, error_message, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            server_info.server_id,
            server_info.endpoint,
            server_info.status,
            json.dumps(server_info.capabilities),
            server_info.last_ping,
            server_info.error_message,
            datetime.now().isoformat()
        ))
        
        # Store tools
        cursor.execute('DELETE FROM tools WHERE server_id = ?', (server_info.server_id,))
        
        for tool_info in server_info.tools:
            cursor.execute('''
                INSERT INTO tools (server_id, name, description, parameters)
                VALUES (?, ?, ?, ?)
            ''', (
                server_info.server_id,
                tool_info.get("name", ""),
                tool_info.get("description", ""),
                json.dumps(tool_info.get("parameters", {}))
            ))
        
        conn.commit()
        conn.close()
    
    def _load_server_info(self, server_id: str) -> Optional[MCPServerInfo]:
        """Load server info from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT server_id, endpoint, status, capabilities,
                   last_ping, error_message
            FROM servers 
            WHERE server_id = ?
        ''', (server_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return MCPServerInfo(
            server_id=row[0],
            endpoint=row[1], 
            status=row[2],
            capabilities=json.loads(row[3]) if row[3] else [],
            last_ping=row[4],
            error_message=row[5]
        )
    
    def _check_rate_limit(self, server_id: str, tool_name: str) -> bool:
        """Check if call is within rate limits"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT calls_per_minute, calls_per_hour, last_reset, current_calls
            FROM limits
            WHERE server_id = ? AND tool_name = ?
        ''', (server_id, tool_name))
        
        row = cursor.fetchone()
        
        if not row:
            # Initialize rate limit entry
            cursor.execute('''
                INSERT INTO limits (server_id, tool_name, last_reset, current_calls)
                VALUES (?, ?, ?, 0)
            ''', (server_id, tool_name, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            return True
        
        calls_per_minute, calls_per_hour, last_reset, current_calls = row
        last_reset_dt = datetime.fromisoformat(last_reset)
        
        # Reset counters if needed
        if datetime.now() - last_reset_dt > timedelta(minutes=1):
            cursor.execute('''
                UPDATE limits 
                SET current_calls = 0, last_reset = ?
                WHERE server_id = ? AND tool_name = ?
            ''', (datetime.now().isoformat(), server_id, tool_name))
            current_calls = 0
        
        conn.commit()
        conn.close()
        
        # Check limits (simplified - just per minute for now)
        return current_calls < calls_per_minute
    
    def _update_rate_limit(self, server_id: str, tool_name: str):
        """Update rate limit counters after successful call"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE limits 
            SET current_calls = current_calls + 1
            WHERE server_id = ? AND tool_name = ?
        ''', (server_id, tool_name))
        
        conn.commit()
        conn.close()


# Mock MCP server for testing
class MockMCPServer:
    """Mock MCP server for testing purposes"""
    
    def __init__(self, port: int = 9000):
        self.port = port
        self.tools = {
            "calculate": {
                "name": "calculate",
                "description": "Perform basic calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    }
                }
            },
            "random_fact": {
                "name": "random_fact",
                "description": "Get a random fact",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "Fact category"}
                    }
                }
            }
        }
    
    def get_describe_response(self) -> Dict[str, Any]:
        """Get mock /describe endpoint response"""
        return {
            "capabilities": ["tools", "resources"],
            "tools": list(self.tools.values())
        }
    
    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock tool call"""
        if tool_name == "calculate":
            expression = args.get("expression", "1+1")
            try:
                # Simple eval for demo (unsafe in production!)
                result = eval(expression)
                return {"result": result, "expression": expression}
            except:
                return {"error": "Invalid expression"}
        
        elif tool_name == "random_fact":
            category = args.get("category", "general")
            facts = {
                "general": "The shortest war in history lasted only 38-45 minutes.",
                "science": "A single cloud can weigh more than a million pounds.",
                "math": "The number π appears in many unexpected places in mathematics."
            }
            return {
                "fact": facts.get(category, facts["general"]),
                "category": category
            }
        
        return {"error": f"Unknown tool: {tool_name}"}