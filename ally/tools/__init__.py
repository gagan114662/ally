"""
Ally tools registry and decorators
"""

import time
import functools
from typing import Callable, Dict, Any
from ..schemas.base import ToolResult, Meta
from ..utils.hashing import hash_inputs, hash_code
from ..utils.audit import AuditLogger

# Global tool registry
TOOL_REGISTRY: Dict[str, Callable] = {}

# Global audit logger
audit_logger = AuditLogger()


def register(name: str):
    """
    Decorator to register a tool in the global registry
    
    Args:
        name: Tool name (e.g., "web.fetch", "data.load_ohlcv")
    """
    def decorator(func: Callable) -> Callable:
        TOOL_REGISTRY[name] = func
        
        @functools.wraps(func)
        def wrapper(**kwargs) -> ToolResult:
            start_time = time.time()
            
            # Create provenance info
            inputs_hash = hash_inputs(kwargs)
            code_hash = hash_code(func)
            
            try:
                # Execute the tool
                result = func(**kwargs)
                
                # Ensure result is a ToolResult
                if not isinstance(result, ToolResult):
                    result = ToolResult.success({"result": result})
                
                # Add timing and provenance
                duration_ms = int((time.time() - start_time) * 1000)
                result.meta.duration_ms = duration_ms
                result.meta.inputs_hash = inputs_hash
                result.meta.code_hash = code_hash
                result.meta.provenance = {
                    "tool_name": name,
                    "inputs": kwargs,
                    "version": "1.0"
                }
                
                # Log execution
                audit_logger.log_tool_execution(name, kwargs, result, code_hash)
                
                return result
                
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                result = ToolResult.error([str(e)])
                result.meta.duration_ms = duration_ms
                result.meta.inputs_hash = inputs_hash
                result.meta.code_hash = code_hash
                
                # Log failed execution
                audit_logger.log_tool_execution(name, kwargs, result, code_hash)
                
                return result
        
        return wrapper
    return decorator


def get_tool(name: str) -> Callable:
    """
    Get tool by name from registry
    
    Args:
        name: Tool name
        
    Returns:
        Tool function
        
    Raises:
        KeyError: If tool not found
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Tool '{name}' not found in registry")
    return TOOL_REGISTRY[name]


def list_tools() -> Dict[str, str]:
    """
    List all registered tools
    
    Returns:
        Dictionary mapping tool names to their docstrings
    """
    tools = {}
    for name, func in TOOL_REGISTRY.items():
        tools[name] = func.__doc__ or "No description available"
    return tools


def execute_tool(name: str, **kwargs) -> ToolResult:
    """
    Execute a tool by name with given arguments
    
    Args:
        name: Tool name
        **kwargs: Tool arguments
        
    Returns:
        ToolResult from the tool execution
    """
    tool_func = get_tool(name)
    return tool_func(**kwargs)


# Import all tool modules to register them
def _import_all_tools():
    """Import all tool modules to ensure they're registered"""
    try:
        from . import web  # Import web tools
    except ImportError:
        pass
    
    try:
        from . import mcp  # Import MCP tools
    except ImportError:
        pass
    
    try:
        from . import data  # Import data tools
    except ImportError:
        pass
    
    try:
        from . import features  # Import features tools
    except ImportError:
        pass
    
    try:
        from . import bt  # Import backtest tools
    except ImportError:
        pass
    
    try:
        from . import cv  # Import computer vision tools
    except ImportError:
        pass
    
    try:
        from . import nlp  # Import NLP tools
    except ImportError:
        pass
    
    try:
        from . import risk  # Import risk tools
    except ImportError:
        pass
    
    try:
        from . import execution  # Import execution tools
    except ImportError:
        pass

# Auto-import tools when module is imported
_import_all_tools()