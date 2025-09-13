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
    
    try:
        from . import memory  # Import memory tools
    except ImportError:
        pass
    
    try:
        from . import reporting  # Import reporting tools
    except ImportError:
        pass
    
    try:
        from . import router  # Import router tools
    except ImportError:
        pass
    
    try:
        from . import runtime  # Import runtime tools
    except ImportError:
        pass

    try:
        from . import broker  # Import broker tools
    except ImportError:
        pass

    try:
        from . import autopilot  # Import autopilot tools
    except ImportError:
        pass

    try:
        from . import grid  # Import grid tools
    except ImportError:
        pass

    try:
        from . import ops  # Import ops tools
    except ImportError:
        pass

    try:
        from . import health  # Import health tools
    except ImportError:
        pass

# Auto-import tools when module is imported
_import_all_tools()

# Register QuantConnect tools
try:
    from .qc_templates import qc_generate_python as _qc_gen, qc_list_templates as _qc_list
    from .qc_lint import qc_lint as _qc_lint
    from .qc_lean import qc_smoke_run as _qc_smoke, qc_validate_project_structure as _qc_validate
    from .qc_runtime_guard import qc_classify_error as _qc_classify
    from .qc_autorepair import qc_autorepair as _qc_repair
    from .qc_universe import (qc_universe_check as _qc_check, qc_normalize_symbols as _qc_norm, 
                              qc_universe_guard as _qc_guard, qc_history_smoke as _qc_history,
                              qc_resolution_matrix as _qc_matrix, canonical_equity as _qc_canonical)
    from .qc_asserts import qc_inject_asserts as _qc_inject, qc_validate_asserts as _qc_validate
    TOOL_REGISTRY["qc.generate_python"] = _qc_gen
    TOOL_REGISTRY["qc.list_templates"] = _qc_list
    TOOL_REGISTRY["qc.lint"] = _qc_lint
    TOOL_REGISTRY["qc.smoke_run"] = _qc_smoke
    TOOL_REGISTRY["qc.validate_project"] = _qc_validate
    TOOL_REGISTRY["qc.classify_error"] = _qc_classify
    TOOL_REGISTRY["qc.autorepair"] = _qc_repair
    TOOL_REGISTRY["qc.universe_check"] = _qc_check
    TOOL_REGISTRY["qc.normalize_symbols"] = _qc_norm
    TOOL_REGISTRY["qc.universe_guard"] = _qc_guard
    TOOL_REGISTRY["qc.history_smoke"] = _qc_history
    TOOL_REGISTRY["qc.resolution_matrix"] = _qc_matrix
    TOOL_REGISTRY["qc.canonical_equity"] = _qc_canonical
    TOOL_REGISTRY["qc.inject_asserts"] = _qc_inject
    TOOL_REGISTRY["qc.validate_asserts"] = _qc_validate
except ImportError:
    pass

# Register broker tools
try:
    from .broker import place_order as _broker_place
    TOOL_REGISTRY["broker.place_order"] = _broker_place
except ImportError:
    pass

# Register autopilot tools
try:
    from .autopilot import run as _autopilot_run
    TOOL_REGISTRY["autopilot.run"] = _autopilot_run
except ImportError:
    pass

# Register grid tools
try:
    from .grid import run as _grid_run
    TOOL_REGISTRY["grid.run"] = _grid_run
except ImportError:
    pass

# Register ops tools
try:
    from .ops import register as _register_ops
    _register_ops()
except ImportError:
    pass

# Register FDR tools
try:
    from .fdr import fdr_gate as _fdr_gate
    TOOL_REGISTRY["fdr.gate"] = _fdr_gate
except ImportError:
    pass

# Register Capacity tools
try:
    from .capacity import capacity_curve as _capacity_curve
    TOOL_REGISTRY["capacity.estimate"] = _capacity_curve
except ImportError:
    pass

# Register Regimes tools
try:
    from .regimes import regime_gate as _regime_gate
    TOOL_REGISTRY["regimes.gate"] = _regime_gate
except ImportError:
    pass

# Register ops bridge
try:
    from .ops_bridge import register as _register_ops_bridge
    _register_ops_bridge()
except ImportError:
    pass

# Register Jules tools
try:
    from .jules import jules_help as _jules_help
    TOOL_REGISTRY["jules.help"] = _jules_help
except ImportError:
    pass