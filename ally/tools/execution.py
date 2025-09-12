"""
Execution Tools for Ally
Order placement, cancellation, and amendment via paper broker
"""

from ..tools import register
from ..schemas.base import ToolResult, Meta
from ..schemas.exec import PlaceOrderIn, CancelOrderIn, AmendOrderIn, ExecutionReport
from ..adapters.brokers.paper import _paper_broker
from ..utils.hashing import hash_inputs, hash_code
from ..utils.serialization import convert_timestamps


@register("exec.place_order")
def place_order(**kwargs) -> ToolResult:
    """
    Place an order via paper broker
    
    Args:
        **kwargs: Parameters matching PlaceOrderIn schema
        
    Returns:
        ToolResult with ExecutionReport
    """
    try:
        params = PlaceOrderIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        # Place order via paper broker
        execution_report = _paper_broker.place_order(params)
        
        # Create metadata
        meta = Meta(
            tool="exec.place_order",
            version="1.0.0",
            timestamp=None,
            provenance={
                "inputs_hash": hash_inputs(params.model_dump()),
                "code_hash": hash_code(place_order),
                "broker": "paper",
                "order_id": execution_report.broker_order_id
            }
        )
        
        result = ToolResult(
            ok=True,
            data=execution_report.model_dump(),
            meta=meta.model_dump()
        )
        
        return convert_timestamps(result)
        
    except Exception as e:
        return ToolResult.error([f"Execution error: {e}"])


@register("exec.cancel_order")
def cancel_order(**kwargs) -> ToolResult:
    """
    Cancel an order via paper broker
    
    Args:
        **kwargs: Parameters matching CancelOrderIn schema
        
    Returns:
        ToolResult with ExecutionReport
    """
    try:
        params = CancelOrderIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        # Cancel order via paper broker
        execution_report = _paper_broker.cancel_order(params)
        
        # Create metadata
        meta = Meta(
            tool="exec.cancel_order",
            version="1.0.0",
            timestamp=None,
            provenance={
                "inputs_hash": hash_inputs(params.model_dump()),
                "code_hash": hash_code(cancel_order),
                "broker": "paper",
                "order_id": execution_report.broker_order_id
            }
        )
        
        result = ToolResult(
            ok=True,
            data=execution_report.model_dump(),
            meta=meta.model_dump()
        )
        
        return convert_timestamps(result)
        
    except Exception as e:
        return ToolResult.error([f"Cancellation error: {e}"])


@register("exec.amend_order")
def amend_order(**kwargs) -> ToolResult:
    """
    Amend an order via paper broker
    
    Args:
        **kwargs: Parameters matching AmendOrderIn schema
        
    Returns:
        ToolResult with ExecutionReport
    """
    try:
        params = AmendOrderIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        # Amend order via paper broker
        execution_report = _paper_broker.amend_order(params)
        
        # Create metadata
        meta = Meta(
            tool="exec.amend_order",
            version="1.0.0",
            timestamp=None,
            provenance={
                "inputs_hash": hash_inputs(params.model_dump()),
                "code_hash": hash_code(amend_order),
                "broker": "paper",
                "order_id": execution_report.broker_order_id
            }
        )
        
        result = ToolResult(
            ok=True,
            data=execution_report.model_dump(),
            meta=meta.model_dump()
        )
        
        return convert_timestamps(result)
        
    except Exception as e:
        return ToolResult.error([f"Amendment error: {e}"])


@register("exec.reset_broker")
def reset_broker(**kwargs) -> ToolResult:
    """
    Reset paper broker state (for testing)
    
    Returns:
        ToolResult with success status
    """
    try:
        _paper_broker.reset()
        
        meta = Meta(
            tool="exec.reset_broker",
            version="1.0.0",
            timestamp=None,
            provenance={
                "inputs_hash": hash_inputs({}),
                "code_hash": hash_code(reset_broker),
                "broker": "paper"
            }
        )
        
        result = ToolResult(
            ok=True,
            data={"status": "reset", "next_order_id": _paper_broker.next_order_id},
            meta=meta.model_dump()
        )
        
        return convert_timestamps(result)
        
    except Exception as e:
        return ToolResult.error([f"Reset error: {e}"])