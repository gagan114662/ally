"""
Base schemas for Ally tool system
Provides uniform ToolResult and Meta structures for all tools
"""

from pydantic import BaseModel, Field
from datetime import datetime, timezone

# Pydantic v2 compatibility check
try:
    from pydantic import field_serializer
    HAS_V2 = True
except ImportError:
    HAS_V2 = False
from typing import Any, Dict, List, Optional
from enum import Enum


class ToolStatus(str, Enum):
    """Tool execution status"""
    SUCCESS = "success"
    ERROR = "error" 
    WARNING = "warning"
    TIMEOUT = "timeout"


class Meta(BaseModel):
    """Metadata for tool execution"""
    ts: datetime = Field(default_factory=datetime.utcnow)
    duration_ms: int = 0
    provenance: Dict[str, Any] = Field(default_factory=dict)
    cost: Dict[str, Any] = Field(default_factory=dict) 
    warnings: List[str] = Field(default_factory=list)
    inputs_hash: Optional[str] = None
    code_hash: Optional[str] = None
    receipt_hash: Optional[str] = None  # SHA-1 hash of raw payload for receipts
    
    if HAS_V2:
        @field_serializer('ts')
        def _ser_ts(self, dt: datetime) -> str:
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ToolResult(BaseModel):
    """Uniform result structure for all Ally tools"""
    ok: bool = True
    status: ToolStatus = ToolStatus.SUCCESS
    data: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    meta: Meta = Field(default_factory=Meta)
    
    @classmethod
    def success(cls, data: Dict[str, Any], warnings: List[str] = None) -> "ToolResult":
        """Create successful result"""
        meta = Meta()
        if warnings:
            meta.warnings = warnings
        return cls(
            ok=True,
            status=ToolStatus.SUCCESS,
            data=data,
            meta=meta
        )
    
    @classmethod
    def error(cls, errors: List[str], data: Dict[str, Any] = None) -> "ToolResult":
        """Create error result"""
        return cls(
            ok=False,
            status=ToolStatus.ERROR,
            data=data or {},
            errors=errors,
            meta=Meta()
        )
    
    @classmethod
    def warning(cls, data: Dict[str, Any], warnings: List[str]) -> "ToolResult":
        """Create warning result (ok=True but with warnings)"""
        meta = Meta(warnings=warnings)
        return cls(
            ok=True,
            status=ToolStatus.WARNING,
            data=data,
            meta=meta
        )
    
    def store_receipt(self, tool_name: str, inputs: Dict[str, Any], 
                     raw_payload: Any = None) -> str:
        """
        Store receipt for this tool result
        
        Args:
            tool_name: Name of the tool that generated this result
            inputs: Tool input parameters
            raw_payload: Optional raw payload (defaults to self.data)
        
        Returns:
            Receipt hash
        """
        # Import here to avoid circular imports
        from ..utils.receipts import store_tool_receipt
        
        if raw_payload is None:
            raw_payload = self.data
            
        receipt_hash = store_tool_receipt(tool_name, inputs, raw_payload)
        self.meta.receipt_hash = receipt_hash
        return receipt_hash


class ToolInput(BaseModel):
    """Base class for tool input validation"""
    pass


class ToolConfig(BaseModel):
    """Configuration for tool behavior"""
    timeout_s: int = 30
    max_retries: int = 3
    cache_enabled: bool = True
    deterministic: bool = True
    seed: Optional[int] = 42