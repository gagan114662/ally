"""
Receipt schemas for M-RealData Gate system
Provides data attestation and anti-fabrication guarantees
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class Receipt(BaseModel):
    """Attestation receipt for live data fetch"""
    vendor: str = Field(..., description="Data provider name (polygon, alphavantage, etc.)")
    endpoint: str = Field(..., description="API endpoint called")
    params: Dict[str, Any] = Field(..., description="Request parameters (no secrets)")
    ts_iso: str = Field(..., description="ISO timestamp of fetch")
    content_sha1: str = Field(..., description="SHA1 hash of response payload")
    bytes: int = Field(..., description="Size of response in bytes")
    cost_cents: Optional[int] = Field(None, description="Estimated cost in cents")
    

class QuorumVerdict(BaseModel):
    """Result of cross-provider agreement check"""
    members: List[str] = Field(..., description="Vendor names in quorum")
    metric: str = Field(..., description="Comparison metric (close, volume, etc.)")
    tolerance_bps: float = Field(..., description="Allowed variance in basis points")
    ok: bool = Field(..., description="Whether quorum agrees within tolerance")
    measurements: List[float] = Field(..., description="Values from each vendor")
    variance_bps: Optional[float] = Field(None, description="Actual variance in basis points")


class LiveDataSession(BaseModel):
    """Tracking for a live data session"""
    session_id: str = Field(..., description="Unique session identifier")
    started_at: str = Field(..., description="Session start timestamp")
    budget_cents: int = Field(..., description="Maximum allowed cost")
    spent_cents: int = Field(0, description="Cumulative cost so far")
    receipts: List[str] = Field(default_factory=list, description="Receipt SHA1s in this session")
    

class LiveAccessError(Exception):
    """Raised when live data access is denied"""
    pass


class BudgetExceededError(Exception):
    """Raised when session budget is exceeded"""
    pass


class QuorumFailureError(Exception):
    """Raised when providers disagree beyond tolerance"""
    pass