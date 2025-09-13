"""Receipt schemas for M-RealData Gate and M-Receipts-Everywhere provenance tracking."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Receipt(BaseModel):
    """Data receipt for anti-fabrication and provenance tracking."""
    vendor: str = Field(..., description="Data provider name")
    endpoint: str = Field(..., description="API endpoint called")
    params: Dict[str, Any] = Field(..., description="Request parameters (no secrets)")
    ts_iso: str = Field(..., description="ISO timestamp of fetch")
    content_sha1: str = Field(..., description="SHA1 hash of response payload")
    bytes: int = Field(..., description="Size of response in bytes")
    cost_cents: Optional[int] = Field(None, description="Estimated cost in cents")


class QuorumVerdict(BaseModel):
    """Cross-provider verification result for data integrity."""
    providers: List[str] = Field(..., description="Providers participating in quorum")
    agreement: bool = Field(..., description="Whether providers agree within tolerance")
    tolerance_pct: float = Field(..., description="Maximum allowed deviation")
    max_deviation_pct: float = Field(..., description="Actual maximum deviation found")
    content_sha1s: List[str] = Field(..., description="Content hashes from each provider")


class LiveDataSession(BaseModel):
    """Session tracking for live data access with budget controls."""
    session_id: str = Field(..., description="Unique session identifier")
    budget_cents: int = Field(..., description="Budget limit in cents")
    spent_cents: int = Field(default=0, description="Amount spent so far")
    receipts: List[Receipt] = Field(default_factory=list, description="Session receipts")
    ts_start: str = Field(..., description="Session start timestamp")
    ts_end: Optional[str] = Field(None, description="Session end timestamp")


class LiveAccessError(Exception):
    """Exception raised when live data access is denied or blocked."""
    pass