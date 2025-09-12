"""
Risk Management Schemas for Ally
Policy-driven risk checks for positions and orders
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any


class RiskCheckIn(BaseModel):
    """Input for risk limit checks"""
    positions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Current positions: [{'symbol':'BTCUSDT','qty':1.2,'price':30000.0}]"
    )
    orders: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pending and incoming orders"
    )
    policy_yaml: str = Field(
        ...,
        description="YAML policy defining risk limits"
    )
    equity: float = Field(
        default=100000.0,
        description="Account equity for leverage calculation"
    )
    prices: Dict[str, float] = Field(
        default_factory=dict,
        description="Symbol to mark price mapping for exposure calculation"
    )


class RiskViolation(BaseModel):
    """A single risk policy violation"""
    code: str = Field(..., description="Violation code (e.g., LEVERAGE_LIMIT)")
    severity: Literal["hard", "soft"] = Field(..., description="Violation severity")
    message: str = Field(..., description="Human-readable violation message")
    subject: Dict[str, Any] = Field(
        default_factory=dict,
        description="Details about the violation (metric, value, limit)"
    )


class RiskCheckOut(BaseModel):
    """Output from risk limit checks"""
    allow: bool = Field(..., description="Whether the action is allowed (no hard violations)")
    violations: List[RiskViolation] = Field(
        default_factory=list,
        description="List of policy violations detected"
    )
    audit_hash: str = Field(..., description="Deterministic hash for reproducibility")