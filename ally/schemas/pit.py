# ally/schemas/pit.py
"""Point-in-Time (PIT) schemas for universe membership and corporate actions.

Eliminates survivorship bias and look-ahead bias by providing historical
snapshots of tradable universes and delisting events.
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class PITUniverseRow(BaseModel):
    """Point-in-time universe membership record.
    
    Represents whether a symbol was tradable/investable at a specific date.
    Used to reconstruct historical universes without survivorship bias.
    """
    date: str = Field(description="Date in YYYY-MM-DD format")
    symbol: str = Field(description="Symbol identifier")
    active: bool = Field(description="True if symbol was tradable on this date")
    market_cap: Optional[float] = Field(default=None, description="Market cap in USD (if available)")
    sector: Optional[str] = Field(default=None, description="Sector classification")
    exchange: Optional[str] = Field(default=None, description="Primary exchange")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        extra = "forbid"


class ActionRow(BaseModel):
    """Corporate action record for delisting events and adjustments.
    
    Captures events that affect historical returns and universe membership,
    preventing look-ahead bias in backtests.
    """
    date: str = Field(description="Effective date in YYYY-MM-DD format")
    symbol: str = Field(description="Symbol affected by action")
    action: str = Field(description="Action type: delisting, merger, spinoff, split, dividend")
    adjustment_factor: Optional[float] = Field(default=None, description="Price adjustment factor")
    final_price: Optional[float] = Field(default=None, description="Final trading price before delisting")
    reason: Optional[str] = Field(default=None, description="Reason for action (bankruptcy, merger, etc.)")
    successor_symbol: Optional[str] = Field(default=None, description="Symbol after merger/spinoff")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional action metadata")
    
    class Config:
        extra = "forbid"


class PITSnapshot(BaseModel):
    """Complete point-in-time universe snapshot.
    
    Contains all active symbols and their metadata for a specific date,
    used for honest backtesting without survivorship bias.
    """
    date: str = Field(description="Snapshot date in YYYY-MM-DD format")
    symbols: List[str] = Field(description="List of active symbols on this date")
    universe_size: int = Field(description="Total number of active symbols")
    actions: List[ActionRow] = Field(default_factory=list, description="Corporate actions on this date")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Snapshot metadata")
    
    class Config:
        extra = "forbid"
    
    def is_active(self, symbol: str) -> bool:
        """Check if symbol was active in this snapshot."""
        return symbol in self.symbols
    
    def get_delistings(self) -> List[ActionRow]:
        """Get delisting actions from this snapshot."""
        return [action for action in self.actions if action.action == "delisting"]


class PITQuery(BaseModel):
    """Query parameters for point-in-time data retrieval."""
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")
    symbols: Optional[List[str]] = Field(default=None, description="Filter by specific symbols")
    min_market_cap: Optional[float] = Field(default=None, description="Minimum market cap filter")
    sectors: Optional[List[str]] = Field(default=None, description="Filter by sectors")
    include_actions: bool = Field(default=True, description="Include corporate actions")
    
    class Config:
        extra = "forbid"