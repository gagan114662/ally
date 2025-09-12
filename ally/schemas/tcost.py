from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market" 
    LIMIT = "limit"
    STOP = "stop"


class FillQuality(str, Enum):
    AGGRESSIVE = "aggressive"  # Market taking
    PASSIVE = "passive"       # Market making
    MIXED = "mixed"          # Partial fills with both


class TransactionCostConfig(BaseModel):
    """Configuration for transaction cost calculations"""
    commission_bps: float = Field(..., ge=0, le=1000, description="Commission in basis points")
    spread_capture_rate: float = Field(0.5, ge=0, le=1, description="Rate of spread capture (0=full cost, 1=full rebate)")
    market_impact_alpha: float = Field(0.6, ge=0, le=2, description="Market impact alpha parameter")
    market_impact_beta: float = Field(0.4, ge=0, le=1, description="Market impact beta (temporary impact)")
    slippage_tolerance_bps: float = Field(50, ge=0, le=500, description="Maximum allowed slippage")
    min_fill_size: int = Field(1, ge=1, description="Minimum fill size")
    
    
class MarketMicrostructure(BaseModel):
    """Market microstructure parameters"""
    symbol: str
    timestamp: datetime
    bid_price: float = Field(..., gt=0)
    ask_price: float = Field(..., gt=0)
    bid_size: int = Field(..., gt=0)
    ask_size: int = Field(..., gt=0)
    last_price: float = Field(..., gt=0)
    volume_1m: int = Field(0, ge=0, description="1-minute volume")
    volatility_1h: float = Field(0, ge=0, description="1-hour volatility")
    
    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points"""
        mid = (self.bid_price + self.ask_price) / 2
        return ((self.ask_price - self.bid_price) / mid) * 10000
    
    @property 
    def mid_price(self) -> float:
        """Mid price"""
        return (self.bid_price + self.ask_price) / 2


class Fill(BaseModel):
    """Individual fill record"""
    fill_id: str
    parent_order_id: str
    symbol: str
    side: OrderSide
    fill_price: float = Field(..., gt=0)
    fill_size: int = Field(..., gt=0)
    fill_time: datetime
    quality: FillQuality
    commission: float = Field(0, ge=0)
    
    # Market data at fill time
    bid_at_fill: float = Field(..., gt=0)
    ask_at_fill: float = Field(..., gt=0)
    volume_before: int = Field(0, ge=0)
    
    @property
    def notional(self) -> float:
        """Notional value of fill"""
        return self.fill_price * self.fill_size
    
    @property
    def mid_at_fill(self) -> float:
        """Mid price at fill time"""
        return (self.bid_at_fill + self.ask_at_fill) / 2


class TransactionCostAnalysis(BaseModel):
    """Complete transaction cost analysis for a set of fills"""
    config: TransactionCostConfig
    fills: List[Fill]
    analysis_time: datetime
    
    # Cost breakdown (in basis points)
    commission_cost_bps: float
    spread_cost_bps: float
    market_impact_bps: float
    slippage_bps: float
    total_cost_bps: float
    
    # Summary statistics
    total_notional: float
    total_commission: float
    fill_count: int
    avg_fill_size: float
    aggressive_fill_ratio: float
    
    # Execution quality metrics
    implementation_shortfall_bps: float
    volume_weighted_price: float
    price_improvement_bps: float