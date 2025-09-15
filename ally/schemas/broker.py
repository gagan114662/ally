"""
Broker schemas for Ally - unified order/fill/account/position models
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
from datetime import datetime
from enum import Enum
from ..schemas.base import ToolInput


class OrderSide(str, Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order status enumeration"""
    NEW = "new"
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(str, Enum):
    """Time in force enumeration"""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class PlaceOrderIn(ToolInput):
    """Input schema for placing orders"""
    symbol: str = Field(..., description="Trading symbol (e.g., AAPL)")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    qty: int = Field(..., description="Order quantity (positive integer)")
    type: OrderType = Field(OrderType.MARKET, description="Order type")
    limit_price: Optional[float] = Field(None, description="Limit price (required for limit orders)")
    stop_price: Optional[float] = Field(None, description="Stop price (required for stop orders)")
    time_in_force: TimeInForce = Field(TimeInForce.DAY, description="Time in force")
    client_order_id: Optional[str] = Field(None, description="Client-assigned order ID")
    backend: str = Field("simulator", description="Broker backend (qc_paper, simulator)")
    live: bool = Field(False, description="Enable live trading (requires ALLY_LIVE=1)")


class CancelOrderIn(ToolInput):
    """Input schema for canceling orders"""
    order_id: str = Field(..., description="Order ID to cancel")
    client_order_id: Optional[str] = Field(None, description="Client order ID (if available)")
    backend: str = Field("simulator", description="Broker backend (qc_paper, simulator)")
    live: bool = Field(False, description="Enable live trading (requires ALLY_LIVE=1)")


class GetAccountIn(ToolInput):
    """Input schema for getting account info"""
    backend: str = Field("simulator", description="Broker backend (qc_paper, simulator)")
    live: bool = Field(False, description="Enable live trading (requires ALLY_LIVE=1)")


class GetPositionsIn(ToolInput):
    """Input schema for getting positions"""
    backend: str = Field("simulator", description="Broker backend (qc_paper, simulator)")
    live: bool = Field(False, description="Enable live trading (requires ALLY_LIVE=1)")


class GetOrdersIn(ToolInput):
    """Input schema for getting orders"""
    status: Optional[OrderStatus] = Field(None, description="Filter by order status")
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    limit: int = Field(100, description="Maximum orders to return")
    backend: str = Field("simulator", description="Broker backend (qc_paper, simulator)")
    live: bool = Field(False, description="Enable live trading (requires ALLY_LIVE=1)")


# Output schemas
class Order(BaseModel):
    """Order representation"""
    order_id: str
    client_order_id: Optional[str] = None
    symbol: str
    side: OrderSide
    qty: int
    type: OrderType
    status: OrderStatus
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    filled_qty: int = 0
    avg_fill_price: Optional[float] = None
    submitted_at: str  # ISO timestamp
    updated_at: str    # ISO timestamp
    provider: str      # "qc_paper", "simulator"
    receipt_hash: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Fill(BaseModel):
    """Fill representation"""
    fill_id: str
    order_id: str
    client_order_id: Optional[str] = None
    symbol: str
    side: OrderSide
    qty: int
    price: float
    timestamp: str     # ISO timestamp
    provider: str      # "qc_paper", "simulator"
    receipt_hash: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Position(BaseModel):
    """Position representation"""
    symbol: str
    qty: int           # Net position (positive=long, negative=short, zero=flat)
    avg_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    cost_basis: Optional[float] = None
    updated_at: str    # ISO timestamp
    provider: str      # "qc_paper", "simulator"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Account(BaseModel):
    """Account representation"""
    account_id: str
    cash: float
    buying_power: float
    total_value: float
    day_pnl: Optional[float] = None
    total_pnl: Optional[float] = None
    updated_at: str    # ISO timestamp
    provider: str      # "qc_paper", "simulator"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BrokerSession(BaseModel):
    """Broker session info"""
    session_id: str
    backend: str       # "qc_paper", "simulator"
    project_slug: Optional[str] = None  # For QC paper sessions
    symbols: List[str]
    started_at: str    # ISO timestamp
    status: Literal["active", "stopped", "error"]
    metadata: Dict[str, Any] = Field(default_factory=dict)