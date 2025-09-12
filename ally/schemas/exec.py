"""
Execution Schemas for Ally
Order placement, cancellation, amendment, and execution reports
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

Side = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
TIF = Literal["gtc", "ioc"]  # Good Till Cancel, Immediate Or Cancel
OrderStatus = Literal["accepted", "working", "partially_filled", "filled", "canceled", "rejected"]


class PlaceOrderIn(BaseModel):
    """Input for placing an order"""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    side: Side = Field(..., description="Order side: buy or sell")
    qty: float = Field(..., gt=0, description="Order quantity")
    type: OrderType = Field(..., description="Order type: market or limit")
    limit_price: Optional[float] = Field(None, gt=0, description="Limit price for limit orders")
    tif: TIF = Field(default="gtc", description="Time in force: gtc or ioc")
    client_order_id: Optional[str] = Field(None, description="Client-provided order ID")
    price: Optional[float] = Field(None, gt=0, description="L1 price input for simulation")
    slippage_bps: float = Field(default=0.0, ge=0, description="Slippage in basis points")
    latency_ms: int = Field(default=0, ge=0, description="Simulated latency in milliseconds")
    liquidity_per_tick: float = Field(default=1.0, gt=0, description="Max fills per match step")


class CancelOrderIn(BaseModel):
    """Input for canceling an order"""
    broker_order_id: str = Field(..., description="Broker-assigned order ID to cancel")


class AmendOrderIn(BaseModel):
    """Input for amending an order"""
    broker_order_id: str = Field(..., description="Broker-assigned order ID to amend")
    limit_price: Optional[float] = Field(None, gt=0, description="New limit price")
    qty: Optional[float] = Field(None, gt=0, description="New quantity")


class Fill(BaseModel):
    """A single execution fill"""
    price: float = Field(..., gt=0, description="Fill price")
    qty: float = Field(..., gt=0, description="Fill quantity")
    ts: str = Field(..., description="Fill timestamp in ISO-8601 with Z suffix")


class ExecutionReport(BaseModel):
    """Execution report for an order"""
    broker_order_id: str = Field(..., description="Broker-assigned order ID")
    client_order_id: Optional[str] = Field(None, description="Client-provided order ID")
    symbol: str = Field(..., description="Trading symbol")
    side: Side = Field(..., description="Order side")
    type: OrderType = Field(..., description="Order type")
    status: OrderStatus = Field(..., description="Current order status")
    avg_price: float = Field(..., ge=0, description="Average fill price (0 if unfilled)")
    filled_qty: float = Field(..., ge=0, description="Total filled quantity")
    remaining_qty: float = Field(..., ge=0, description="Remaining unfilled quantity")
    fills: List[Fill] = Field(default_factory=list, description="List of execution fills")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")