# ally/schemas/broker.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict
from datetime import datetime

Side = Literal["buy","sell"]
Venue = Literal["paper","ibkr","alpaca","binance"]

class BrokerConfig(BaseModel):
    venue: Venue = "paper"
    live: bool = False
    max_cost_cents: int = 500 # per-session budget guard
    session_id: str = "SESSION_MLB_DRY"

class OrderIn(BaseModel):
    symbol: str
    side: Side
    qty: float
    price: Optional[float] = None
    type: Literal["market","limit"] = "market"
    client_order_id: Optional[str] = None

class FillOut(BaseModel):
    ok: bool
    order_id: str
    filled_qty: float
    avg_price: float
    status: Literal["filled","rejected","partial","accepted"]
    ts: str

class Receipt(BaseModel):
    sha1: str
    broker_venue: Venue
    endpoint: str
    ts: str
    payload_size: int
    cost_cents: int = 0
    session_id: str

class PlaceOrderOut(BaseModel):
    ok: bool
    fill: FillOut
    receipt: Receipt
    mode: Literal["dry","live"] = "dry"