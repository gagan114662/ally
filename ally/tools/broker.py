# ally/tools/broker.py
from __future__ import annotations
import os
from typing import Dict, Any
from ally.schemas.broker import BrokerConfig, OrderIn, PlaceOrderOut
from ally.adapters.brokers import paper
from ally.utils.killswitch import evaluate_killswitch, KillSwitchConfig
from ally.schemas.base import ToolResult

def _live_guard(cfg:BrokerConfig)->None:
    if cfg.live and os.getenv("ALLY_LIVE") != "1":
        raise RuntimeError("live_denied: set ALLY_LIVE=1 + live=True to enable live broker")

def place_order(symbol:str, side:str, qty:float, price:float|None=None, live:bool=False, venue:str="paper", session_id:str="SESSION_MLB") -> ToolResult:
    cfg = BrokerConfig(venue=venue, live=live, session_id=session_id)
    _live_guard(cfg)
    out: PlaceOrderOut = paper.place_order(OrderIn(symbol=symbol, side=side, qty=qty, price=price), cfg)
    # demo killswitch eval in dry
    ks_tripped = evaluate_killswitch(slippage_bps=10, latency_ms=100, loss_bps=10, cfg=KillSwitchConfig())
    data = {
        "ok": out.ok, "fill": out.fill.dict(), "receipt": out.receipt.dict(),
        "mode": out.mode, "killswitch_tripped": ks_tripped
    }
    return ToolResult(ok=True, data=data)