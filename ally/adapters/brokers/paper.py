# ally/adapters/brokers/paper.py
from __future__ import annotations
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from ally.schemas.broker import OrderIn, FillOut, PlaceOrderOut, Receipt, BrokerConfig

RUNS = Path("runs"); RUNS.mkdir(exist_ok=True)
RECEIPTS = RUNS / "receipts"; RECEIPTS.mkdir(exist_ok=True)

def _sha1_bytes(b:bytes)->str: return hashlib.sha1(b).hexdigest()

def place_order(order:OrderIn, cfg:BrokerConfig)->PlaceOrderOut:
    # DRY MODE ONLY for CI; live requires ALLY_LIVE=1 (guarded by tools layer)
    filled_qty = float(order.qty)
    px = float(order.price or 100.00)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

    fill = FillOut(ok=True, order_id=_sha1_bytes(now.encode())[:12],
                   filled_qty=filled_qty, avg_price=px, status="filled", ts=now)

    raw = {
      "endpoint":"place_order","symbol":order.symbol,"side":order.side,"qty":order.qty,
      "avg_price":px,"status":fill.status,"ts":now,"venue":"paper","session_id":cfg.session_id
    }
    raw_b = json.dumps(raw, separators=(",",":")).encode()
    sha1 = _sha1_bytes(raw_b)

    (RECEIPTS / f"{sha1}.json").write_bytes(raw_b)
    rec = Receipt(sha1=sha1, broker_venue="paper", endpoint="place_order",
                  ts=now, payload_size=len(raw_b), cost_cents=0, session_id=cfg.session_id)

    return PlaceOrderOut(ok=True, fill=fill, receipt=rec, mode="dry")