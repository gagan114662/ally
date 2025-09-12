from __future__ import annotations
import os, json, hashlib
from datetime import datetime, timedelta
from typing import Any, Dict
from ally.schemas.base import ToolResult, Meta
from ally.providers.polygon import PolygonProvider
from ally.providers.alpha_vantage import AlphaVantageProvider
from ally.providers.finnhub import FinnhubProvider
from ally.providers.fred import FredProvider
from ally.providers.quandl import QuandlProvider

PROVIDERS = {
    "polygon": PolygonProvider,
    "alphavantage": AlphaVantageProvider,
    "finnhub": FinnhubProvider,
    "fred": FredProvider,
    "quandl": QuandlProvider,
}

def data_fetch_live(source: str, symbol: str, interval: str, start: str | None=None, end: str | None=None,
                    use_cache: bool=True, live: bool=False) -> ToolResult:
    src = source.lower()
    if src not in PROVIDERS:
        return ToolResult(ok=False, data={"error": f"unknown provider '{source}'"}, errors=["unknown_provider"],
                          meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name":"data.fetch_live"}))

    # live guard (offline by default)
    if not live and os.getenv("ALLY_LIVE","0") != "1":
        return ToolResult(ok=False, data={"error":"live disabled; set live=true or ALLY_LIVE=1"}, errors=["live_disabled"],
                          meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name":"data.fetch_live"}))

    provider = PROVIDERS[src]()
    resp = provider.get_ohlcv(symbol=symbol, interval=interval, start=start, end=end)

    meta = Meta(ts=datetime.utcnow(), duration_ms=0, provenance={
        "tool_name":"data.fetch_live","provider":src,"cached":resp.cached,"source":resp.source
    })
    return ToolResult(ok=resp.ok, data={"frame": resp.data}, errors=[] if resp.ok else [resp.data], meta=meta)