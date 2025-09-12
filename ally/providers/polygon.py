import os
from datetime import timedelta
from typing import Any, Dict
from ally.providers.base import DataProvider, ProviderResponse
from ally.utils.cache import SqliteCache

class PolygonProvider(DataProvider):
    name = "polygon"

    def __init__(self, api_key: str | None = None, cache: SqliteCache | None = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self.cache = cache or SqliteCache()

    def _normalize(self, recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Expect [{"timestamp": "...", "open":..., "high":..., "low":..., "close":..., "volume":...}, ...]
        return recs

    def get_ohlcv(self, symbol: str, interval: str, start=None, end=None) -> ProviderResponse:
        # Offline-first: if ALLY_LIVE != "1" or missing key, return a helpful error or fixture
        live = os.getenv("ALLY_LIVE", "0") == "1"
        payload = {"symbol": symbol, "interval": interval, "start": str(start), "end": str(end)}
        key = self.cache.key("polygon_ohlcv", payload)
        cached = self.cache.get(key)
        if cached:
            return ProviderResponse(ok=True, data=self._normalize(cached), cached=True, source=self.name)

        if not live or not self.api_key:
            # try fixture fallback
            try:
                from pathlib import Path
                import json
                p = Path("data/fixtures/live/ohlcv_sample.json")
                recs = json.loads(p.read_text())
                self.cache.set(key, recs, ttl=timedelta(hours=6))
                return ProviderResponse(ok=True, data=self._normalize(recs), cached=False, source="fixture")
            except Exception:
                return ProviderResponse(ok=False, data={"error":"live disabled or API key missing; no fixture available"}, source=self.name)

        # (Live path placeholder – implement actual HTTP call later)
        return ProviderResponse(ok=False, data={"error":"live path not implemented in this PR"}, source=self.name)

    def get_news(self, query: str, limit: int = 50) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"news not implemented for polygon"}, source=self.name)

    def get_macro(self, series: str) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"macro not implemented for polygon"}, source=self.name)