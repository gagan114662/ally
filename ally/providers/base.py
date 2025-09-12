from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import timedelta

@dataclass
class ProviderResponse:
    ok: bool
    data: Any
    rate_info: Dict[str, Any] | None = None
    cached: bool = False
    source: str = ""

class RateLimiter:
    def acquire(self, key: str) -> None: ...

class Cache:
    def get(self, key: str) -> Optional[Any]: ...
    def set(self, key: str, value: Any, ttl: timedelta) -> None: ...

class DataProvider:
    name: str = "base"
    def get_ohlcv(self, symbol: str, interval: str, start=None, end=None) -> ProviderResponse:
        raise NotImplementedError
    def get_news(self, query: str, limit: int = 50) -> ProviderResponse:
        raise NotImplementedError
    def get_macro(self, series: str) -> ProviderResponse:
        raise NotImplementedError