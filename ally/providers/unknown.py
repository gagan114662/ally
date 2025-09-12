import os
from ally.providers.base import DataProvider, ProviderResponse

class UnknownProvider(DataProvider):
    name = "unknown"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("UNKNOWN_API_KEY")

    def get_ohlcv(self, symbol: str, interval: str, start=None, end=None) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"secret provider API not disclosed"}, source=self.name)

    def get_news(self, query: str, limit: int = 50) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"secret provider API not disclosed"}, source=self.name)

    def get_macro(self, series: str) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"secret provider API not disclosed"}, source=self.name)