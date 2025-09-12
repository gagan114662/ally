import os
from ally.providers.base import DataProvider, ProviderResponse

class TavilyProvider(DataProvider):
    name = "tavily"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    def get_ohlcv(self, symbol: str, interval: str, start=None, end=None) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"ohlcv not supported by Tavily"}, source=self.name)

    def get_news(self, query: str, limit: int = 50) -> ProviderResponse:
        live = os.getenv("ALLY_LIVE", "0") == "1"
        if not live or not self.api_key:
            return ProviderResponse(ok=False, data={"error":"live disabled or API key missing"}, source=self.name)
        return ProviderResponse(ok=False, data={"error":"news not implemented in this PR"}, source=self.name)

    def get_macro(self, series: str) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"macro not supported by Tavily"}, source=self.name)