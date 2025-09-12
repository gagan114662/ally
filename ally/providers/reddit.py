import os
from ally.providers.base import DataProvider, ProviderResponse

class RedditProvider(DataProvider):
    name = "reddit"

    def __init__(self, client_id: str | None = None, client_secret: str | None = None):
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")

    def get_ohlcv(self, symbol: str, interval: str, start=None, end=None) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"ohlcv not supported by Reddit"}, source=self.name)

    def get_news(self, query: str, limit: int = 50) -> ProviderResponse:
        live = os.getenv("ALLY_LIVE", "0") == "1"
        if not live or not self.client_id or not self.client_secret:
            return ProviderResponse(ok=False, data={"error":"live disabled or credentials missing"}, source=self.name)
        return ProviderResponse(ok=False, data={"error":"news not implemented in this PR"}, source=self.name)

    def get_macro(self, series: str) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"macro not supported by Reddit"}, source=self.name)