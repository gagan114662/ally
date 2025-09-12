import os
from ally.providers.base import DataProvider, ProviderResponse

class GitHubProvider(DataProvider):
    name = "github"

    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN")

    def get_ohlcv(self, symbol: str, interval: str, start=None, end=None) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"ohlcv not supported by GitHub"}, source=self.name)

    def get_news(self, query: str, limit: int = 50) -> ProviderResponse:
        live = os.getenv("ALLY_LIVE", "0") == "1"
        if not live or not self.token:
            return ProviderResponse(ok=False, data={"error":"live disabled or token missing"}, source=self.name)
        return ProviderResponse(ok=False, data={"error":"news not implemented in this PR"}, source=self.name)

    def get_macro(self, series: str) -> ProviderResponse:
        return ProviderResponse(ok=False, data={"error":"macro not supported by GitHub"}, source=self.name)