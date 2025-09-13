"""
Provider registry for M-RealData Gate system
Manages allow-listed data providers with rate limiting and cost tracking
"""

from __future__ import annotations
import os
import time
import requests
from typing import Dict, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from ally.schemas.receipts import LiveAccessError, BudgetExceededError


@dataclass
class ProviderConfig:
    """Configuration for a data provider"""
    name: str
    base_url: str
    rate_limit_per_min: int
    cost_per_call_cents: int
    auth_header: str
    env_key: str


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def can_call(self) -> bool:
        """Check if we can make a call within rate limit"""
        now = datetime.now()
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls 
                     if call_time > now - timedelta(minutes=1)]
        
        return len(self.calls) < self.calls_per_minute
    
    def record_call(self) -> None:
        """Record a successful call"""
        self.calls.append(datetime.now())


class ProviderRegistry:
    """Registry of allow-listed data providers"""
    
    def __init__(self):
        self.providers = self._init_providers()
        self.rate_limiters = {name: RateLimiter(config.rate_limit_per_min) 
                             for name, config in self.providers.items()}
    
    def _init_providers(self) -> Dict[str, ProviderConfig]:
        """Initialize allow-listed providers"""
        return {
            "polygon": ProviderConfig(
                name="polygon",
                base_url="https://api.polygon.io",
                rate_limit_per_min=5,  # Free tier limit
                cost_per_call_cents=0,  # Free tier
                auth_header="apikey",
                env_key="POLYGON_API_KEY"
            ),
            "alphavantage": ProviderConfig(
                name="alphavantage", 
                base_url="https://www.alphavantage.co",
                rate_limit_per_min=5,  # Free tier limit
                cost_per_call_cents=0,  # Free tier
                auth_header="apikey",
                env_key="ALPHAVANTAGE_API_KEY"
            ),
            "finnhub": ProviderConfig(
                name="finnhub",
                base_url="https://finnhub.io/api/v1",
                rate_limit_per_min=30,  # Free tier limit
                cost_per_call_cents=0,  # Free tier
                auth_header="token",
                env_key="FINNHUB_API_KEY"
            ),
            "fred": ProviderConfig(
                name="fred",
                base_url="https://api.stlouisfed.org/fred",
                rate_limit_per_min=120,  # Generous limit
                cost_per_call_cents=0,  # Free
                auth_header="api_key",
                env_key="FRED_API_KEY"
            ),
            "quandl": ProviderConfig(
                name="quandl",
                base_url="https://www.quandl.com/api/v3",
                rate_limit_per_min=20,
                cost_per_call_cents=1,  # Estimate
                auth_header="api_key",
                env_key="QUANDL_API_KEY"
            ),
            "reddit": ProviderConfig(
                name="reddit",
                base_url="https://oauth.reddit.com",
                rate_limit_per_min=60,
                cost_per_call_cents=0,  # Free
                auth_header="Authorization",
                env_key="REDDIT_ACCESS_TOKEN"
            ),
            "github": ProviderConfig(
                name="github",
                base_url="https://api.github.com",
                rate_limit_per_min=60,
                cost_per_call_cents=0,  # Free for most calls
                auth_header="Authorization",
                env_key="GITHUB_TOKEN"
            ),
            "tavily": ProviderConfig(
                name="tavily",
                base_url="https://api.tavily.com",
                rate_limit_per_min=10,
                cost_per_call_cents=2,  # Estimate
                auth_header="api_key",
                env_key="TAVILY_API_KEY"
            ),
            "valyu": ProviderConfig(
                name="valyu",
                base_url="https://api.valyu.com",
                rate_limit_per_min=30,
                cost_per_call_cents=1,  # Estimate
                auth_header="api_key",
                env_key="VALYU_API_KEY"
            ),
            "binance": ProviderConfig(
                name="binance",
                base_url="https://api.binance.com",
                rate_limit_per_min=1200,  # Weight-based, simplified
                cost_per_call_cents=0,  # Free for market data
                auth_header="X-MBX-APIKEY",
                env_key="BINANCE_API_KEY"
            ),
            "coinbase": ProviderConfig(
                name="coinbase",
                base_url="https://api.coinbase.com/v2",
                rate_limit_per_min=10000,  # High limit for public data
                cost_per_call_cents=0,  # Free for market data
                auth_header="Authorization",
                env_key="COINBASE_API_KEY"
            )
        }
    
    def get_provider(self, name: str) -> ProviderConfig:
        """Get provider config by name"""
        if name not in self.providers:
            raise ValueError(f"Provider '{name}' not in allow list. "
                           f"Allowed: {list(self.providers.keys())}")
        return self.providers[name]
    
    def check_rate_limit(self, provider_name: str) -> bool:
        """Check if we can make a call to provider within rate limit"""
        if provider_name not in self.rate_limiters:
            return False
        return self.rate_limiters[provider_name].can_call()
    
    def record_call(self, provider_name: str) -> None:
        """Record a successful call to provider"""
        if provider_name in self.rate_limiters:
            self.rate_limiters[provider_name].record_call()
    
    def fetch_with_auth(self, provider_name: str, endpoint: str, 
                       params: Dict[str, Any]) -> bytes:
        """
        Fetch data from provider with authentication
        Raises LiveAccessError if ALLY_LIVE!=1
        """
        # Enforce live mode gate
        if os.getenv("ALLY_LIVE") != "1":
            raise LiveAccessError(f"Provider {provider_name}: ALLY_LIVE!=1, network access denied")
        
        provider = self.get_provider(provider_name)
        
        # Check rate limit
        if not self.check_rate_limit(provider_name):
            raise RuntimeError(f"Rate limit exceeded for {provider_name}")
        
        # Get API key from environment
        api_key = os.getenv(provider.env_key)
        if not api_key:
            raise RuntimeError(f"Missing {provider.env_key} environment variable")
        
        # Build request
        url = f"{provider.base_url}{endpoint}"
        headers = {provider.auth_header: api_key}
        
        # Add API key to params if expected in query string
        if provider.auth_header == "apikey" and "apikey" not in params:
            params["apikey"] = api_key
        elif provider.auth_header == "token" and "token" not in params:
            params["token"] = api_key
        elif provider.auth_header == "api_key" and "api_key" not in params:
            params["api_key"] = api_key
        
        # Make request
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Record successful call
            self.record_call(provider_name)
            
            return response.content
            
        except requests.RequestException as e:
            raise RuntimeError(f"Provider {provider_name} fetch failed: {e}")


# Global registry instance
PROVIDER_REGISTRY = ProviderRegistry()