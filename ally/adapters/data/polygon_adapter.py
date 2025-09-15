"""
Polygon.io data adapter for Ally
Handles live data fetching with proper gating and receipt generation
"""

import os
import json
import time
import requests
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from ...utils.gating import check_live_mode_allowed
from ...utils.receipts import store_tool_receipt
from ...utils.hashing import hash_inputs


class PolygonAdapter:
    """Polygon.io API adapter with gating and receipts"""
    
    BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
    
    # Interval mapping: Ally interval -> (multiplier, timespan)
    INTERVAL_MAP = {
        "1min":  (1, "minute"),
        "5min":  (5, "minute"),
        "15min": (15, "minute"),
        "30min": (30, "minute"),
        "60min": (1, "hour"),
        "1h":    (1, "hour"),
        "4h":    (4, "hour"),
        "1d":    (1, "day"),
        "daily": (1, "day"),
        "1wk":   (1, "week"),
        "weekly": (1, "week"),
        "1mo":   (1, "month"),
        "monthly": (1, "month"),
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        
    def load_ohlcv(self, symbols: List[str], interval: str, start: str, end: str, 
                   live: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for symbols
        
        Args:
            symbols: List of symbols to fetch
            interval: Time interval (1min, 5min, 15min, 30min, 1h, 4h, 1d, etc.)
            start: Start date string (YYYY-MM-DD)
            end: End date string (YYYY-MM-DD)
            live: Whether to fetch live data (requires gating)
            
        Returns:
            Dictionary of symbol -> DataFrame mappings
        """
        # Check gating requirements
        check_live_mode_allowed(
            live=live, 
            api_key=self.api_key, 
            service_name="Polygon.io"
        )
        
        if not live:
            # Return mock data for offline mode
            return self._generate_mock_data(symbols, interval, start, end)
        
        # Validate interval
        if interval not in self.INTERVAL_MAP:
            supported = ", ".join(self.INTERVAL_MAP.keys())
            raise ValueError(f"Unsupported interval '{interval}'. Supported: {supported}")
        
        # Live data fetching
        results = {}
        for symbol in symbols:
            try:
                # Fetch data for symbol
                raw_data = self._fetch_symbol_data(symbol, interval, start, end)
                
                # Store receipt for this symbol
                tool_inputs = {
                    "symbol": symbol,
                    "interval": interval, 
                    "start": start,
                    "end": end,
                    "source": "polygon"
                }
                
                receipt_hash = store_tool_receipt(
                    tool_name="data.polygon.fetch_symbol",
                    inputs=tool_inputs,
                    raw_payload=raw_data
                )
                
                # Convert to DataFrame
                df = self._convert_to_dataframe(raw_data, symbol, interval)
                if not df.empty:
                    # Add receipt hash to metadata
                    df.attrs['receipt_hash'] = receipt_hash
                    df.attrs['provider'] = 'polygon'
                    df.attrs['fetched_at'] = datetime.utcnow().isoformat()
                    results[symbol] = df
                    
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol} from Polygon.io: {e}")
                continue
        
        return results
    
    def _fetch_symbol_data(self, symbol: str, interval: str, start: str, end: str) -> Dict[str, Any]:
        """Fetch raw data from Polygon.io API"""
        
        multiplier, timespan = self.INTERVAL_MAP[interval]
        
        # Build URL
        url = f"{self.BASE_URL}/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
        
        # Parameters
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,  # Max results per request
            "apikey": self.api_key
        }
        
        # Make API request
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if data.get("status") == "ERROR":
            error_msg = data.get("error", "Unknown Polygon.io API error")
            raise ValueError(f"Polygon.io API error: {error_msg}")
        
        if data.get("status") == "OK" and data.get("resultsCount", 0) == 0:
            # Handle gracefully - might be weekend, holiday, or no data
            print(f"Warning: No data returned for {symbol} {interval} {start}-{end}")
        
        # Check for rate limiting
        if "results" not in data and data.get("status") != "OK":
            if "rate" in str(data).lower() or "limit" in str(data).lower():
                raise ValueError(f"Polygon.io rate limit exceeded. Please upgrade plan or retry later.")
        
        # Check for tier restrictions (e.g., 4h bars on free tier)
        if interval == "4h" and data.get("status") != "OK":
            raise ValueError(f"4h interval may require paid Polygon tier. Error: {data.get('message', 'Unknown')}")
        
        return data
    
    def _convert_to_dataframe(self, raw_data: Dict[str, Any], symbol: str, interval: str) -> pd.DataFrame:
        """Convert Polygon.io response to standardized DataFrame"""
        
        results = raw_data.get("results", [])
        if not results:
            return pd.DataFrame()
        
        # Convert to list of records
        records = []
        for bar in results:
            # Polygon timestamps are in milliseconds since epoch
            timestamp_ms = bar.get("t", 0)
            timestamp = pd.to_datetime(timestamp_ms, unit='ms', utc=True)
            
            # Extract OHLCV values
            record = {
                "timestamp": timestamp,
                "open": float(bar.get("o", 0)),
                "high": float(bar.get("h", 0)),
                "low": float(bar.get("l", 0)), 
                "close": float(bar.get("c", 0)),
                "volume": int(bar.get("v", 0)),
                "symbol": symbol
            }
            records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        if not df.empty:
            # Sort by timestamp (should already be sorted from API)
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Ensure timestamps are UTC
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        return df
    
    def _generate_mock_data(self, symbols: List[str], interval: str, start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Generate mock data for offline testing"""
        results = {}
        
        # Parse dates
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        # Generate timestamps based on interval
        if interval in ["1min", "5min", "15min", "30min"]:
            # Minute-based intervals
            if interval == "1min":
                freq = "1T"
            elif interval == "5min":
                freq = "5T"
            elif interval == "15min":
                freq = "15T"
            elif interval == "30min":
                freq = "30T"
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq=freq, tz='UTC')
        elif interval in ["60min", "1h"]:
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq="1H", tz='UTC')
        elif interval == "4h":
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq="4H", tz='UTC')
        elif interval in ["1d", "daily"]:
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq="D", tz='UTC')
        elif interval in ["1wk", "weekly"]:
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq="W", tz='UTC')
        elif interval in ["1mo", "monthly"]:
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq="MS", tz='UTC')
        else:
            # Default to daily
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq="D", tz='UTC')
        
        # Limit to reasonable size
        if len(timestamps) > 10000:
            timestamps = timestamps[-10000:]
        
        for i, symbol in enumerate(symbols):
            # Generate deterministic mock data 
            import numpy as np
            np.random.seed(hash(symbol) % 2**32)  # Deterministic per symbol
            
            base_price = 150.0 + i * 25.0  # Different base prices
            records = []
            
            for j, ts in enumerate(timestamps):
                # Random walk with mean reversion
                if j == 0:
                    close = base_price
                else:
                    # Mean reverting random walk
                    prev_close = records[-1]["close"]
                    drift = 0.0001 * (base_price - prev_close)  # Mean reversion
                    shock = np.random.normal(0, 0.015)  # Volatility
                    close = prev_close * (1 + drift + shock)
                
                # Generate OHLC with realistic relationships
                volatility = abs(np.random.normal(0, 0.008))
                open_price = close * (1 + np.random.normal(0, 0.002))
                
                high = max(open_price, close) * (1 + volatility * np.random.uniform(0, 1))
                low = min(open_price, close) * (1 - volatility * np.random.uniform(0, 1))
                
                # Volume correlated with price movement
                price_change = abs(close - open_price) / open_price
                base_volume = 2000000  # Higher base volume than Alpha Vantage
                volume = int(base_volume * (1 + price_change * 15) * np.random.lognormal(0, 0.3))
                
                record = {
                    "timestamp": ts,
                    "open": round(open_price, 2),
                    "high": round(high, 2), 
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "volume": volume,
                    "symbol": symbol
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            if not df.empty:
                # Ensure UTC timezone
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
                df.attrs['provider'] = 'polygon_mock'
                df.attrs['generated_at'] = datetime.utcnow().isoformat()
            results[symbol] = df
        
        return results