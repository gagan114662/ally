"""
Finnhub data adapter for Ally
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


class FinnhubAdapter:
    """Finnhub API adapter with gating and receipts"""
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    # Interval mapping: Ally interval -> Finnhub resolution
    INTERVAL_MAP = {
        "1min":  "1",
        "5min":  "5", 
        "15min": "15",
        "30min": "30",
        "60min": "60",
        "1h":    "60",
        "1d":    "D",
        "daily": "D",
        "1wk":   "W",
        "weekly": "W",
        "1mo":   "M",
        "monthly": "M",
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY")
        
    def load_ohlcv(self, symbols: List[str], interval: str, start: str, end: str, 
                   live: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for symbols
        
        Args:
            symbols: List of symbols to fetch
            interval: Time interval (1min, 5min, 15min, 30min, 1h, 1d, etc.)
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
            service_name="Finnhub"
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
                    "source": "finnhub"
                }
                
                receipt_hash = store_tool_receipt(
                    tool_name="data.finnhub.fetch_symbol",
                    inputs=tool_inputs,
                    raw_payload=raw_data
                )
                
                # Convert to DataFrame
                df = self._convert_to_dataframe(raw_data, symbol, interval)
                if not df.empty:
                    # Add receipt hash to metadata
                    df.attrs['receipt_hash'] = receipt_hash
                    df.attrs['provider'] = 'finnhub'
                    df.attrs['fetched_at'] = datetime.utcnow().isoformat()
                    results[symbol] = df
                    
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol} from Finnhub: {e}")
                continue
        
        return results
    
    def _fetch_symbol_data(self, symbol: str, interval: str, start: str, end: str) -> Dict[str, Any]:
        """Fetch raw data from Finnhub API"""
        
        resolution = self.INTERVAL_MAP[interval]
        
        # Convert dates to Unix timestamps
        start_ts = int(pd.to_datetime(start).timestamp())
        end_ts = int(pd.to_datetime(end).timestamp())
        
        # Build URL
        url = f"{self.BASE_URL}/stock/candle"
        
        # Parameters
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": start_ts,
            "to": end_ts,
            "token": self.api_key
        }
        
        # Make API request
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if data.get("s") == "no_data":
            print(f"Warning: No data returned for {symbol} {interval} {start}-{end}")
            return {"s": "no_data", "c": [], "h": [], "l": [], "o": [], "t": [], "v": []}
        
        if data.get("s") != "ok":
            error_msg = data.get("message", f"API returned status: {data.get('s', 'unknown')}")
            raise ValueError(f"Finnhub API error: {error_msg}")
        
        # Check for rate limiting
        if response.status_code == 429:
            raise ValueError("Finnhub API rate limit exceeded. Please upgrade plan or retry later.")
        
        return data
    
    def _convert_to_dataframe(self, raw_data: Dict[str, Any], symbol: str, interval: str) -> pd.DataFrame:
        """Convert Finnhub response to standardized DataFrame"""
        
        # Check if we have data
        if raw_data.get("s") == "no_data" or not raw_data.get("c"):
            return pd.DataFrame()
        
        # Extract arrays from response
        closes = raw_data.get("c", [])
        highs = raw_data.get("h", [])
        lows = raw_data.get("l", [])
        opens = raw_data.get("o", [])
        timestamps = raw_data.get("t", [])
        volumes = raw_data.get("v", [])
        
        if not all(len(arr) == len(closes) for arr in [highs, lows, opens, timestamps, volumes]):
            raise ValueError("Inconsistent array lengths in Finnhub response")
        
        # Convert to list of records
        records = []
        for i in range(len(closes)):
            # Finnhub timestamps are Unix seconds
            timestamp = pd.to_datetime(timestamps[i], unit='s', utc=True)
            
            record = {
                "timestamp": timestamp,
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]), 
                "close": float(closes[i]),
                "volume": int(volumes[i]),
                "symbol": symbol
            }
            records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        if not df.empty:
            # Sort by timestamp
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
            np.random.seed(hash(symbol + "finnhub") % 2**32)  # Different seed from other providers
            
            base_price = 140.0 + i * 20.0  # Different base prices from other providers
            records = []
            
            for j, ts in enumerate(timestamps):
                # Random walk with mean reversion
                if j == 0:
                    close = base_price
                else:
                    # Mean reverting random walk
                    prev_close = records[-1]["close"]
                    drift = 0.0002 * (base_price - prev_close)  # Slightly higher drift
                    shock = np.random.normal(0, 0.012)  # Different volatility
                    close = prev_close * (1 + drift + shock)
                
                # Generate OHLC with realistic relationships
                volatility = abs(np.random.normal(0, 0.007))
                open_price = close * (1 + np.random.normal(0, 0.003))
                
                high = max(open_price, close) * (1 + volatility * np.random.uniform(0, 1))
                low = min(open_price, close) * (1 - volatility * np.random.uniform(0, 1))
                
                # Volume correlated with price movement
                price_change = abs(close - open_price) / open_price
                base_volume = 1800000  # Different base volume
                volume = int(base_volume * (1 + price_change * 12) * np.random.lognormal(0, 0.35))
                
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
                df.attrs['provider'] = 'finnhub_mock'
                df.attrs['generated_at'] = datetime.utcnow().isoformat()
            results[symbol] = df
        
        return results