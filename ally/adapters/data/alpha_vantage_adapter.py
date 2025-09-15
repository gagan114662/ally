"""
Alpha Vantage data adapter for Ally
Handles live data fetching with proper gating and receipt generation
"""

import os
import json
import time
import requests
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ...utils.gating import check_live_mode_allowed
from ...utils.receipts import store_tool_receipt
from ...utils.hashing import hash_inputs


class AlphaVantageAdapter:
    """Alpha Vantage API adapter with gating and receipts"""
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        
    def load_ohlcv(self, symbols: List[str], interval: str, start: str, end: str, 
                   live: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data for symbols
        
        Args:
            symbols: List of symbols to fetch
            interval: Time interval (1min, 5min, 15min, 30min, 60min, daily)
            start: Start date string
            end: End date string  
            live: Whether to fetch live data (requires gating)
            
        Returns:
            Dictionary of symbol -> DataFrame mappings
        """
        # Check gating requirements
        check_live_mode_allowed(
            live=live, 
            api_key=self.api_key, 
            service_name="Alpha Vantage"
        )
        
        if not live:
            # Return mock data for offline mode
            return self._generate_mock_data(symbols, interval, start, end)
        
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
                    "source": "alpha_vantage"
                }
                
                receipt_hash = store_tool_receipt(
                    tool_name="data.alpha_vantage.fetch_symbol",
                    inputs=tool_inputs,
                    raw_payload=raw_data
                )
                
                # Convert to DataFrame
                df = self._convert_to_dataframe(raw_data, symbol, interval)
                if not df.empty:
                    # Add receipt hash to metadata
                    df.attrs['receipt_hash'] = receipt_hash
                    df.attrs['provider'] = 'alpha_vantage'
                    df.attrs['fetched_at'] = datetime.utcnow().isoformat()
                    results[symbol] = df
                    
            except Exception as e:
                print(f"Warning: Failed to fetch {symbol} from Alpha Vantage: {e}")
                continue
        
        return results
    
    def _fetch_symbol_data(self, symbol: str, interval: str, start: str, end: str) -> Dict[str, Any]:
        """Fetch raw data from Alpha Vantage API"""
        
        # Map our intervals to Alpha Vantage intervals
        av_interval_map = {
            "1min": "1min",
            "5min": "5min", 
            "15min": "15min",
            "30min": "30min",
            "60min": "60min",
            "1h": "60min",
            "1d": "daily",
            "daily": "daily"
        }
        
        av_interval = av_interval_map.get(interval, "daily")
        
        # Choose function based on interval
        if av_interval in ["1min", "5min", "15min", "30min", "60min"]:
            function = "TIME_SERIES_INTRADAY"
            params = {
                "function": function,
                "symbol": symbol,
                "interval": av_interval,
                "apikey": self.api_key,
                "outputsize": "full",
                "datatype": "json"
            }
        else:
            function = "TIME_SERIES_DAILY"  
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": self.api_key,
                "outputsize": "full", 
                "datatype": "json"
            }
        
        # Make API request
        response = requests.get(self.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Check for API errors
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
        
        if "Note" in data:
            raise ValueError(f"Alpha Vantage API limit: {data['Note']}")
        
        return data
    
    def _convert_to_dataframe(self, raw_data: Dict[str, Any], symbol: str, interval: str) -> pd.DataFrame:
        """Convert Alpha Vantage response to standardized DataFrame"""
        
        # Find the time series key in response
        time_series_key = None
        for key in raw_data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if not time_series_key:
            return pd.DataFrame()
        
        time_series = raw_data[time_series_key]
        
        # Convert to list of records
        records = []
        for timestamp_str, ohlcv in time_series.items():
            # Parse timestamp
            timestamp = pd.to_datetime(timestamp_str)
            
            # Extract OHLCV values (Alpha Vantage uses numbered keys)
            record = {
                "timestamp": timestamp,
                "open": float(ohlcv.get("1. open", 0)),
                "high": float(ohlcv.get("2. high", 0)),
                "low": float(ohlcv.get("3. low", 0)), 
                "close": float(ohlcv.get("4. close", 0)),
                "volume": int(float(ohlcv.get("5. volume", 0)))
            }
            records.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        if not df.empty:
            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            # Add symbol column
            df["symbol"] = symbol
        
        return df
    
    def _generate_mock_data(self, symbols: List[str], interval: str, start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Generate mock data for offline testing"""
        results = {}
        
        # Parse dates
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        # Generate timestamps based on interval
        if interval in ["1min", "5min", "15min", "30min", "60min", "1h"]:
            freq_map = {
                "1min": "1T", "5min": "5T", "15min": "15T", 
                "30min": "30T", "60min": "60T", "1h": "1H"
            }
            freq = freq_map.get(interval, "1H")
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq=freq)
        else:
            timestamps = pd.date_range(start=start_dt, end=end_dt, freq="D")
        
        # Limit to reasonable size
        if len(timestamps) > 5000:
            timestamps = timestamps[-5000:]
        
        for i, symbol in enumerate(symbols):
            # Generate deterministic mock data 
            import numpy as np
            np.random.seed(hash(symbol) % 2**32)  # Deterministic per symbol
            
            base_price = 100.0 + i * 50.0  # Different base prices
            records = []
            
            for j, ts in enumerate(timestamps):
                # Random walk with mean reversion
                if j == 0:
                    close = base_price
                else:
                    # Mean reverting random walk
                    prev_close = records[-1]["close"]
                    drift = 0.0001 * (base_price - prev_close)  # Mean reversion
                    shock = np.random.normal(0, 0.01)  # Volatility
                    close = prev_close * (1 + drift + shock)
                
                # Generate OHLC with realistic relationships
                volatility = abs(np.random.normal(0, 0.005))
                open_price = close * (1 + np.random.normal(0, 0.001))
                
                high = max(open_price, close) * (1 + volatility * np.random.uniform(0, 1))
                low = min(open_price, close) * (1 - volatility * np.random.uniform(0, 1))
                
                # Volume correlated with price movement
                price_change = abs(close - open_price) / open_price
                base_volume = 1000000
                volume = int(base_volume * (1 + price_change * 10) * np.random.lognormal(0, 0.2))
                
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
            df.attrs['provider'] = 'alpha_vantage_mock'
            df.attrs['generated_at'] = datetime.utcnow().isoformat()
            results[symbol] = df
        
        return results