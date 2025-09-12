"""
CSV data adapter for local file-based OHLCV data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class CSVDataAdapter:
    """
    Adapter for loading OHLCV data from CSV files
    
    Expected CSV format:
    timestamp,open,high,low,close,volume
    2024-01-01 00:00:00,100.0,102.0,99.0,101.0,1000000
    """
    
    def __init__(self, data_path: str = "data/ohlcv"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
    def load_ohlcv(self, symbols: List[str], interval: str, start: str, end: str) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data from CSV files
        
        Args:
            symbols: List of symbols
            interval: Time interval 
            start: Start date
            end: End date
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                df = self._load_symbol_data(symbol, interval, start, end)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                print(f"Warning: Failed to load {symbol}: {e}")
                
        return results
        
    def _load_symbol_data(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """Load data for a single symbol"""
        # Try different file naming conventions
        possible_files = [
            self.data_path / f"{symbol}_{interval}.csv",
            self.data_path / f"{symbol.lower()}_{interval}.csv",
            self.data_path / f"{symbol}.csv",
            self.data_path / f"{symbol.lower()}.csv"
        ]
        
        df = pd.DataFrame()
        
        for file_path in possible_files:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    break
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue
                    
        if df.empty:
            print(f"No data file found for {symbol} in {self.data_path}")
            return pd.DataFrame()
            
        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Try alternative column names
            alt_names = {
                'timestamp': ['date', 'datetime', 'time'],
                'open': ['o'],
                'high': ['h'],
                'low': ['l'],
                'close': ['c'],
                'volume': ['v', 'vol']
            }
            
            for col in missing_cols:
                for alt in alt_names.get(col, []):
                    if alt in df.columns:
                        df.rename(columns={alt: col}, inplace=True)
                        missing_cols.remove(col)
                        break
                        
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by date range
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df[required_cols]  # Return only required columns in order
        
    def create_sample_data(self, symbol: str, interval: str, days: int = 365) -> None:
        """Create sample OHLCV data for testing"""
        start_date = datetime.now() - timedelta(days=days)
        
        # Generate time series based on interval
        if interval == '1d':
            freq = 'D'
        elif interval == '1h':
            freq = 'H'
        elif interval == '5m':
            freq = '5T'
        elif interval == '1m':
            freq = '1T'
        else:
            freq = 'D'  # Default to daily
            
        timestamps = pd.date_range(start=start_date, periods=days, freq=freq)
        
        # Generate realistic price data using random walk
        np.random.seed(42)  # For reproducible data
        
        initial_price = 100.0
        returns = np.random.normal(0.0005, 0.02, len(timestamps))  # Small drift, realistic volatility
        
        # Generate OHLCV data
        data = []
        current_price = initial_price
        
        for i, timestamp in enumerate(timestamps):
            # Price movement
            if i > 0:
                current_price = current_price * (1 + returns[i])
                
            # Generate OHLC from close price
            volatility = abs(returns[i]) * 2  # Intraday volatility
            
            high = current_price * (1 + volatility * np.random.uniform(0, 1))
            low = current_price * (1 - volatility * np.random.uniform(0, 1))
            
            # Ensure OHLC consistency
            open_price = current_price + np.random.normal(0, current_price * 0.005)  # Small gap
            close = current_price
            
            # Adjust for consistency
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            # Generate volume
            base_volume = 1000000
            volume = int(base_volume * np.random.lognormal(0, 0.5))
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
            
        # Create DataFrame and save
        df = pd.DataFrame(data)
        file_path = self.data_path / f"{symbol}_{interval}.csv"
        df.to_csv(file_path, index=False)
        
        print(f"Created sample data for {symbol} at {file_path}")
        return df