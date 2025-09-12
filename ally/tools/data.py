"""
Data tools for Ally - load OHLCV and other financial data
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..tools import register
from ..schemas.base import ToolResult
from ..schemas.data import LoadOHLCVIn, OHLCVData, DataPanel


@register("data.load_ohlcv")
def data_load_ohlcv(**kwargs) -> ToolResult:
    """
    Load OHLCV data for multiple symbols with alignment
    
    Supports multiple data sources and returns aligned time series data
    """
    try:
        inputs = LoadOHLCVIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    warnings = []
    start_time = time.time()
    
    try:
        if inputs.source == "mock":
            # Generate mock data for testing
            results = _generate_mock_ohlcv(inputs.symbols, inputs.interval, inputs.start, inputs.end)
        elif inputs.source == "csv":
            # Load from CSV files
            from ..adapters.data.csv_adapter import CSVDataAdapter
            adapter = CSVDataAdapter(inputs.data_path or "data/ohlcv")
            results = adapter.load_ohlcv(inputs.symbols, inputs.interval, inputs.start, inputs.end)
        else:
            # Default to mock for now
            results = _generate_mock_ohlcv(inputs.symbols, inputs.interval, inputs.start, inputs.end)
            warnings.append(f"Source '{inputs.source}' not implemented, using mock data")
        
        if not results:
            return ToolResult.error(["No data loaded for any symbols"])
        
        # Align data across symbols
        aligned_data = _align_ohlcv_data(results)
        
        # Create data panel
        all_timestamps = set()
        for symbol_data in results.values():
            all_timestamps.update(symbol_data['timestamp'])
            
        panel = DataPanel(
            symbols=list(results.keys()),
            interval=inputs.interval,
            start_date=pd.to_datetime(inputs.start),
            end_date=pd.to_datetime(inputs.end),
            total_rows=len(all_timestamps),
            aligned_data=aligned_data,
            metadata={
                "source": inputs.source,
                "load_time": time.time() - start_time,
                "symbols_requested": inputs.symbols,
                "symbols_loaded": list(results.keys())
            }
        )
        
        return ToolResult.success(
            data={
                'panel': panel.model_dump(),
                'summary': {
                    'symbols_loaded': len(results),
                    'total_rows': len(all_timestamps),
                    'date_range': f"{inputs.start} to {inputs.end}",
                    'interval': inputs.interval
                }
            },
            warnings=warnings
        )
        
    except Exception as e:
        return ToolResult.error([f"Data loading failed: {e}"])


@register("data.create_sample")
def data_create_sample(**kwargs) -> ToolResult:
    """
    Create sample OHLCV data for testing
    
    Generates realistic price data with proper OHLC relationships
    """
    symbols = kwargs.get('symbols', ['BTCUSDT', 'ETHUSDT'])
    interval = kwargs.get('interval', '1h')
    days = kwargs.get('days', 30)
    data_path = kwargs.get('data_path', 'data/fixtures/ohlcv')
    
    try:
        from ..adapters.data.csv_adapter import CSVDataAdapter
        adapter = CSVDataAdapter(data_path)
        
        created_files = []
        for symbol in symbols:
            df = adapter.create_sample_data(symbol, interval, days)
            created_files.append(f"{symbol}_{interval}.csv")
            
        return ToolResult.success(
            data={
                'files_created': created_files,
                'data_path': data_path,
                'symbols': symbols,
                'interval': interval,
                'days': days
            }
        )
        
    except Exception as e:
        return ToolResult.error([f"Sample data creation failed: {e}"])


def _generate_mock_ohlcv(symbols: List[str], interval: str, start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Generate mock OHLCV data for testing"""
    results = {}
    
    # Parse date range
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    
    # Generate timestamps based on interval
    if interval == '1d':
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq='D')
    elif interval == '1h':
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq='H')
    elif interval == '5m':
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq='5T')
    else:
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq='H')  # Default to hourly
    
    # Limit to reasonable size for mock data
    if len(timestamps) > 10000:
        timestamps = timestamps[-10000:]  # Keep last 10k points
    
    for i, symbol in enumerate(symbols):
        # Generate realistic price data
        np.random.seed(42 + i)  # Different seed per symbol
        
        initial_price = 100.0 * (i + 1)  # Different base prices
        returns = np.random.normal(0.0002, 0.015, len(timestamps))  # Realistic return distribution
        
        data = []
        current_price = initial_price
        
        for j, timestamp in enumerate(timestamps):
            if j > 0:
                current_price = current_price * (1 + returns[j])
            
            # Generate OHLC with realistic relationships
            volatility = abs(returns[j]) * 2
            
            open_price = current_price + np.random.normal(0, current_price * 0.002)
            close = current_price
            
            high = max(open_price, close) * (1 + volatility * np.random.uniform(0, 1))
            low = min(open_price, close) * (1 - volatility * np.random.uniform(0, 1))
            
            # Volume correlated with volatility
            base_volume = 1000000 + i * 500000
            volume = int(base_volume * (1 + volatility * 10) * np.random.lognormal(0, 0.3))
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })
        
        results[symbol] = pd.DataFrame(data)
    
    return results


def _align_ohlcv_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    """Align OHLCV data across symbols to common timestamps"""
    if not data_dict:
        return {}
    
    # Find common timestamp range
    all_timestamps = None
    
    for symbol, df in data_dict.items():
        if all_timestamps is None:
            all_timestamps = set(df['timestamp'])
        else:
            all_timestamps = all_timestamps.intersection(set(df['timestamp']))
    
    if not all_timestamps:
        # No common timestamps, return original data
        return {symbol: df.to_dict('records') for symbol, df in data_dict.items()}
    
    # Sort timestamps
    common_timestamps = sorted(all_timestamps)
    
    # Align all symbols to common timestamps
    aligned = {}
    
    for symbol, df in data_dict.items():
        # Filter to common timestamps
        aligned_df = df[df['timestamp'].isin(common_timestamps)]
        aligned_df = aligned_df.sort_values('timestamp').reset_index(drop=True)
        aligned[symbol] = aligned_df.to_dict('records')
    
    return aligned


if __name__ == "__main__":
    # Test data loading
    result = data_load_ohlcv(
        symbols=["BTCUSDT", "ETHUSDT"], 
        interval="1h",
        start="2024-01-01",
        end="2024-01-07",
        source="mock"
    )
    
    print(f"Data loading test: {result.ok}")
    if result.ok:
        print(f"Symbols loaded: {result.data['summary']['symbols_loaded']}")
        print(f"Total rows: {result.data['summary']['total_rows']}")
    else:
        print(f"Errors: {result.errors}")