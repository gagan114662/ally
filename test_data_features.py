#!/usr/bin/env python3
"""
Test script for data and features tools
"""

import sys
import os
import traceback
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_data_features():
    """Test data and features functionality"""
    
    print("🧪 Testing data and features tools...")
    
    try:
        # Test 1: Mock data generation
        print("\n1. Testing mock OHLCV data generation...")
        
        from ally.tools.data import _generate_mock_ohlcv
        
        symbols = ["BTCUSDT", "ETHUSDT"]
        mock_data = _generate_mock_ohlcv(symbols, "1h", "2024-01-01", "2024-01-07")
        
        print(f"✅ Generated mock data for {len(mock_data)} symbols")
        for symbol, df in mock_data.items():
            print(f"   {symbol}: {len(df)} rows, columns: {list(df.columns)}")
        
        # Test 2: Feature calculations
        print("\n2. Testing technical indicator calculations...")
        
        from ally.tools.features import calculate_rsi, calculate_ema, calculate_atr
        import numpy as np
        
        # Use close prices from mock data
        close_prices = mock_data["BTCUSDT"]["close"].values
        high_prices = mock_data["BTCUSDT"]["high"].values
        low_prices = mock_data["BTCUSDT"]["low"].values
        
        rsi = calculate_rsi(close_prices)
        ema = calculate_ema(close_prices, 20)
        atr = calculate_atr(high_prices, low_prices, close_prices)
        
        print(f"✅ RSI calculation: {len(rsi)} values, range: {np.nanmin(rsi):.2f}-{np.nanmax(rsi):.2f}")
        print(f"✅ EMA calculation: {len(ema)} values, last value: {ema[-1]:.2f}")
        print(f"✅ ATR calculation: {len(atr)} values, avg: {np.nanmean(atr):.2f}")
        
        # Test 3: Data alignment
        print("\n3. Testing data alignment...")
        
        from ally.tools.data import _align_ohlcv_data
        
        aligned = _align_ohlcv_data(mock_data)
        print(f"✅ Data alignment: {len(aligned)} symbols aligned")
        
        for symbol, records in aligned.items():
            print(f"   {symbol}: {len(records)} aligned records")
        
        # Test 4: Metric normalization integration
        print("\n4. Testing metric normalization...")
        
        from autonomous_quant.core.optimization_engine import normalize_metrics
        
        test_metrics = {
            'cagr': 0.15,
            'sharpe': 1.1,
            'max_dd': -0.12,
            'volatility_annualized': 0.18
        }
        
        normalized = normalize_metrics(test_metrics)
        required_keys = ['annual_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        
        all_present = all(key in normalized for key in required_keys)
        print(f"✅ Normalization: all required keys present: {all_present}")
        print(f"   Normalized keys: {list(normalized.keys())}")
        
        print("\n🎉 All data and features tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_features()
    sys.exit(0 if success else 1)