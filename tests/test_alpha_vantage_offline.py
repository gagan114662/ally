#!/usr/bin/env python3
"""
Alpha Vantage offline tests - deterministic mock data and offline behavior
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ally.adapters.data.alpha_vantage_adapter import AlphaVantageAdapter
from ally.tools.data import data_load_ohlcv


class TestAlphaVantageOffline:
    
    def test_mock_data_deterministic(self):
        """Test that mock data is deterministic across runs"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = AlphaVantageAdapter()
            
            # First run
            result1 = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1d",
                start="2024-01-01", 
                end="2024-01-10",
                live=False
            )
            
            # Second run with same parameters
            result2 = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1d",
                start="2024-01-01",
                end="2024-01-10", 
                live=False
            )
            
            # Results should be identical
            df1 = result1["AAPL"]
            df2 = result2["AAPL"]
            
            assert len(df1) == len(df2), "Mock data length not deterministic"
            
            # Compare key columns (allowing for small floating point differences)
            pd.testing.assert_series_equal(df1['close'], df2['close'], check_names=False)
            pd.testing.assert_series_equal(df1['volume'], df2['volume'], check_names=False)
            pd.testing.assert_series_equal(df1['timestamp'], df2['timestamp'], check_names=False)
    
    def test_different_symbols_different_data(self):
        """Test that different symbols produce different mock data"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = AlphaVantageAdapter()
            
            result = adapter.load_ohlcv(
                symbols=["AAPL", "MSFT", "GOOGL"],
                interval="1d",
                start="2024-01-01",
                end="2024-01-05",
                live=False
            )
            
            # Should have data for all symbols
            assert "AAPL" in result
            assert "MSFT" in result
            assert "GOOGL" in result
            
            # Data should be different between symbols
            aapl_close = result["AAPL"]['close'].iloc[0]
            msft_close = result["MSFT"]['close'].iloc[0]
            googl_close = result["GOOGL"]['close'].iloc[0]
            
            # Base prices should be different (100 + i * 50)
            assert abs(aapl_close - 100) < 50  # Around 100
            assert abs(msft_close - 150) < 50  # Around 150
            assert abs(googl_close - 200) < 50  # Around 200
    
    def test_ohlc_relationships_valid(self):
        """Test that OHLC relationships are realistic in mock data"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = AlphaVantageAdapter()
            
            result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1h",
                start="2024-01-01",
                end="2024-01-02", 
                live=False
            )
            
            df = result["AAPL"]
            
            # Check OHLC relationships for each row
            for _, row in df.iterrows():
                high, low, open_price, close = row['high'], row['low'], row['open'], row['close']
                
                # High should be >= max(open, close)
                assert high >= max(open_price, close), f"High {high} < max(open {open_price}, close {close})"
                
                # Low should be <= min(open, close)  
                assert low <= min(open_price, close), f"Low {low} > min(open {open_price}, close {close})"
                
                # All prices should be positive
                assert high > 0 and low > 0 and open_price > 0 and close > 0
                
                # Volume should be positive integer
                assert row['volume'] > 0
                assert isinstance(row['volume'], (int, float))
    
    def test_interval_frequency_mapping(self):
        """Test that different intervals produce appropriate data frequency"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = AlphaVantageAdapter()
            
            # Daily data
            daily_result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1d",
                start="2024-01-01",
                end="2024-01-05",  # 5 days
                live=False
            )
            
            # Hourly data  
            hourly_result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1h",
                start="2024-01-01",
                end="2024-01-02",  # 1 day
                live=False
            )
            
            daily_df = daily_result["AAPL"]
            hourly_df = hourly_result["AAPL"]
            
            # Daily should have fewer records than hourly for same time span
            assert len(daily_df) <= 5, f"Daily data has {len(daily_df)} records, expected <= 5"
            assert len(hourly_df) >= 20, f"Hourly data has {len(hourly_df)} records, expected >= 20"
    
    def test_metadata_attributes(self):
        """Test that mock data includes proper metadata attributes"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = AlphaVantageAdapter()
            
            result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1d",
                start="2024-01-01",
                end="2024-01-05",
                live=False
            )
            
            df = result["AAPL"]
            
            # Check DataFrame attributes
            assert hasattr(df, 'attrs'), "DataFrame missing attrs"
            assert df.attrs['provider'] == 'alpha_vantage_mock'
            assert 'generated_at' in df.attrs
            
            # Timestamp should be ISO format
            generated_at = df.attrs['generated_at']
            datetime.fromisoformat(generated_at.replace('Z', '+00:00'))  # Should not raise
    
    def test_tool_level_offline_integration(self):
        """Test complete tool integration in offline mode"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            
            result = data_load_ohlcv(
                symbols=["AAPL", "MSFT"],
                interval="1d",
                start="2024-01-01", 
                end="2024-01-05",
                source="alpha_vantage",
                live=False
            )
            
            assert result.ok == True
            assert len(result.warnings) == 0
            
            # Check summary data
            summary = result.data['summary']
            assert summary['symbols_loaded'] == 2
            assert summary['provider'] == 'alpha_vantage'
            assert summary['interval'] == '1d'
            assert isinstance(summary['receipt_hashes'], list)
            assert len(summary['receipt_hashes']) == 0  # No receipts for offline mode
            
            # Check panel metadata
            panel_meta = result.data['panel']['metadata']
            assert panel_meta['source'] == 'alpha_vantage'
            assert panel_meta['live_mode'] == False
            assert 'utc_timestamp' in panel_meta
            assert 'load_time' in panel_meta
            
            # Check that provider_info shows mock providers
            provider_info = panel_meta['provider_info']
            assert 'AAPL' in provider_info
            assert 'MSFT' in provider_info
            assert provider_info['AAPL']['provider'] == 'alpha_vantage_mock'
            assert provider_info['MSFT']['provider'] == 'alpha_vantage_mock'
    
    def test_large_date_range_limited(self):
        """Test that large date ranges are limited to reasonable size"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = AlphaVantageAdapter()
            
            # Request a very large date range
            result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1min",
                start="2020-01-01",
                end="2024-01-01",  # 4 years of 1-minute data
                live=False
            )
            
            df = result["AAPL"]
            
            # Should be limited to 5000 records max
            assert len(df) <= 5000, f"Mock data not limited: {len(df)} records"
    
    def test_no_network_calls_offline(self):
        """Test that no network calls are made in offline mode"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            with patch('requests.get') as mock_get:
                adapter = AlphaVantageAdapter()
                
                result = adapter.load_ohlcv(
                    symbols=["AAPL"],
                    interval="1d",
                    start="2024-01-01",
                    end="2024-01-05",
                    live=False
                )
                
                # No network calls should be made
                mock_get.assert_not_called()
                
                # Should still return valid data
                assert "AAPL" in result
                assert len(result["AAPL"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])