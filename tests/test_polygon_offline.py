#!/usr/bin/env python3
"""
Polygon offline tests - deterministic mock data and offline behavior
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ally.adapters.data.polygon_adapter import PolygonAdapter
from ally.tools.data import data_load_ohlcv


class TestPolygonOffline:
    
    def test_mock_data_deterministic(self):
        """Test that mock data is deterministic across runs"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = PolygonAdapter()
            
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
            pd.testing.assert_index_equal(df1.index, df2.index, check_names=False)
    
    def test_different_symbols_different_data(self):
        """Test that different symbols produce different mock data"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = PolygonAdapter()
            
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
            
            # Base prices should be different (150 + i * 25)
            assert abs(aapl_close - 150) < 50  # Around 150
            assert abs(msft_close - 175) < 50  # Around 175
            assert abs(googl_close - 200) < 50  # Around 200
    
    def test_utc_timezone_enforcement(self):
        """Test that all timestamps are properly in UTC timezone"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = PolygonAdapter()
            
            result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1h",
                start="2024-01-01",
                end="2024-01-02", 
                live=False
            )
            
            df = result["AAPL"]
            
            # Check timezone
            assert df.index.tz is not None, "Timestamps should be timezone-aware"
            assert str(df.index.tz) in ("UTC", "tzutc()", "UTC+00:00"), f"Expected UTC, got {df.index.tz}"
            
            # Check that all individual timestamps are UTC
            for ts in df.index[:5]:  # Check first 5 timestamps
                assert ts.tz is not None, f"Timestamp {ts} should be timezone-aware"
                assert str(ts.tz) in ("UTC", "tzutc()", "UTC+00:00"), f"Timestamp {ts} should be UTC"
    
    def test_ohlc_relationships_valid(self):
        """Test that OHLC relationships are realistic in mock data"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = PolygonAdapter()
            
            result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1h",
                start="2024-01-01",
                end="2024-01-02", 
                live=False
            )
            
            df = result["AAPL"]
            
            # Check OHLC relationships for each row
            for idx, row in df.iterrows():
                high, low, open_price, close = row['high'], row['low'], row['open'], row['close']
                
                # High should be >= max(open, close)
                assert high >= max(open_price, close), f"High {high} < max(open {open_price}, close {close}) at {idx}"
                
                # Low should be <= min(open, close)  
                assert low <= min(open_price, close), f"Low {low} > min(open {open_price}, close {close}) at {idx}"
                
                # All prices should be positive
                assert high > 0 and low > 0 and open_price > 0 and close > 0, f"Non-positive prices at {idx}"
                
                # Volume should be positive integer
                assert row['volume'] > 0, f"Non-positive volume at {idx}"
                assert isinstance(row['volume'], (int, float)), f"Volume not numeric at {idx}"
    
    def test_interval_frequency_mapping(self):
        """Test that different intervals produce appropriate data frequency"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = PolygonAdapter()
            
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
            
            # Weekly data
            weekly_result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1wk",
                start="2024-01-01",
                end="2024-02-01",  # 1 month
                live=False
            )
            
            daily_df = daily_result["AAPL"]
            hourly_df = hourly_result["AAPL"]
            weekly_df = weekly_result["AAPL"]
            
            # Daily should have fewer records than hourly for same time span
            assert len(daily_df) <= 5, f"Daily data has {len(daily_df)} records, expected <= 5"
            assert len(hourly_df) >= 20, f"Hourly data has {len(hourly_df)} records, expected >= 20"
            assert len(weekly_df) <= 5, f"Weekly data has {len(weekly_df)} records, expected <= 5"
    
    def test_interval_equivalence(self):
        """Test that equivalent intervals produce same mock data"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = PolygonAdapter()
            
            # 1h and 60min should produce equivalent results
            result_1h = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1h",
                start="2024-01-01",
                end="2024-01-03",
                live=False
            )
            
            result_60min = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="60min", 
                start="2024-01-01",
                end="2024-01-03",
                live=False
            )
            
            df_1h = result_1h["AAPL"]
            df_60min = result_60min["AAPL"]
            
            # Should have same structure
            assert len(df_1h.columns) == len(df_60min.columns)
            assert list(df_1h.columns) == list(df_60min.columns)
            
            # Should have same timezone
            assert str(df_1h.index.tz) == str(df_60min.index.tz)
    
    def test_metadata_attributes(self):
        """Test that mock data includes proper metadata attributes"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = PolygonAdapter()
            
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
            assert df.attrs['provider'] == 'polygon_mock'
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
                source="polygon",
                live=False
            )
            
            assert result.ok == True, f"Tool failed: {result.errors}"
            assert len(result.warnings) == 0
            
            # Check summary data
            summary = result.data['summary']
            assert summary['symbols_loaded'] == 2
            assert summary['provider'] == 'polygon'
            assert summary['interval'] == '1d'
            assert isinstance(summary['receipt_hashes'], list)
            assert len(summary['receipt_hashes']) == 0  # No receipts for offline mode
            
            # Check panel metadata
            panel_meta = result.data['panel']['metadata']
            assert panel_meta['source'] == 'polygon'
            assert panel_meta['live_mode'] == False
            assert 'utc_timestamp' in panel_meta
            assert 'load_time' in panel_meta
            
            # Check that provider_info shows mock providers
            provider_info = panel_meta['provider_info']
            assert 'AAPL' in provider_info
            assert 'MSFT' in provider_info
            assert provider_info['AAPL']['provider'] == 'polygon_mock'
            assert provider_info['MSFT']['provider'] == 'polygon_mock'
    
    def test_unsupported_interval_error(self):
        """Test that unsupported intervals raise clear errors"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = PolygonAdapter()
            
            with pytest.raises(ValueError) as exc_info:
                adapter.load_ohlcv(
                    symbols=["AAPL"],
                    interval="2h",  # Not in INTERVAL_MAP
                    start="2024-01-01",
                    end="2024-01-02",
                    live=False
                )
            
            error_msg = str(exc_info.value).lower()
            assert "unsupported interval" in error_msg
            assert "2h" in error_msg
            assert "supported:" in error_msg
    
    def test_large_date_range_limited(self):
        """Test that large date ranges are limited to reasonable size"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = PolygonAdapter()
            
            # Request a very large date range
            result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1min",
                start="2020-01-01",
                end="2024-01-01",  # 4 years of 1-minute data
                live=False
            )
            
            df = result["AAPL"]
            
            # Should be limited to 10000 records max
            assert len(df) <= 10000, f"Mock data not limited: {len(df)} records"
    
    def test_no_network_calls_offline(self):
        """Test that no network calls are made in offline mode"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            with patch('requests.get') as mock_get:
                adapter = PolygonAdapter()
                
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
    
    def test_polygon_volume_higher_than_alpha_vantage(self):
        """Test that Polygon mock data has higher volume than Alpha Vantage (2M vs 1M base)"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = PolygonAdapter()
            
            result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1d",
                start="2024-01-01",
                end="2024-01-05",
                live=False
            )
            
            df = result["AAPL"]
            
            # Polygon should have higher base volume (2M vs Alpha Vantage's 1M)
            avg_volume = df['volume'].mean()
            assert avg_volume > 1500000, f"Expected higher volume for Polygon, got {avg_volume}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])