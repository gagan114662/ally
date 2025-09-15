#!/usr/bin/env python3
"""
Alpha Vantage schema validation tests - verify data structure and formats
"""

import os
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import patch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ally.adapters.data.alpha_vantage_adapter import AlphaVantageAdapter
from ally.tools.data import data_load_ohlcv
from ally.schemas.data import LoadOHLCVIn, DataPanel


class TestAlphaVantageSchema:
    
    def test_input_schema_validation(self):
        """Test LoadOHLCVIn schema accepts Alpha Vantage parameters"""
        
        # Valid Alpha Vantage input
        valid_input = LoadOHLCVIn(
            symbols=["AAPL", "MSFT"],
            interval="1h",
            start="2024-01-01",
            end="2024-01-05",
            source="alpha_vantage",
            live=False,
            api_key="test_key"
        )
        
        assert valid_input.symbols == ["AAPL", "MSFT"]
        assert valid_input.source == "alpha_vantage"
        assert valid_input.live == False
        assert valid_input.api_key == "test_key"
        
        # Test with live=True
        live_input = LoadOHLCVIn(
            symbols=["AAPL"],
            interval="1d", 
            start="2024-01-01",
            end="2024-01-05",
            source="alpha_vantage",
            live=True,
            api_key="demo_key"
        )
        
        assert live_input.live == True
        
        # Test defaults
        default_input = LoadOHLCVIn(
            symbols=["AAPL"],
            interval="1d",
            start="2024-01-01", 
            end="2024-01-05"
        )
        
        assert default_input.source == "mock"  # Default
        assert default_input.live == False  # Default
        assert default_input.api_key is None  # Default
    
    def test_dataframe_structure_schema(self):
        """Test that returned DataFrames have correct structure"""
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
            
            # Required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
            for col in required_columns:
                assert col in df.columns, f"Missing column: {col}"
            
            # Data types
            assert pd.api.types.is_datetime64_any_dtype(df['timestamp']), "timestamp not datetime"
            assert pd.api.types.is_numeric_dtype(df['open']), "open not numeric"
            assert pd.api.types.is_numeric_dtype(df['high']), "high not numeric"  
            assert pd.api.types.is_numeric_dtype(df['low']), "low not numeric"
            assert pd.api.types.is_numeric_dtype(df['close']), "close not numeric"
            assert pd.api.types.is_integer_dtype(df['volume']), "volume not integer"
            assert pd.api.types.is_string_dtype(df['symbol']), "symbol not string"
            
            # All symbols should match
            assert all(df['symbol'] == "AAPL"), "Symbol column inconsistent"
    
    def test_data_panel_schema_compliance(self):
        """Test that DataPanel schema is properly formed"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            
            result = data_load_ohlcv(
                symbols=["AAPL", "MSFT"],
                interval="1d",
                start="2024-01-01",
                end="2024-01-05", 
                source="alpha_vantage",
                live=False
            )
            
            # Validate DataPanel structure can be created
            panel_data = result.data['panel']
            panel = DataPanel(**panel_data)
            
            # Check required fields
            assert panel.symbols == ["AAPL", "MSFT"]
            assert panel.interval == "1d"
            assert isinstance(panel.start_date, datetime)
            assert isinstance(panel.end_date, datetime)
            assert panel.total_rows >= 0
            assert panel.index_column == "timestamp"  # Default
            
            # Check aligned_data structure
            assert "AAPL" in panel.aligned_data
            assert "MSFT" in panel.aligned_data
            
            # Each symbol should have list of records
            for symbol in panel.symbols:
                records = panel.aligned_data[symbol]
                assert isinstance(records, list)
                
                if records:  # If data exists
                    first_record = records[0]
                    assert isinstance(first_record, dict)
                    
                    # Check required fields in record
                    required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
                    for field in required_fields:
                        assert field in first_record, f"Missing field {field} in record"
    
    def test_metadata_schema_structure(self):
        """Test metadata has expected structure and fields"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            
            result = data_load_ohlcv(
                symbols=["AAPL", "MSFT"],
                interval="1h",
                start="2024-01-01",
                end="2024-01-05",
                source="alpha_vantage", 
                live=False
            )
            
            metadata = result.data['panel']['metadata']
            
            # Required metadata fields
            required_fields = [
                'source', 'load_time', 'symbols_requested', 'symbols_loaded',
                'provider_info', 'receipt_hashes', 'live_mode', 'utc_timestamp'
            ]
            
            for field in required_fields:
                assert field in metadata, f"Missing metadata field: {field}"
            
            # Field type validations
            assert metadata['source'] == 'alpha_vantage'
            assert isinstance(metadata['load_time'], (int, float))
            assert isinstance(metadata['symbols_requested'], list)
            assert isinstance(metadata['symbols_loaded'], list)
            assert isinstance(metadata['provider_info'], dict)
            assert isinstance(metadata['receipt_hashes'], list)
            assert isinstance(metadata['live_mode'], bool)
            assert isinstance(metadata['utc_timestamp'], str)
            
            # UTC timestamp should be valid ISO format
            datetime.fromisoformat(metadata['utc_timestamp'].replace('Z', '+00:00'))
            
            # Provider info structure
            for symbol in metadata['symbols_loaded']:
                if symbol in metadata['provider_info']:
                    provider_info = metadata['provider_info'][symbol]
                    assert 'provider' in provider_info
                    assert 'fetched_at' in provider_info
                    # receipt_hash may be None for offline mode
    
    def test_summary_schema_structure(self):
        """Test summary data has expected structure"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            
            result = data_load_ohlcv(
                symbols=["AAPL"],
                interval="1d",
                start="2024-01-01", 
                end="2024-01-05",
                source="alpha_vantage",
                live=False
            )
            
            summary = result.data['summary']
            
            # Required summary fields
            required_fields = [
                'symbols_loaded', 'total_rows', 'date_range', 
                'interval', 'provider', 'receipt_hashes'
            ]
            
            for field in required_fields:
                assert field in summary, f"Missing summary field: {field}"
            
            # Type validations
            assert isinstance(summary['symbols_loaded'], int)
            assert isinstance(summary['total_rows'], int)
            assert isinstance(summary['date_range'], str)
            assert summary['interval'] == "1d"
            assert summary['provider'] == "alpha_vantage"
            assert isinstance(summary['receipt_hashes'], list)
            
            # Date range format
            assert " to " in summary['date_range']
            assert "2024-01-01" in summary['date_range']
            assert "2024-01-05" in summary['date_range']
    
    def test_tool_result_schema_compliance(self):
        """Test that ToolResult structure is valid"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            
            result = data_load_ohlcv(
                symbols=["AAPL"],
                interval="1d",
                start="2024-01-01",
                end="2024-01-05",
                source="alpha_vantage",
                live=False
            )
            
            # ToolResult structure
            assert hasattr(result, 'ok')
            assert hasattr(result, 'status')
            assert hasattr(result, 'data')
            assert hasattr(result, 'errors')
            assert hasattr(result, 'meta')
            assert hasattr(result, 'warnings')
            
            assert result.ok == True
            assert isinstance(result.data, dict)
            assert isinstance(result.errors, list)
            assert isinstance(result.warnings, list)
            
            # Meta structure
            assert hasattr(result.meta, 'ts')
            assert hasattr(result.meta, 'duration_ms')
            assert hasattr(result.meta, 'receipt_hash')  # May be None
            
            # Data structure
            assert 'panel' in result.data
            assert 'summary' in result.data
    
    def test_column_data_ranges_realistic(self):
        """Test that numeric columns have realistic value ranges"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = AlphaVantageAdapter()
            
            result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1d",
                start="2024-01-01",
                end="2024-01-10",
                live=False
            )
            
            df = result["AAPL"]
            
            # Price columns should be positive and reasonable for stocks
            assert df['open'].min() > 0, "Open prices should be positive"
            assert df['high'].min() > 0, "High prices should be positive"
            assert df['low'].min() > 0, "Low prices should be positive"
            assert df['close'].min() > 0, "Close prices should be positive"
            
            # Prices shouldn't be extreme
            assert df['close'].max() < 10000, "Prices seem too high"
            assert df['close'].min() > 1, "Prices seem too low"
            
            # Volume should be positive integers
            assert df['volume'].min() > 0, "Volume should be positive"
            assert all(df['volume'] % 1 == 0), "Volume should be integers"
            
            # Timestamps should be in order
            assert df['timestamp'].is_monotonic_increasing, "Timestamps not in order"
    
    def test_interval_mapping_consistency(self):
        """Test that intervals map to expected data frequency"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = AlphaVantageAdapter()
            
            # Test various interval formats
            intervals_to_test = ["1d", "1h", "daily", "60min"]
            
            for interval in intervals_to_test:
                result = adapter.load_ohlcv(
                    symbols=["AAPL"],
                    interval=interval,
                    start="2024-01-01",
                    end="2024-01-02",
                    live=False
                )
                
                assert "AAPL" in result, f"No data for interval {interval}"
                df = result["AAPL"]
                assert len(df) > 0, f"Empty data for interval {interval}"
                
                # All records should have same symbol
                assert all(df['symbol'] == "AAPL"), f"Inconsistent symbols for {interval}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])