#!/usr/bin/env python3
"""
Alpha Vantage gating tests - ensure proper live mode enforcement
"""

import os
import pytest
from unittest.mock import patch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ally.utils.gating import LiveModeError
from ally.adapters.data.alpha_vantage_adapter import AlphaVantageAdapter
from ally.tools.data import data_load_ohlcv


class TestAlphaVantageGating:
    
    def test_offline_mode_works_without_api_key(self):
        """Test that offline mode works without API key or ALLY_LIVE"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}, clear=True):
            adapter = AlphaVantageAdapter(api_key=None)
            
            # Should work in offline mode
            result = adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1d", 
                start="2024-01-01",
                end="2024-01-05",
                live=False
            )
            
            assert "AAPL" in result
            assert len(result["AAPL"]) > 0
            assert result["AAPL"].attrs['provider'] == 'alpha_vantage_mock'
    
    def test_live_mode_blocked_without_ally_live(self):
        """Test that live=True fails when ALLY_LIVE != 1"""
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            adapter = AlphaVantageAdapter(api_key="valid_test_key")
            
            with pytest.raises(LiveModeError) as exc_info:
                adapter.load_ohlcv(
                    symbols=["AAPL"],
                    interval="1d",
                    start="2024-01-01", 
                    end="2024-01-05",
                    live=True
                )
            
            assert "ALLY_LIVE=0" in str(exc_info.value)
            assert "Alpha Vantage" in str(exc_info.value)
    
    def test_live_mode_blocked_without_api_key(self):
        """Test that live=True fails when API key is missing"""
        with patch.dict(os.environ, {"ALLY_LIVE": "1"}):
            # Test with None API key
            adapter = AlphaVantageAdapter(api_key=None)
            
            with pytest.raises(LiveModeError) as exc_info:
                adapter.load_ohlcv(
                    symbols=["AAPL"],
                    interval="1d",
                    start="2024-01-01",
                    end="2024-01-05", 
                    live=True
                )
            
            assert "no API key provided" in str(exc_info.value)
    
    def test_live_mode_blocked_with_placeholder_key(self):
        """Test that live=True fails with placeholder API key"""
        with patch.dict(os.environ, {"ALLY_LIVE": "1"}):
            adapter = AlphaVantageAdapter(api_key="your_alpha_vantage_api_key_here")
            
            with pytest.raises(LiveModeError) as exc_info:
                adapter.load_ohlcv(
                    symbols=["AAPL"],
                    interval="1d",
                    start="2024-01-01",
                    end="2024-01-05",
                    live=True
                )
            
            assert "invalid API key" in str(exc_info.value)
    
    def test_tool_level_gating_integration(self):
        """Test gating integration at data_load_ohlcv tool level"""
        
        # Test offline mode works
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            result = data_load_ohlcv(
                symbols=["AAPL"],
                interval="1d",
                start="2024-01-01",
                end="2024-01-05",
                source="alpha_vantage",
                live=False
            )
            
            assert result.ok == True
            assert result.data['summary']['provider'] == 'alpha_vantage'
            assert result.data['summary']['symbols_loaded'] == 1
        
        # Test live mode blocked
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            result = data_load_ohlcv(
                symbols=["AAPL"],
                interval="1d", 
                start="2024-01-01",
                end="2024-01-05",
                source="alpha_vantage",
                live=True,
                api_key="test_key"
            )
            
            # Should fail with gating error
            assert result.ok == False
            assert any("ALLY_LIVE" in error for error in result.errors)
    
    def test_env_var_api_key_pickup(self):
        """Test that adapter picks up API key from environment"""
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "env_test_key", "ALLY_LIVE": "0"}):
            adapter = AlphaVantageAdapter()  # No explicit API key
            
            # Should use env var API key
            assert adapter.api_key == "env_test_key"
            
            # Offline mode should still work
            result = adapter.load_ohlcv(
                symbols=["MSFT"],
                interval="1h", 
                start="2024-01-01",
                end="2024-01-02",
                live=False
            )
            
            assert "MSFT" in result


def test_data_tool_schema_validation():
    """Test that data tool properly validates Alpha Vantage inputs"""
    
    # Test valid inputs
    with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
        result = data_load_ohlcv(
            symbols=["AAPL", "MSFT"],
            interval="1h",
            start="2024-01-01", 
            end="2024-01-05",
            source="alpha_vantage",
            live=False
        )
        
        assert result.ok == True
        assert result.data['summary']['symbols_loaded'] == 2
        assert 'receipt_hashes' in result.data['summary']  # Should be empty for offline
        assert result.data['panel']['metadata']['live_mode'] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])