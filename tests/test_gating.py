#!/usr/bin/env python3
"""
Tests for gating system - ensure live mode is properly controlled
"""

import os
import pytest
from unittest.mock import patch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ally.utils.gating import (
    check_live_mode_allowed, 
    is_live_mode_enabled, 
    require_offline_mode,
    LiveModeError,
    get_live_mode_status
)


class TestLiveModeGating:
    
    def test_offline_mode_always_allowed(self):
        """Test that offline mode (live=False) always works regardless of ALLY_LIVE"""
        # Test with ALLY_LIVE=0
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            check_live_mode_allowed(live=False, api_key=None, service_name="test")
        
        # Test with ALLY_LIVE=1
        with patch.dict(os.environ, {"ALLY_LIVE": "1"}):
            check_live_mode_allowed(live=False, api_key=None, service_name="test")
        
        # Test with ALLY_LIVE unset
        with patch.dict(os.environ, {}, clear=True):
            check_live_mode_allowed(live=False, api_key=None, service_name="test")
    
    def test_live_mode_requires_ally_live_1(self):
        """Test that live=True fails if ALLY_LIVE != 1"""
        # Test with ALLY_LIVE=0
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            with pytest.raises(LiveModeError) as exc_info:
                check_live_mode_allowed(live=True, api_key="valid_key", service_name="test")
            assert "ALLY_LIVE=0" in str(exc_info.value)
        
        # Test with ALLY_LIVE unset
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(LiveModeError) as exc_info:
                check_live_mode_allowed(live=True, api_key="valid_key", service_name="test")
            assert "ALLY_LIVE=0" in str(exc_info.value)  # Default is 0
    
    def test_live_mode_requires_valid_api_key(self):
        """Test that live=True requires a valid API key"""
        with patch.dict(os.environ, {"ALLY_LIVE": "1"}):
            # Test with None API key
            with pytest.raises(LiveModeError) as exc_info:
                check_live_mode_allowed(live=True, api_key=None, service_name="test")
            assert "no API key provided" in str(exc_info.value)
            
            # Test with placeholder API key
            with pytest.raises(LiveModeError) as exc_info:
                check_live_mode_allowed(live=True, api_key="your_api_key_here", service_name="test")
            assert "invalid API key" in str(exc_info.value)
            
            # Test with empty API key
            with pytest.raises(LiveModeError) as exc_info:
                check_live_mode_allowed(live=True, api_key="", service_name="test")
            assert "invalid API key" in str(exc_info.value)
    
    def test_live_mode_success_with_valid_requirements(self):
        """Test that live=True works with ALLY_LIVE=1 and valid API key"""
        with patch.dict(os.environ, {"ALLY_LIVE": "1"}):
            # Should not raise any exception
            check_live_mode_allowed(live=True, api_key="valid_api_key_123", service_name="test")
    
    def test_is_live_mode_enabled(self):
        """Test the is_live_mode_enabled helper function"""
        # Test with ALLY_LIVE=1
        with patch.dict(os.environ, {"ALLY_LIVE": "1"}):
            assert is_live_mode_enabled() == True
        
        # Test with ALLY_LIVE=0
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            assert is_live_mode_enabled() == False
        
        # Test with ALLY_LIVE unset
        with patch.dict(os.environ, {}, clear=True):
            assert is_live_mode_enabled() == False
    
    def test_require_offline_mode(self):
        """Test the require_offline_mode helper function"""
        # Should work in offline mode
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            require_offline_mode("test_operation")
        
        # Should work with ALLY_LIVE unset
        with patch.dict(os.environ, {}, clear=True):
            require_offline_mode("test_operation")
        
        # Should fail in live mode
        with patch.dict(os.environ, {"ALLY_LIVE": "1"}):
            with pytest.raises(LiveModeError) as exc_info:
                require_offline_mode("test_operation")
            assert "not allowed in live mode" in str(exc_info.value)
    
    def test_get_live_mode_status(self):
        """Test the get_live_mode_status helper function"""
        # Test with ALLY_LIVE=1
        with patch.dict(os.environ, {"ALLY_LIVE": "1"}):
            status = get_live_mode_status()
            assert status["ally_live_env"] == "1"
            assert status["is_live_enabled"] == True
            assert status["is_offline_mode"] == False
        
        # Test with ALLY_LIVE=0
        with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
            status = get_live_mode_status()
            assert status["ally_live_env"] == "0"
            assert status["is_live_enabled"] == False
            assert status["is_offline_mode"] == True


def test_tool_integration_example():
    """Example of how a tool would use gating"""
    
    def example_data_fetcher(symbol: str, live: bool = False, api_key: str = None):
        """Example tool that fetches data"""
        # Check gating first
        check_live_mode_allowed(live=live, api_key=api_key, service_name="DataProvider")
        
        if live:
            # Would make network call here
            return {"symbol": symbol, "price": 150.0, "source": "live"}
        else:
            # Return mock/cached data
            return {"symbol": symbol, "price": 100.0, "source": "mock"}
    
    # Test offline mode works
    with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
        result = example_data_fetcher("AAPL", live=False)
        assert result["source"] == "mock"
    
    # Test live mode blocked without ALLY_LIVE=1
    with patch.dict(os.environ, {"ALLY_LIVE": "0"}):
        with pytest.raises(LiveModeError):
            example_data_fetcher("AAPL", live=True, api_key="valid_key")
    
    # Test live mode works with proper gating
    with patch.dict(os.environ, {"ALLY_LIVE": "1"}):
        result = example_data_fetcher("AAPL", live=True, api_key="valid_key")
        assert result["source"] == "live"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])