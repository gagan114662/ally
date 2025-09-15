#!/usr/bin/env python3
"""
Polygon gating tests - ensure proper live mode enforcement
"""

import os
import pytest

# Ensure CI stays dry
os.environ["ALLY_LIVE"] = "0"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ally.tools.data import data_load_ohlcv


def _run_live(symbol="AAPL"):
    """Helper to try a live call (which must fail in CI)."""
    return data_load_ohlcv(
        symbols=[symbol],
        interval="1d",
        start="2024-01-01",
        end="2024-01-03",
        source="polygon",
        live=True,  # <- live path requested
    )


def test_polygon_live_blocked_when_ally_live_0(monkeypatch):
    """live=True must hard-fail if ALLY_LIVE != 1."""
    monkeypatch.setenv("ALLY_LIVE", "0")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    
    res = _run_live()
    
    # Should fail with gating error
    assert res.ok == False, "Live mode should be blocked when ALLY_LIVE=0"
    assert len(res.errors) > 0, "Should have error messages"
    
    error_msg = " ".join(res.errors).lower()
    assert "ally_live" in error_msg and ("live" in error_msg or "network" in error_msg), f"Expected gating error, got: {res.errors}"


def test_polygon_live_blocked_when_missing_key(monkeypatch):
    """Even if ALLY_LIVE=1, missing API key must hard-fail."""
    monkeypatch.setenv("ALLY_LIVE", "1")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    
    res = _run_live()
    
    # Should fail with API key error
    assert res.ok == False, "Live mode should be blocked when API key missing"
    assert len(res.errors) > 0, "Should have error messages"
    
    error_msg = " ".join(res.errors).lower()
    assert "api key" in error_msg and ("missing" in error_msg or "provided" in error_msg), f"Expected API key error, got: {res.errors}"


def test_polygon_live_blocked_with_placeholder_key(monkeypatch):
    """Placeholder keys must be rejected explicitly."""
    monkeypatch.setenv("ALLY_LIVE", "1")
    # Common placeholders you might forbid in code:
    monkeypatch.setenv("POLYGON_API_KEY", "your_api_key_here")
    
    res = _run_live()
    
    # Should fail with placeholder key error
    assert res.ok == False, "Live mode should be blocked with placeholder API key"
    assert len(res.errors) > 0, "Should have error messages"
    
    error_msg = " ".join(res.errors).lower()
    assert "api key" in error_msg and ("placeholder" in error_msg or "invalid" in error_msg), f"Expected placeholder key error, got: {res.errors}"


def test_polygon_offline_does_not_require_key(monkeypatch):
    """live=False path must work offline and not require any key."""
    monkeypatch.setenv("ALLY_LIVE", "0")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)

    res = data_load_ohlcv(
        symbols=["AAPL"],
        interval="1d",
        start="2024-01-01",
        end="2024-01-03",
        source="polygon",
        live=False,  # offline deterministic path
    )

    # Should succeed in offline mode
    assert res.ok == True, f"Offline mode should work without API key, got errors: {res.errors}"
    
    # Sanity: result object exists and includes minimal metadata
    assert 'panel' in res.data, "Should have panel data"
    meta = res.data['panel']['metadata']
    assert isinstance(meta, dict), "Metadata should be dict"
    assert meta.get('source') == 'polygon', "Source should be polygon"
    assert meta.get('live_mode') == False, "Should be offline mode"


def test_polygon_additional_placeholder_variants(monkeypatch):
    """Test additional placeholder key variants that should be rejected."""
    monkeypatch.setenv("ALLY_LIVE", "1")
    
    placeholder_keys = [
        "changeme",
        "your_polygon_api_key_here", 
        "INSERT_YOUR_KEY_HERE",
        "",  # empty string
        "demo",
        "test_key"
    ]
    
    for placeholder in placeholder_keys:
        monkeypatch.setenv("POLYGON_API_KEY", placeholder)
        
        res = _run_live()
        
        assert res.ok == False, f"Placeholder key '{placeholder}' should be rejected"
        
        if res.errors:
            error_msg = " ".join(res.errors).lower()
            assert "api key" in error_msg and ("placeholder" in error_msg or "invalid" in error_msg or "empty" in error_msg), \
                f"Expected placeholder rejection for '{placeholder}', got: {res.errors}"


def test_polygon_env_var_pickup(monkeypatch):
    """Test that adapter picks up API key from POLYGON_API_KEY environment variable."""
    monkeypatch.setenv("ALLY_LIVE", "0")  # Keep offline for testing
    monkeypatch.setenv("POLYGON_API_KEY", "env_test_key")
    
    # Test offline mode works with env var key
    res = data_load_ohlcv(
        symbols=["MSFT"],
        interval="1h", 
        start="2024-01-01",
        end="2024-01-02",
        source="polygon",
        live=False  # Offline mode
    )
    
    assert res.ok == True, f"Should work with env var API key in offline mode, got: {res.errors}"
    assert res.data['summary']['symbols_loaded'] == 1


def test_polygon_direct_adapter_gating():
    """Test gating at the adapter level directly."""
    from ally.adapters.data.polygon_adapter import PolygonAdapter
    from ally.utils.gating import LiveModeError
    
    # Test ALLY_LIVE=0 blocks live mode
    with monkeypatch.context() as m:
        m.setenv("ALLY_LIVE", "0")
        adapter = PolygonAdapter(api_key="test_key")
        
        with pytest.raises(LiveModeError) as exc_info:
            adapter.load_ohlcv(
                symbols=["AAPL"],
                interval="1d",
                start="2024-01-01",
                end="2024-01-05",
                live=True
            )
        
        assert "ALLY_LIVE=0" in str(exc_info.value)
        assert "Polygon.io" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])