#!/usr/bin/env python3
"""
Polygon schema validation tests - verify data structure and formats
"""

import os
import pandas as pd
import pytest

# Ensure CI stays dry
os.environ["ALLY_LIVE"] = os.environ.get("ALLY_LIVE", "0")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ally.tools.data import data_load_ohlcv


REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]


def _extract_frames(res):
    """
    Robustly get the {symbol: DataFrame} mapping from the tool result.
    Supports either attribute-based or dict-based returns.
    """
    # Most likely: a result object with .data containing panel with aligned_data
    if hasattr(res, 'data') and 'panel' in res.data:
        # Convert aligned_data back to DataFrames
        aligned_data = res.data['panel']['aligned_data']
        frames = {}
        for symbol, records in aligned_data.items():
            if records:
                df = pd.DataFrame(records)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                frames[symbol] = df
        return frames
    
    # Fallback for direct dict access
    data = getattr(res, "data", None) or getattr(res, "frames", None)
    if data is None and isinstance(res, dict):
        data = res.get("data") or res.get("frames")
    assert isinstance(data, dict), "Expected a dict of {symbol: DataFrame}"
    return data


def _assert_ohlcv_schema(df: pd.DataFrame):
    # Columns present (check original columns, not index)
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        assert col in df.columns, f"Missing required column '{col}'"

    # Index: datetime, tz-aware, UTC, monotonic
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be a DatetimeIndex"
    assert df.index.tz is not None, "Index must be timezone-aware"
    assert str(df.index.tz) in ("UTC", "tzutc()", "UTC+00:00"), "Index must be in UTC"
    assert df.index.is_monotonic_increasing, "Index must be strictly monotonic increasing"

    # Dtypes: numeric for OHLCV
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' must be numeric"

    # No NaNs in required columns (allow NaNs only if explicitly documented)
    assert not df[numeric_cols].isna().any().any(), "NaNs found in required OHLCV columns"


@pytest.mark.parametrize(
    "interval",
    ["1d", "daily", "1h", "60min", "15min", "1wk", "1mo"],  # cover day/hour/min/week/month
)
def test_polygon_schema_offline(interval):
    """
    Offline schema validation for Polygon adapter. No network calls.
    """
    res = data_load_ohlcv(
        symbols=["AAPL"],
        interval=interval,
        start="2024-01-01",
        end="2024-01-05",
        source="polygon",
        live=False,  # force offline deterministic path
    )

    # Basic contract: must be successful
    assert res.ok == True, f"Tool failed: {res.errors}"

    # Basic contract: must include provider metadata
    if hasattr(res, 'data') and 'panel' in res.data:
        meta = res.data['panel']['metadata']
        assert meta.get('source') == 'polygon', "Source should be polygon"
        assert meta.get('live_mode') == False, "Should be offline mode"

    frames = _extract_frames(res)
    assert "AAPL" in frames, "Result must include requested symbol"
    df = frames["AAPL"]
    _assert_ohlcv_schema(df)


def test_polygon_interval_mapping_equivalence():
    """
    1h and 60min should map to the same Polygon params under the hood.
    We don't inspect the private mapping; we assert output is shape-equivalent.
    """
    res_h = data_load_ohlcv(
        symbols=["AAPL"],
        interval="1h",
        start="2024-01-01",
        end="2024-01-03",
        source="polygon",
        live=False,
    )
    res_60 = data_load_ohlcv(
        symbols=["AAPL"],
        interval="60min",
        start="2024-01-01",
        end="2024-01-03",
        source="polygon",
        live=False,
    )

    fh = _extract_frames(res_h)["AAPL"]
    f60 = _extract_frames(res_60)["AAPL"]

    # Same required columns and index type
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    assert all(col in fh.columns for col in ohlcv_cols)
    assert all(col in f60.columns for col in ohlcv_cols)
    assert isinstance(fh.index, pd.DatetimeIndex) and isinstance(f60.index, pd.DatetimeIndex)
    
    # Both should have same timezone (UTC)
    assert fh.index.tz is not None and f60.index.tz is not None
    assert str(fh.index.tz) == str(f60.index.tz)


def test_polygon_unsupported_interval_raises():
    """
    Intervals that aren't in the adapter map (e.g., '2h', '2min') should raise a clear error.
    """
    with pytest.raises((KeyError, ValueError)) as ei:
        data_load_ohlcv(
            symbols=["AAPL"],
            interval="2h",  # not supported by the mapping
            start="2024-01-01",
            end="2024-01-02",
            source="polygon",
            live=False,
        )
    assert "interval" in str(ei.value).lower() or "unsupported" in str(ei.value).lower()


def test_polygon_multi_symbol_offline():
    """
    Multi-symbol requests should return a frame per symbol with the same schema.
    """
    res = data_load_ohlcv(
        symbols=["AAPL", "MSFT"],
        interval="1d",
        start="2024-01-01",
        end="2024-01-05",
        source="polygon",
        live=False,
    )
    
    assert res.ok == True, f"Tool failed: {res.errors}"
    
    frames = _extract_frames(res)
    for sym in ["AAPL", "MSFT"]:
        assert sym in frames, f"Missing symbol {sym}"
        _assert_ohlcv_schema(frames[sym])


def test_polygon_timezone_utc_enforcement():
    """
    Test that all timestamps are properly converted to UTC
    """
    res = data_load_ohlcv(
        symbols=["AAPL"],
        interval="1d",
        start="2024-01-01",
        end="2024-01-05",
        source="polygon",
        live=False,
    )
    
    frames = _extract_frames(res)
    df = frames["AAPL"]
    
    # Verify UTC timezone
    assert df.index.tz is not None, "Timestamps must be timezone-aware"
    assert str(df.index.tz) in ("UTC", "tzutc()", "UTC+00:00"), f"Expected UTC timezone, got {df.index.tz}"
    
    # Verify no naive timestamps
    assert not df.index.tz_localize(None).equals(df.index), "Timestamps should not be timezone-naive"


def test_polygon_ohlc_relationships():
    """
    Test that OHLC relationships are valid (high >= max(open,close), etc.)
    """
    res = data_load_ohlcv(
        symbols=["AAPL"],
        interval="1h",
        start="2024-01-01",
        end="2024-01-02",
        source="polygon",
        live=False,
    )
    
    frames = _extract_frames(res)
    df = frames["AAPL"]
    
    # Check OHLC relationships for each row
    for idx, row in df.iterrows():
        high, low, open_price, close = row['high'], row['low'], row['open'], row['close']
        
        # High should be >= max(open, close)
        assert high >= max(open_price, close), f"High {high} < max(open {open_price}, close {close}) at {idx}"
        
        # Low should be <= min(open, close)  
        assert low <= min(open_price, close), f"Low {low} > min(open {open_price}, close {close}) at {idx}"
        
        # All prices should be positive
        assert all(price > 0 for price in [high, low, open_price, close]), f"Negative prices at {idx}"
        
        # Volume should be non-negative integer
        assert row['volume'] >= 0, f"Negative volume at {idx}"
        assert isinstance(row['volume'], (int, float)), f"Volume not numeric at {idx}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])