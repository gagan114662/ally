"""
Tests for timestamp serialization utilities
"""

import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add Ally to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "Ally"))

from ally.utils.serialization import convert_timestamps


def test_dataframe_index_and_cols_to_iso():
    """Test DataFrame with datetime index and columns converts to ISO strings"""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")  # naive
    df = pd.DataFrame({"open": [1, 2, 3], "dt": idx}, index=idx)
    out = convert_timestamps(df)
    recs = out.to_dict(orient="records")
    
    for r in recs:
        assert "timestamp" in r
        assert r["timestamp"].endswith("Z")
        assert r["dt"].endswith("Z")
        # Verify it's a valid ISO string
        assert "T" in r["timestamp"]
        assert "T" in r["dt"]


def test_nested_structures_convert():
    """Test nested data structures with timestamps convert properly"""
    ts = pd.Timestamp("2024-02-03 12:00:00", tz="US/Eastern")
    obj = {
        "a": [ts, {"b": pd.Timestamp("2024-02-03 12:00:00")}], 
        "c": ts.tz_convert("UTC")
    }
    out = convert_timestamps(obj)
    
    assert out["a"][0].endswith("Z")
    assert out["a"][1]["b"].endswith("Z")
    assert out["c"].endswith("Z")


def test_series_conversion():
    """Test pandas Series conversion"""
    dates = pd.date_range("2024-01-01", periods=3, freq="H")
    series = pd.Series([1, 2, 3], index=dates)
    
    converted_series = convert_timestamps(series)
    assert all(isinstance(x, (int, float)) for x in converted_series.values)


def test_datetimeindex_conversion():
    """Test DatetimeIndex conversion"""
    idx = pd.date_range("2024-01-01", periods=3, freq="H")
    converted = convert_timestamps(idx)
    
    assert isinstance(converted, list)
    assert all(ts.endswith("Z") for ts in converted)
    assert len(converted) == 3


def test_mixed_types_preserved():
    """Test that non-datetime types are preserved"""
    obj = {
        "string": "test",
        "number": 42,
        "bool": True,
        "list": [1, 2, "three"],
        "datetime": pd.Timestamp("2024-01-01 12:00:00")
    }
    
    out = convert_timestamps(obj)
    
    assert out["string"] == "test"
    assert out["number"] == 42
    assert out["bool"] is True
    assert out["list"] == [1, 2, "three"]
    assert out["datetime"].endswith("Z")


def test_timezone_handling():
    """Test that timezones are properly converted to UTC"""
    # Test various timezone inputs
    timestamps = {
        "naive": pd.Timestamp("2024-01-01 12:00:00"),
        "utc": pd.Timestamp("2024-01-01 12:00:00", tz="UTC"),
        "eastern": pd.Timestamp("2024-01-01 12:00:00", tz="US/Eastern"),
        "pacific": pd.Timestamp("2024-01-01 12:00:00", tz="US/Pacific")
    }
    
    converted = convert_timestamps(timestamps)
    
    # All should end with Z (UTC)
    for key, value in converted.items():
        assert value.endswith("Z"), f"Timestamp for {key} should end with Z: {value}"
        assert "T" in value, f"Timestamp for {key} should contain T: {value}"


def test_empty_structures():
    """Test empty data structures don't break"""
    empty_cases = [
        {},
        [],
        pd.DataFrame(),
        pd.Series(dtype='float64')
    ]
    
    for case in empty_cases:
        result = convert_timestamps(case)
        # Should not raise an exception
        assert result is not None


def test_periodindex_and_existing_timestamp_col():
    """Test PeriodIndex with existing timestamp column"""
    try:
        idx = pd.period_range("2024-03", periods=3, freq="D")
        df = pd.DataFrame({"timestamp": ["pre", "existing", "values"], "x": [1, 2, 3]}, index=idx)
        out = convert_timestamps(df)
        recs = out.to_dict(orient="records")
        
        # Original 'timestamp' preserved
        assert "timestamp" in recs[0]
        assert recs[0]["timestamp"] == "pre"  # Original value preserved
        
        # Index became '__ally_idx__' 
        assert "__ally_idx__" in recs[0]
        assert recs[0]["__ally_idx__"].endswith("Z")
        
    except ImportError:
        # Skip if pandas doesn't have period_range
        pass


def test_numpy_datetime64_scalar_and_array():
    """Test numpy datetime64 scalars and arrays"""
    import numpy as np
    
    # Test scalar
    dt64 = np.datetime64("2024-01-01T12:34:56")
    assert to_iso_utc(dt64).endswith("Z")
    
    # Test array
    arr = [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
    out = convert_timestamps(arr)
    assert all(isinstance(x, str) and x.endswith("Z") for x in out)


def test_fast_path_datetime64_ns():
    """Test the fast path for datetime64[ns] dtype"""
    dates = pd.date_range("2024-01-01", periods=3, freq="h")
    df = pd.DataFrame({"dt": dates, "value": [1, 2, 3]})
    
    # Verify this is datetime64[ns]
    assert pd.api.types.is_datetime64_ns_dtype(df["dt"].dtype)
    
    out = convert_timestamps(df)
    recs = out.to_dict(orient="records")
    
    for r in recs:
        assert r["dt"].endswith("Z")
        assert "T" in r["dt"]


def test_collision_avoidance():
    """Test that index renaming avoids collisions"""
    idx = pd.date_range("2024-01-01", periods=2, freq="h")
    df = pd.DataFrame({
        "timestamp": ["user_ts_1", "user_ts_2"],
        "value": [1, 2]
    }, index=idx)
    
    out = convert_timestamps(df)
    recs = out.to_dict(orient="records")
    
    # User's timestamp column preserved
    assert recs[0]["timestamp"] == "user_ts_1"
    assert recs[1]["timestamp"] == "user_ts_2"
    
    # Index preserved under different name
    assert "__ally_idx__" in recs[0]
    assert recs[0]["__ally_idx__"].endswith("Z")


if __name__ == "__main__":
    # Import required modules
    from Ally.utils.serialization import to_iso_utc, convert_timestamps
    
    # Run tests manually
    print("Running timestamp serialization tests...")
    
    test_functions = [
        test_dataframe_index_and_cols_to_iso,
        test_nested_structures_convert,
        test_series_conversion,
        test_datetimeindex_conversion,
        test_mixed_types_preserved,
        test_timezone_handling,
        test_empty_structures,
        test_periodindex_and_existing_timestamp_col,
        test_numpy_datetime64_scalar_and_array,
        test_fast_path_datetime64_ns,
        test_collision_avoidance
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__} passed")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All timestamp serialization tests passed!")
    else:
        print("❌ Some tests failed")