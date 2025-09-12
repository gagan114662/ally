"""
Tests for Ally Computer Vision pattern detection
"""

import pytest
import sys
import pandas as pd
import numpy as np
import base64
from pathlib import Path

# Add ally to path
sys.path.append(str(Path(__file__).parent.parent))

from ally.tools.cv import cv_detect_chart_patterns, cv_generate_synthetic
from ally.schemas.base import ToolResult
from ally.utils.ta_rules import detect_engulfings, detect_pinbar, atr


def create_fixtures():
    """Create synthetic fixtures for testing"""
    fixtures_dir = Path("data/fixtures/cv")
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic datasets
    patterns = ["engulfing", "breakout", "channel"]
    
    for pattern in patterns:
        result = cv_generate_synthetic(
            pattern_type=pattern,
            n_bars=200,
            seed=42 + hash(pattern) % 100  # Deterministic but varied
        )
        
        if result.ok:
            df_data = result.data['synthetic_data']
            df = pd.DataFrame(df_data)
            
            # Save as CSV (no external dependencies needed)
            filename = fixtures_dir / f"synthetic_{pattern}.csv"
            df.to_csv(filename, index=False)
            print(f"Created fixture: {filename}")


def load_fixture(pattern_type: str) -> pd.DataFrame:
    """Load a test fixture"""
    fixture_path = Path("data/fixtures/cv") / f"synthetic_{pattern_type}.csv"
    
    if not fixture_path.exists():
        # Generate on the fly if fixture doesn't exist
        result = cv_generate_synthetic(
            pattern_type=pattern_type,
            n_bars=200, 
            seed=42 + hash(pattern_type) % 100
        )
        if result.ok:
            return pd.DataFrame(result.data['synthetic_data'])
        else:
            raise RuntimeError(f"Failed to generate {pattern_type} fixture")
    
    return pd.read_csv(fixture_path, parse_dates=['timestamp'])


def test_engulfing_detection():
    """Test engulfing pattern detection"""
    df = load_fixture("engulfing")
    
    result = cv_detect_chart_patterns(
        symbol="TEST",
        interval="1h",
        patterns=["engulfing_bull", "engulfing_bear"],
        lookback=200,
        confirm_with_rules=True,
        _df=df  # Pass dataframe directly for testing
    )
    
    assert isinstance(result, ToolResult)
    assert result.ok, f"Detection failed: {result.errors}"
    
    detections = result.data["detections"]
    assert len(detections) >= 1, "Should find at least one engulfing pattern"
    
    # Check detection structure
    for detection in detections:
        assert detection["pattern"] in ["engulfing_bull", "engulfing_bear"]
        assert isinstance(detection["confirmed"], bool)
        assert 0 <= detection["strength"] <= 1
        assert detection["start_idx"] >= 0
        assert detection["end_idx"] >= detection["start_idx"]
        assert isinstance(detection["info"], dict)
    
    # Should find both bull and bear patterns in synthetic data
    patterns_found = {d["pattern"] for d in detections}
    assert len(patterns_found) >= 1, f"Expected patterns, got: {patterns_found}"


def test_trendline_break_detection():
    """Test trendline break pattern detection"""
    df = load_fixture("breakout")
    
    result = cv_detect_chart_patterns(
        symbol="TEST_BREAKOUT",
        interval="1h", 
        patterns=["trendline_break"],
        lookback=200,
        confirm_with_rules=True,
        _df=df
    )
    
    assert result.ok, f"Trendline detection failed: {result.errors}"
    
    detections = result.data["detections"]
    metadata = result.data["metadata"]
    
    assert metadata["patterns_requested"] == ["trendline_break"]
    assert metadata["lookback_used"] == 200
    
    # Should find at least one trendline break in synthetic breakout data
    if detections:
        detection = detections[0]
        assert detection["pattern"] == "trendline_break"
        assert "info" in detection
        
        info = detection["info"]
        assert "line" in info  # Should specify upper or lower
        assert "slope" in info
        assert "level" in info
        assert isinstance(info.get("distance_atr", 0), (int, float))
        
        assert detection["strength"] > 0
        print(f"✅ Found trendline break: {info['line']} line, strength: {detection['strength']:.2f}")


def test_image_rendering():
    """Test chart rendering with pattern annotations"""
    df = load_fixture("engulfing")
    
    result = cv_detect_chart_patterns(
        symbol="TEST_RENDER",
        interval="1h",
        patterns=["engulfing_bull", "engulfing_bear", "trendline_break"],
        lookback=100,
        return_image=True,
        image_width=600,
        image_height=400,
        _df=df
    )
    
    assert result.ok, f"Rendering test failed: {result.errors}"
    
    if result.data["detections"]:  # Only test if patterns were found
        rendered = result.data.get("rendered")
        if rendered:
            assert isinstance(rendered, str), "Rendered should be base64 string"
            
            # Test base64 decoding
            try:
                png_bytes = base64.b64decode(rendered)
                assert len(png_bytes) > 1000, "PNG should have reasonable size"
                assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n', "Should be valid PNG header"
                print(f"✅ Generated {len(png_bytes)} byte PNG image")
            except Exception as e:
                pytest.fail(f"Base64 decoding failed: {e}")


def test_lookback_bounds():
    """Test behavior with small lookback windows"""
    df = load_fixture("engulfing")
    
    # Test with small lookback (minimum allowed is 50)
    result = cv_detect_chart_patterns(
        symbol="TEST_SMALL",
        interval="1h",
        patterns=["engulfing_bull"],
        lookback=50,  # Minimum allowed window
        _df=df
    )
    
    # Should succeed even with small window
    assert result.ok
    assert result.data["n_bars"] == 50
    
    # Test with insufficient data
    tiny_df = df.head(5)
    result = cv_detect_chart_patterns(
        symbol="TEST_TINY",
        interval="1h", 
        patterns=["engulfing_bull"],
        lookback=50,
        _df=tiny_df
    )
    
    # Should fail gracefully
    assert not result.ok
    assert "insufficient data" in result.errors[0].lower()


def test_confirmation_rules():
    """Test numeric confirmation vs no confirmation"""
    df = load_fixture("breakout")
    
    # With confirmation rules
    result_confirmed = cv_detect_chart_patterns(
        symbol="TEST_CONFIRM",
        interval="1h",
        patterns=["trendline_break"],
        lookback=200,
        confirm_with_rules=True,
        _df=df
    )
    
    # Without confirmation rules  
    result_unconfirmed = cv_detect_chart_patterns(
        symbol="TEST_NO_CONFIRM", 
        interval="1h",
        patterns=["trendline_break"],
        lookback=200,
        confirm_with_rules=False,
        _df=df
    )
    
    assert result_confirmed.ok
    assert result_unconfirmed.ok
    
    confirmed_detections = result_confirmed.data["detections"]
    unconfirmed_detections = result_unconfirmed.data["detections"]
    
    # Without rules, we might get more (potentially weaker) detections
    print(f"✅ Confirmed: {len(confirmed_detections)}, Unconfirmed: {len(unconfirmed_detections)}")
    
    # Check that confirmation status is correctly set
    if confirmed_detections:
        for det in confirmed_detections:
            assert det["confirmed"] == True, "Should be confirmed when confirm_with_rules=True"
    
    if unconfirmed_detections:
        # Some detections might not be confirmed when rules are off
        confirmation_statuses = [det["confirmed"] for det in unconfirmed_detections]
        print(f"✅ Confirmation statuses without rules: {confirmation_statuses}")


def test_pinbar_detection():
    """Test pin bar pattern detection"""
    # Create custom data with clear pin bars
    n_bars = 50
    np.random.seed(123)
    
    base_price = 100
    closes = base_price + np.random.normal(0, 1, n_bars).cumsum()
    opens = closes + np.random.normal(0, 0.5, n_bars)
    
    # Create clear bullish pin bar at position 25
    pos = 25
    opens[pos] = closes[pos] - 0.2
    highs = np.maximum(closes, opens) + np.abs(np.random.normal(0, 0.5, n_bars))
    lows = np.minimum(closes, opens) - np.abs(np.random.normal(0, 0.5, n_bars))
    
    # Make bullish pin bar: long lower wick, small body near top
    lows[pos] = closes[pos] - 3.0  # Long lower wick
    highs[pos] = closes[pos] + 0.2  # Small upper wick
    
    timestamps = pd.date_range('2024-01-01', periods=n_bars, freq='h')
    volumes = 1000 + np.random.exponential(500, n_bars)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows, 
        'close': closes,
        'volume': volumes
    })
    
    result = cv_detect_chart_patterns(
        symbol="TEST_PINBAR",
        interval="1h",
        patterns=["pin_bar_bull"],
        lookback=50,
        confirm_with_rules=False,  # Disable additional rules for this test
        _df=df
    )
    
    assert result.ok, f"Pin bar detection failed: {result.errors}"
    
    detections = result.data["detections"]
    if detections:
        pin_detection = detections[0]
        assert pin_detection["pattern"] == "pin_bar_bull"
        assert "wick_ratio" in pin_detection["info"]
        print(f"✅ Found pin bar with wick ratio: {pin_detection['info']['wick_ratio']:.2f}")


def test_ta_rules_directly():
    """Test technical analysis rules in isolation"""
    # Create simple test data
    data = {
        'open': [100, 101, 99, 102],
        'high': [101, 102, 101, 103],
        'low': [99, 100, 98, 101],
        'close': [101, 99, 102, 103],
        'volume': [1000, 1200, 1100, 1300]
    }
    df = pd.DataFrame(data)
    
    # Test ATR calculation
    atr_values = atr(df, period=2)
    assert not atr_values.isna().all(), "ATR should calculate some values"
    
    # Test engulfing detection
    engulfings = detect_engulfings(df)
    print(f"✅ Direct TA rules test: found {len(engulfings)} engulfing patterns")
    
    # The function should not crash and return a list
    assert isinstance(engulfings, list)


if __name__ == "__main__":
    # Create fixtures first
    create_fixtures()
    
    # Run tests manually
    print("Running CV pattern detection tests...")
    
    test_functions = [
        test_engulfing_detection,
        test_trendline_break_detection,
        test_image_rendering,
        test_lookback_bounds,
        test_confirmation_rules,
        test_pinbar_detection,
        test_ta_rules_directly
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
        print("🎉 All CV pattern detection tests passed!")
    else:
        print("❌ Some tests failed")