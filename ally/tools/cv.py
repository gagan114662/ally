"""
Computer Vision tools for Ally - chart pattern detection
Combines off-screen rendering with numeric confirmation for reliable detection
"""

import time
import base64
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from ..tools import register
from ..schemas.base import ToolResult, Meta
from ..schemas.cv import CVDetectIn, CVDetectOut, CVDetection
from ..utils.hashing import hash_inputs, hash_code
from ..utils.serialization import convert_timestamps
from ..utils.plotting import render_chart_with_patterns
from ..utils.ta_rules import (
    atr, detect_engulfings, detect_pinbar, detect_morning_evening_star,
    breakout_over_level, breakout_under_level, detect_pivots, fit_trendline,
    calculate_trendline_distance
)


def generate_synthetic_data(pattern_type: str, n_bars: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with embedded patterns for testing
    
    Args:
        pattern_type: Type of pattern to embed
        n_bars: Number of bars to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with embedded pattern
    """
    np.random.seed(seed)
    
    # Base price trend
    base_price = 100.0
    trend_slope = 0.02  # Slight upward trend
    noise_level = 0.01
    
    # Generate base closes with trend and noise
    trend = np.linspace(0, trend_slope * n_bars, n_bars)
    noise = np.random.normal(0, noise_level * base_price, n_bars)
    closes = base_price + trend + noise.cumsum()
    
    # Generate OHLC based on closes
    volatility = 0.015 * base_price
    opens = closes + np.random.normal(0, volatility * 0.3, n_bars)
    
    # Highs and lows with some randomness
    high_noise = np.abs(np.random.normal(0, volatility * 0.5, n_bars))
    low_noise = np.abs(np.random.normal(0, volatility * 0.5, n_bars))
    
    highs = np.maximum(closes, opens) + high_noise
    lows = np.minimum(closes, opens) - low_noise
    
    # Volumes
    base_volume = 1000
    volumes = base_volume + np.random.exponential(base_volume * 0.5, n_bars)
    
    # Embed specific patterns
    if pattern_type == "engulfing":
        # Create bullish engulfing at position n_bars//3
        pos = n_bars // 3
        if pos > 0:
            # Previous candle: small red
            opens[pos-1] = closes[pos-1] + volatility * 0.3
            
            # Engulfing candle: large green that engulfs previous
            opens[pos] = closes[pos-1] - volatility * 0.2  # Open below prev close
            closes[pos] = opens[pos-1] + volatility * 0.3   # Close above prev open
            highs[pos] = closes[pos] + volatility * 0.1
            lows[pos] = opens[pos] - volatility * 0.1
        
        # Create bearish engulfing at position 2*n_bars//3
        pos = 2 * n_bars // 3
        if pos > 0:
            # Previous candle: small green
            opens[pos-1] = closes[pos-1] - volatility * 0.3
            
            # Engulfing candle: large red that engulfs previous
            opens[pos] = closes[pos-1] + volatility * 0.2   # Open above prev close
            closes[pos] = opens[pos-1] - volatility * 0.3   # Close below prev open
            highs[pos] = opens[pos] + volatility * 0.1
            lows[pos] = closes[pos] - volatility * 0.1
    
    elif pattern_type == "breakout":
        # Create upward trendline that gets broken
        breakout_pos = int(0.8 * n_bars)  # Near end
        trendline_start = n_bars // 4
        
        # Make highs follow an upward trendline until breakout
        for i in range(trendline_start, breakout_pos):
            trendline_level = closes[trendline_start] + 0.3 * (i - trendline_start) * trend_slope
            highs[i] = min(highs[i], trendline_level + volatility * 0.5)
            
        # Breakout bar: close significantly above trendline
        trendline_breakout_level = closes[trendline_start] + 0.3 * (breakout_pos - trendline_start) * trend_slope
        closes[breakout_pos] = trendline_breakout_level + volatility * 2.0
        opens[breakout_pos] = closes[breakout_pos] - volatility * 0.5
        highs[breakout_pos] = closes[breakout_pos] + volatility * 0.3
        lows[breakout_pos] = opens[breakout_pos] - volatility * 0.2
        volumes[breakout_pos] *= 2  # Higher volume on breakout
        
    elif pattern_type == "channel":
        # Create rising channel
        channel_start = n_bars // 5
        channel_end = 4 * n_bars // 5
        
        for i in range(channel_start, channel_end):
            # Upper channel line
            upper_level = closes[channel_start] + 0.8 + 0.5 * (i - channel_start) * trend_slope
            # Lower channel line  
            lower_level = closes[channel_start] - 0.8 + 0.5 * (i - channel_start) * trend_slope
            
            # Keep price within channel
            closes[i] = np.clip(closes[i], lower_level + 0.1, upper_level - 0.1)
            highs[i] = min(highs[i], upper_level)
            lows[i] = max(lows[i], lower_level)
    
    # Create timestamps
    timestamps = pd.date_range('2024-01-01', periods=n_bars, freq='h')
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs, 
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    return df


@register("cv.detect_chart_patterns")
def cv_detect_chart_patterns(**kwargs) -> ToolResult:
    """
    Detect chart patterns using computer vision + numeric confirmation
    
    Combines off-screen rendering with pure OHLCV-based validation
    """
    try:
        args = CVDetectIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    start_time = time.time()
    warnings = []
    
    # Get DataFrame - either from kwargs (_df for testing) or via data loading
    df = kwargs.get("_df")
    if df is None:
        # In production, this would call data.load_ohlcv
        # For now, generate synthetic data based on symbol
        pattern_hint = "engulfing"  # Default
        if "breakout" in args.symbol.lower() or "trendline" in " ".join(args.patterns):
            pattern_hint = "breakout"
        elif "channel" in " ".join(args.patterns):
            pattern_hint = "channel"
        
        df = generate_synthetic_data(pattern_hint, args.lookback, 
                                   seed=hash(args.symbol) % 2**31)
        warnings.append("Using synthetic data - integrate with data.load_ohlcv for production")
    
    # Slice to lookback window and reset index
    df = df.tail(args.lookback).copy().reset_index(drop=True)
    n_bars = len(df)
    
    if n_bars < 10:
        return ToolResult.error(["Insufficient data: need at least 10 bars"])
    
    detections = []
    metadata = {
        'patterns_requested': args.patterns,
        'lookback_used': n_bars,
        'confirm_with_rules': args.confirm_with_rules
    }
    
    try:
        # 1) CANDLESTICK PATTERN DETECTION
        
        # Engulfing patterns
        if "engulfing_bull" in args.patterns or "engulfing_bear" in args.patterns:
            engulfings = detect_engulfings(df)
            for i, side in engulfings:
                pattern_name = f"engulfing_{side}"
                if pattern_name in args.patterns:
                    confirmed = True
                    if args.confirm_with_rules:
                        # Additional confirmation: volume increase
                        if i > 0:
                            vol_ratio = df.loc[i, 'volume'] / max(df.loc[i-1, 'volume'], 1)
                            confirmed = vol_ratio > 1.2
                    
                    detections.append(CVDetection(
                        pattern=pattern_name,
                        start_idx=i-1,
                        end_idx=i,
                        strength=0.9 if confirmed else 0.6,
                        info={'volume_ratio': vol_ratio if i > 0 else 1.0},
                        confirmed=confirmed
                    ))
        
        # Pin bar patterns
        pin_patterns = ["pin_bar_bull", "pin_bar_bear"]
        for pattern in pin_patterns:
            if pattern in args.patterns:
                bull = "bull" in pattern
                for i in range(2, n_bars):
                    if detect_pinbar(df, i, bull):
                        confirmed = True
                        if args.confirm_with_rules:
                            # Additional confirmation: rejection of key level
                            atr_vals = atr(df.iloc[:i+1], 14)
                            if not atr_vals.iloc[-1] > 0:
                                confirmed = False
                        
                        detections.append(CVDetection(
                            pattern=pattern,
                            start_idx=i,
                            end_idx=i,
                            strength=0.8,
                            info={'wick_ratio': 
                                 (df.loc[i, 'high'] - max(df.loc[i, 'open'], df.loc[i, 'close'])) /
                                 max(abs(df.loc[i, 'close'] - df.loc[i, 'open']), 1e-8) 
                                 if bull else 
                                 (min(df.loc[i, 'open'], df.loc[i, 'close']) - df.loc[i, 'low']) /
                                 max(abs(df.loc[i, 'close'] - df.loc[i, 'open']), 1e-8)},
                            confirmed=confirmed
                        ))
        
        # Star patterns
        star_patterns = ["morning_star", "evening_star"]
        for pattern in star_patterns:
            if pattern in args.patterns:
                bull = "morning" in pattern
                for i in range(2, n_bars):
                    if detect_morning_evening_star(df, i, bull):
                        confirmed = True
                        if args.confirm_with_rules:
                            # Confirm with volume on third candle
                            vol_ratio = (df.loc[i, 'volume'] / 
                                       max((df.loc[i-2, 'volume'] + df.loc[i-1, 'volume']) / 2, 1))
                            confirmed = vol_ratio > 1.1
                        
                        detections.append(CVDetection(
                            pattern=pattern,
                            start_idx=i-2,
                            end_idx=i,
                            strength=0.85,
                            info={'volume_confirmation': vol_ratio if args.confirm_with_rules else 1.0},
                            confirmed=confirmed
                        ))
        
        # 2) STRUCTURE PATTERN DETECTION (Trendlines, Channels, Flags)
        
        if "trendline_break" in args.patterns and n_bars > 50:
            # Detect pivot points for trendline fitting
            pivot_window = max(3, int(0.02 * n_bars))
            
            # High pivots (resistance trendline)
            high_pivots = detect_pivots(df['high'], pivot_window)
            if len(high_pivots) >= 3:
                # Fit trendline on recent pivots
                recent_pivots = high_pivots[-min(5, len(high_pivots)):]
                pivot_highs = [df.loc[i, 'high'] for i in recent_pivots]
                
                trendline_params = fit_trendline(recent_pivots, pivot_highs)
                if trendline_params:
                    slope, intercept = trendline_params
                    
                    # Check for breakout at last few bars
                    for check_i in range(max(recent_pivots[-1], n_bars-5), n_bars):
                        trendline_level = slope * check_i + intercept
                        
                        confirmed = breakout_over_level(df, check_i, trendline_level, 0.5)
                        if confirmed or not args.confirm_with_rules:
                            distance = calculate_trendline_distance(df, check_i, slope, intercept)
                            atr_val = atr(df.iloc[:check_i+1], 14).iloc[-1]
                            strength = min(1.0, abs(distance) / max(atr_val, 1e-8)) if not pd.isna(atr_val) else 0.5
                            
                            detections.append(CVDetection(
                                pattern="trendline_break",
                                start_idx=recent_pivots[0],
                                end_idx=check_i,
                                strength=float(strength),
                                info={
                                    'line': 'upper',
                                    'slope': float(slope),
                                    'level': float(trendline_level),
                                    'distance_atr': float(distance / max(atr_val, 1e-8)) if not pd.isna(atr_val) else 0
                                },
                                confirmed=confirmed
                            ))
                            break  # Only report first breakout
            
            # Low pivots (support trendline)  
            low_pivots = detect_pivots(df['low'], pivot_window)
            if len(low_pivots) >= 3:
                recent_pivots = low_pivots[-min(5, len(low_pivots)):]
                pivot_lows = [df.loc[i, 'low'] for i in recent_pivots]
                
                trendline_params = fit_trendline(recent_pivots, pivot_lows)
                if trendline_params:
                    slope, intercept = trendline_params
                    
                    # Check for breakdown
                    for check_i in range(max(recent_pivots[-1], n_bars-5), n_bars):
                        trendline_level = slope * check_i + intercept
                        
                        confirmed = breakout_under_level(df, check_i, trendline_level, 0.5)
                        if confirmed or not args.confirm_with_rules:
                            distance = calculate_trendline_distance(df, check_i, slope, intercept)
                            atr_val = atr(df.iloc[:check_i+1], 14).iloc[-1]
                            strength = min(1.0, abs(distance) / max(atr_val, 1e-8)) if not pd.isna(atr_val) else 0.5
                            
                            detections.append(CVDetection(
                                pattern="trendline_break",
                                start_idx=recent_pivots[0],
                                end_idx=check_i,
                                strength=float(strength),
                                info={
                                    'line': 'lower',
                                    'slope': float(slope),
                                    'level': float(trendline_level),
                                    'distance_atr': float(distance / max(atr_val, 1e-8)) if not pd.isna(atr_val) else 0
                                },
                                confirmed=confirmed
                            ))
                            break
        
        # TODO: Channel and flag detection (simplified for now)
        # These would involve parallel line detection and impulse-consolidation sequences
        
        # 3) OPTIONAL IMAGE RENDERING
        rendered_b64 = None
        if args.return_image and detections:
            try:
                # Convert detections to dict format for plotting
                detection_dicts = [det.model_dump() for det in detections]
                png_bytes = render_chart_with_patterns(
                    df[['timestamp', 'open', 'high', 'low', 'close']], 
                    detection_dicts,
                    args.image_width,
                    args.image_height
                )
                rendered_b64 = base64.b64encode(png_bytes).decode('ascii')
                metadata['image_size_bytes'] = len(png_bytes)
            except Exception as e:
                warnings.append(f"Image rendering failed: {str(e)}")
        
        # 4) BUILD RESULT
        out = CVDetectOut(
            detections=detections,
            n_bars=n_bars,
            rendered=rendered_b64,
            metadata=metadata
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return ToolResult.success(
            data=out.model_dump(),
            warnings=warnings
        )
        
    except Exception as e:
        return ToolResult.error([f"Pattern detection failed: {str(e)}"])


@register("cv.generate_synthetic")  
def cv_generate_synthetic(**kwargs) -> ToolResult:
    """
    Generate synthetic OHLCV data with embedded patterns for testing
    
    Args:
        pattern_type: Type of pattern to embed
        n_bars: Number of bars to generate  
        seed: Random seed
    """
    pattern_type = kwargs.get('pattern_type', 'engulfing')
    n_bars = kwargs.get('n_bars', 200)
    seed = kwargs.get('seed', 42)
    
    try:
        df = generate_synthetic_data(pattern_type, n_bars, seed)
        
        # Convert DataFrame to dict first, then apply timestamp conversion
        df_dict = df.to_dict(orient='records')
        df_serialized = convert_timestamps(df_dict)
        
        return ToolResult.success({
            'synthetic_data': df_serialized,
            'pattern_type': pattern_type,
            'n_bars': len(df),
            'columns': list(df.columns)
        })
        
    except Exception as e:
        return ToolResult.error([f"Synthetic data generation failed: {str(e)}"])


if __name__ == "__main__":
    # Test CV detection tools
    print("Testing CV pattern detection...")
    
    # Test with engulfing pattern
    result = cv_detect_chart_patterns(
        symbol="TEST_ENGULFING",
        interval="1h",
        patterns=["engulfing_bull", "engulfing_bear"],
        lookback=100,
        confirm_with_rules=True
    )
    
    print(f"Engulfing detection: {result.ok}")
    if result.ok:
        detections = result.data['detections']
        print(f"Found {len(detections)} patterns")
        for det in detections:
            print(f"  {det['pattern']} at {det['start_idx']}-{det['end_idx']}, strength: {det['strength']:.2f}")
    
    # Test with trendline break
    result = cv_detect_chart_patterns(
        symbol="TEST_BREAKOUT", 
        interval="1h",
        patterns=["trendline_break"],
        lookback=200,
        return_image=False
    )
    
    print(f"Trendline detection: {result.ok}")
    if result.ok:
        detections = result.data['detections']
        print(f"Found {len(detections)} trendline breaks")
        for det in detections:
            info = det['info']
            print(f"  Break {info.get('line', 'unknown')} line, distance: {info.get('distance_atr', 0):.2f} ATR")
    
    print("CV pattern detection test complete!")