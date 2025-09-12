"""
Technical analysis numeric confirmation rules
Pure OHLCV-based pattern validation without vision dependency
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high = df['high']
    low = df['low'] 
    close = df['close']
    
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def is_engulfing(df: pd.DataFrame, i: int, bull: bool, body_ratio: float = 1.05, 
                min_body_frac: float = 0.2) -> bool:
    """
    Check if candle i engulfs candle i-1
    
    Args:
        df: OHLCV DataFrame
        i: Current candle index
        bull: True for bullish engulfing, False for bearish
        body_ratio: How much bigger the engulfing body must be
        min_body_frac: Minimum body size relative to range
    """
    if i < 1 or i >= len(df):
        return False
    
    # Current candle (engulfing)
    o1, h1, l1, c1 = df.loc[i, ['open', 'high', 'low', 'close']]
    # Previous candle (engulfed) 
    o0, h0, l0, c0 = df.loc[i-1, ['open', 'high', 'low', 'close']]
    
    # Body sizes
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    range0 = h0 - l0
    range1 = h1 - l1
    
    # Minimum body size filter
    if body0 < min_body_frac * range0 or body1 < min_body_frac * range1:
        return False
    
    # Body ratio check
    if body1 < body_ratio * body0:
        return False
    
    if bull:
        # Bullish engulfing: prev red, curr green, curr body engulfs prev
        return (c0 < o0) and (c1 > o1) and (o1 < c0) and (c1 > o0)
    else:
        # Bearish engulfing: prev green, curr red, curr body engulfs prev  
        return (c0 > o0) and (c1 < o1) and (o1 > c0) and (c1 < o0)


def detect_engulfings(df: pd.DataFrame) -> List[Tuple[int, str]]:
    """Return list of (index, 'bull'|'bear') for all engulfing patterns"""
    engulfings = []
    
    for i in range(1, len(df)):
        if is_engulfing(df, i, bull=True):
            engulfings.append((i, 'bull'))
        elif is_engulfing(df, i, bull=False):
            engulfings.append((i, 'bear'))
    
    return engulfings


def detect_pinbar(df: pd.DataFrame, i: int, bull: bool, wick_to_body_min: float = 2.0) -> bool:
    """
    Detect pin bar (doji with long wick)
    
    Args:
        df: OHLCV DataFrame
        i: Candle index
        bull: True for bullish pin bar (long lower wick)
        wick_to_body_min: Minimum wick to body ratio
    """
    if i >= len(df):
        return False
    
    o, h, l, c = df.loc[i, ['open', 'high', 'low', 'close']]
    
    body = abs(c - o)
    total_range = h - l
    
    # Avoid division by zero
    if body < 1e-8 or total_range < 1e-8:
        return False
    
    if bull:
        # Bullish pin bar: long lower wick, small body near top
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        
        return (lower_wick > wick_to_body_min * body and 
                lower_wick > 0.6 * total_range and
                upper_wick < 0.3 * total_range)
    else:
        # Bearish pin bar: long upper wick, small body near bottom
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        
        return (upper_wick > wick_to_body_min * body and
                upper_wick > 0.6 * total_range and
                lower_wick < 0.3 * total_range)


def detect_morning_evening_star(df: pd.DataFrame, i: int, bull: bool) -> bool:
    """
    Detect 3-candle morning/evening star pattern
    
    Args:
        df: OHLCV DataFrame  
        i: Index of third candle
        bull: True for morning star, False for evening star
    """
    if i < 2 or i >= len(df):
        return False
    
    # Three candles: i-2, i-1, i
    o0, h0, l0, c0 = df.loc[i-2, ['open', 'high', 'low', 'close']]
    o1, h1, l1, c1 = df.loc[i-1, ['open', 'high', 'low', 'close']]  
    o2, h2, l2, c2 = df.loc[i, ['open', 'high', 'low', 'close']]
    
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    
    if bull:
        # Morning star: red, small, green
        # First candle: strong red
        if not (c0 < o0 and body0 > 0.6 * (h0 - l0)):
            return False
        
        # Middle candle: small body, gaps down
        if not (body1 < 0.3 * body0 and max(o1, c1) < min(o0, c0)):
            return False
        
        # Third candle: strong green, closes well into first body
        return (c2 > o2 and body2 > 0.6 * (h2 - l2) and 
                c2 > (o0 + c0) / 2)
    else:
        # Evening star: green, small, red
        # First candle: strong green
        if not (c0 > o0 and body0 > 0.6 * (h0 - l0)):
            return False
        
        # Middle candle: small body, gaps up
        if not (body1 < 0.3 * body0 and min(o1, c1) > max(o0, c0)):
            return False
        
        # Third candle: strong red, closes well into first body
        return (c2 < o2 and body2 > 0.6 * (h2 - l2) and 
                c2 < (o0 + c0) / 2)


def breakout_over_level(df: pd.DataFrame, i: int, level: float, k_atr: float = 0.5) -> bool:
    """
    Check if close[i] exceeds level by k*ATR confirmation
    
    Args:
        df: OHLCV DataFrame
        i: Bar index
        level: Price level to break
        k_atr: ATR multiplier for confirmation
    """
    if i < 14 or i >= len(df):  # Need 14 bars for ATR
        return False
    
    close = df.loc[i, 'close']
    atr_val = atr(df.iloc[:i+1], 14).iloc[-1]
    
    if pd.isna(atr_val) or atr_val < 1e-8:
        return False
    
    return close > level + k_atr * atr_val


def breakout_under_level(df: pd.DataFrame, i: int, level: float, k_atr: float = 0.5) -> bool:
    """Check if close[i] breaks below level by k*ATR confirmation"""
    if i < 14 or i >= len(df):
        return False
    
    close = df.loc[i, 'close']
    atr_val = atr(df.iloc[:i+1], 14).iloc[-1]
    
    if pd.isna(atr_val) or atr_val < 1e-8:
        return False
    
    return close < level - k_atr * atr_val


def detect_pivots(series: pd.Series, window: int = 5) -> List[int]:
    """
    Find pivot points (local extrema) in a price series
    
    Args:
        series: Price series (highs or lows)
        window: Window size for pivot detection
        
    Returns:
        List of pivot indices
    """
    pivots = []
    n = len(series)
    
    for i in range(window, n - window):
        left_max = series.iloc[i-window:i].max()
        right_max = series.iloc[i+1:i+window+1].max()
        
        # Peak: higher than both sides
        if series.iloc[i] > left_max and series.iloc[i] > right_max:
            pivots.append(i)
        
        # Valley: lower than both sides  
        left_min = series.iloc[i-window:i].min()
        right_min = series.iloc[i+1:i+window+1].min()
        
        if series.iloc[i] < left_min and series.iloc[i] < right_min:
            pivots.append(i)
    
    return pivots


def fit_trendline(indices: List[int], values: List[float]) -> Optional[Tuple[float, float]]:
    """
    Fit a trendline through pivot points
    
    Args:
        indices: X coordinates (bar indices)
        values: Y coordinates (prices)
        
    Returns:
        (slope, intercept) or None if insufficient data
    """
    if len(indices) < 2:
        return None
    
    x = np.array(indices, dtype=float)
    y = np.array(values, dtype=float)
    
    try:
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)
    except:
        return None


def calculate_trendline_distance(df: pd.DataFrame, i: int, slope: float, intercept: float) -> float:
    """
    Calculate distance from close price to trendline at bar i
    Positive = above trendline, Negative = below trendline
    """
    if i >= len(df):
        return 0.0
    
    close = df.loc[i, 'close']
    trendline_value = slope * i + intercept
    
    return float(close - trendline_value)