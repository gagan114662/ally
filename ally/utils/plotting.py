"""
Off-screen plotting utilities for Ally CV tools
Renders candlestick charts without GUI dependency
"""

import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np
import io
from typing import List, Dict, Any, Optional


def render_candles(df: pd.DataFrame, width: int = 900, height: int = 500, 
                  overlays: Optional[List[Dict[str, Any]]] = None) -> bytes:
    """
    Render OHLC candlestick chart to PNG bytes (off-screen)
    
    Args:
        df: DataFrame with columns [timestamp, open, high, low, close]
        width: Image width in pixels
        height: Image height in pixels  
        overlays: List of drawing overlays
                 [{"type":"line","x":[i0,i1],"y":[y0,y1],"color":"red"},
                  {"type":"box","i":i,"color":"green"}]
                  
    Returns:
        PNG image as bytes
    """
    if overlays is None:
        overlays = []
    
    # Set up figure with specified dimensions
    dpi = 100
    fig_width = width / dpi
    fig_height = height / dpi
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    fig.patch.set_facecolor('white')
    
    # Extract OHLC data
    opens = df['open'].values
    highs = df['high'].values  
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)
    
    if n_bars == 0:
        # Empty chart
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=16)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    
    # X-axis indices
    indices = np.arange(n_bars)
    
    # Draw candlesticks
    for i in range(n_bars):
        x = indices[i]
        open_price = opens[i]
        high_price = highs[i] 
        low_price = lows[i]
        close_price = closes[i]
        
        # Determine candle color
        if close_price >= open_price:
            # Green/white candle (bullish)
            body_color = 'lightgreen'
            edge_color = 'darkgreen'
        else:
            # Red/black candle (bearish)
            body_color = 'lightcoral'
            edge_color = 'darkred'
        
        # Draw high-low line (wick)
        ax.plot([x, x], [low_price, high_price], color='black', linewidth=1)
        
        # Draw open-close rectangle (body)
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        # Candle body width
        body_width = 0.6
        
        rect = Rectangle((x - body_width/2, body_bottom), body_width, body_height,
                        facecolor=body_color, edgecolor=edge_color, linewidth=1)
        ax.add_patch(rect)
    
    # Draw overlays
    for overlay in overlays:
        overlay_type = overlay.get('type', '')
        
        if overlay_type == 'line':
            x_coords = overlay.get('x', [])
            y_coords = overlay.get('y', [])
            color = overlay.get('color', 'blue')
            linewidth = overlay.get('linewidth', 2)
            linestyle = overlay.get('linestyle', '-')
            
            if len(x_coords) >= 2 and len(y_coords) >= 2:
                ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, 
                       linestyle=linestyle, alpha=0.7)
        
        elif overlay_type == 'box':
            i = overlay.get('i', 0)
            color = overlay.get('color', 'blue')
            
            if 0 <= i < n_bars:
                # Highlight candle with colored box
                high_price = highs[i]
                low_price = lows[i]
                rect = Rectangle((i - 0.4, low_price), 0.8, high_price - low_price,
                               facecolor='none', edgecolor=color, linewidth=3)
                ax.add_patch(rect)
        
        elif overlay_type == 'marker':
            x = overlay.get('x', 0)
            y = overlay.get('y', 0)
            color = overlay.get('color', 'red')
            marker = overlay.get('marker', 'o')
            size = overlay.get('size', 50)
            
            ax.scatter([x], [y], color=color, marker=marker, s=size, alpha=0.8, zorder=5)
        
        elif overlay_type == 'hline':
            y = overlay.get('y', 0)
            color = overlay.get('color', 'gray')
            linestyle = overlay.get('linestyle', '--')
            linewidth = overlay.get('linewidth', 1)
            
            ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.7)
    
    # Formatting
    ax.set_xlim(-0.5, n_bars - 0.5)
    
    # Y-axis: add some padding
    price_range = highs.max() - lows.min()
    padding = price_range * 0.1
    ax.set_ylim(lows.min() - padding, highs.max() + padding)
    
    # Labels and grid
    ax.set_xlabel('Bar Index', fontsize=10)
    ax.set_ylabel('Price', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Title with symbol and timeframe info if available
    symbol = df.get('symbol', ['Unknown'])[0] if 'symbol' in df.columns else 'Unknown'
    ax.set_title(f'{symbol} Candlestick Chart', fontsize=12, fontweight='bold')
    
    # Tight layout
    plt.tight_layout()
    
    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close(fig)  # Important: close figure to free memory
    
    buf.seek(0)
    return buf.read()


def create_pattern_overlays(detections: List[Dict[str, Any]], 
                          df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Create overlay annotations for detected patterns
    
    Args:
        detections: List of pattern detections
        df: OHLCV DataFrame
        
    Returns:
        List of overlay dictionaries for render_candles
    """
    overlays = []
    
    for detection in detections:
        pattern = detection.get('pattern', '')
        start_idx = detection.get('start_idx', 0)
        end_idx = detection.get('end_idx', 0)
        strength = detection.get('strength', 0.5)
        info = detection.get('info', {})
        confirmed = detection.get('confirmed', False)
        
        # Color based on confirmation and pattern type
        if confirmed:
            if 'bull' in pattern:
                color = 'green'
            elif 'bear' in pattern:
                color = 'red' 
            else:
                color = 'blue'
        else:
            color = 'gray'
        
        # Pattern-specific overlays
        if 'engulfing' in pattern or 'pin_bar' in pattern:
            # Highlight the specific candle(s)
            for i in range(start_idx, end_idx + 1):
                overlays.append({
                    'type': 'box',
                    'i': i,
                    'color': color
                })
        
        elif 'star' in pattern:
            # Highlight all three candles of the star pattern
            for i in range(start_idx, end_idx + 1):
                overlays.append({
                    'type': 'box', 
                    'i': i,
                    'color': color
                })
        
        elif pattern == 'trendline_break':
            # Draw the trendline
            line_type = info.get('line', 'upper')
            slope = info.get('slope', 0)
            level = info.get('level', 0)
            
            # Draw trendline from start to end
            y_start = slope * start_idx + (level - slope * end_idx)
            y_end = level
            
            overlays.append({
                'type': 'line',
                'x': [start_idx, end_idx],
                'y': [y_start, y_end],
                'color': color,
                'linewidth': 2,
                'linestyle': '--'
            })
            
            # Mark the breakout point
            if end_idx < len(df):
                breakout_price = df.loc[end_idx, 'close']
                overlays.append({
                    'type': 'marker',
                    'x': end_idx,
                    'y': breakout_price,
                    'color': color,
                    'marker': '^' if 'upper' in line_type else 'v',
                    'size': 80
                })
        
        elif 'channel' in pattern:
            # Draw channel lines (if info contains line parameters)
            upper_slope = info.get('upper_slope')
            lower_slope = info.get('lower_slope')
            
            if upper_slope is not None and lower_slope is not None:
                # Draw upper and lower channel lines
                x_range = [start_idx, end_idx]
                
                upper_y = [upper_slope * x + info.get('upper_intercept', 0) for x in x_range]
                lower_y = [lower_slope * x + info.get('lower_intercept', 0) for x in x_range]
                
                overlays.extend([
                    {'type': 'line', 'x': x_range, 'y': upper_y, 'color': color, 'linestyle': '--'},
                    {'type': 'line', 'x': x_range, 'y': lower_y, 'color': color, 'linestyle': '--'}
                ])
        
        elif 'flag' in pattern:
            # Draw flag pattern (short channel after impulse)
            flag_slope = info.get('flag_slope', 0)
            flag_start = info.get('flag_start', start_idx)
            
            x_range = [flag_start, end_idx]
            flag_high = [info.get('flag_high_start', 0) + flag_slope * (x - flag_start) for x in x_range]
            flag_low = [info.get('flag_low_start', 0) + flag_slope * (x - flag_start) for x in x_range]
            
            overlays.extend([
                {'type': 'line', 'x': x_range, 'y': flag_high, 'color': color, 'linestyle': ':'},
                {'type': 'line', 'x': x_range, 'y': flag_low, 'color': color, 'linestyle': ':'}
            ])
    
    return overlays


def render_chart_with_patterns(df: pd.DataFrame, detections: List[Dict[str, Any]], 
                              width: int = 900, height: int = 500) -> bytes:
    """
    Convenience function to render chart with pattern annotations
    
    Args:
        df: OHLCV DataFrame  
        detections: Pattern detections from cv.detect_chart_patterns
        width: Image width
        height: Image height
        
    Returns:
        PNG bytes with annotated patterns
    """
    overlays = create_pattern_overlays(detections, df)
    return render_candles(df, width, height, overlays)