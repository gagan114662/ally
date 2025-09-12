#!/usr/bin/env python3
"""
Test CV pattern detection
"""
import sys
import pandas as pd
from pathlib import Path

# Add Ally to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "Ally"))

from Ally.tools.cv import cv_detect_chart_patterns, cv_generate_synthetic

# Generate test data
gen_result = cv_generate_synthetic(pattern_type="engulfing", n_bars=50, seed=42)
if not gen_result.ok:
    print(f"Generation failed: {gen_result.errors}")
    exit(1)

# Convert back to DataFrame for detection
df_data = gen_result.data['synthetic_data']
df = pd.DataFrame(df_data)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Test pattern detection
result = cv_detect_chart_patterns(
    symbol="TEST",
    interval="1h",
    patterns=["engulfing_bull", "engulfing_bear"],
    lookback=50,
    _df=df
)

print(f"Detection OK: {result.ok}")
if result.ok:
    print(f"Data keys: {result.data.keys()}")
    detections = result.data["detections"]
    print(f"Found {len(detections)} detections")
    for d in detections:
        print(f"  {d['pattern']} at {d['start_idx']}-{d['end_idx']}")
else:
    print(f"Errors: {result.errors}")