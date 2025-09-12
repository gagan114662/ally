#!/usr/bin/env python3
"""
Simple test to debug CV tool serialization issue
"""
import sys
from pathlib import Path

# Add Ally to path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "Ally"))

from Ally.tools.cv import cv_generate_synthetic

# Test synthetic data generation
result = cv_generate_synthetic(pattern_type="engulfing", n_bars=50, seed=42)
print(f"Result OK: {result.ok}")
if result.ok:
    print(f"Data keys: {result.data.keys()}")
    print(f"Sample data: {result.data['synthetic_data'][:2]}")
else:
    print(f"Errors: {result.errors}")