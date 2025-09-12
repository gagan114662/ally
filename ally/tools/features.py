"""
Features tools for Ally - build technical indicators and validate for leakage
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import warnings as python_warnings
python_warnings.filterwarnings('ignore')

from ..tools import register
from ..schemas.base import ToolResult
from ..schemas.features import BuildFeaturesIn, FeatureData, ValidateLeakageIn, LeakageValidationResult, FeatureConfig
from ..utils.serialization import convert_timestamps


# Feature calculation functions
def calculate_rsi(prices: np.array, period: int = 14) -> np.array:
    """Calculate RSI (Relative Strength Index)"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = np.nan
    rsi[period] = 100. - 100. / (1. + rs)
    
    for i in range(period + 1, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi


def calculate_ema(prices: np.array, period: int) -> np.array:
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    alpha = 2.0 / (period + 1)
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema


def calculate_sma(prices: np.array, period: int) -> np.array:
    """Calculate Simple Moving Average"""
    sma = np.full_like(prices, np.nan)
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])
    return sma


def calculate_atr(high: np.array, low: np.array, close: np.array, period: int = 14) -> np.array:
    """Calculate Average True Range"""
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    # Set first value for rolled arrays
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    
    true_range = np.maximum.reduce([tr1, tr2, tr3])
    
    # ATR is EMA of True Range
    return calculate_ema(true_range, period)


def calculate_bbands(prices: np.array, period: int = 20, std_dev: float = 2.0) -> Dict[str, np.array]:
    """Calculate Bollinger Bands"""
    sma = calculate_sma(prices, period)
    
    # Calculate rolling standard deviation
    std = np.full_like(prices, np.nan)
    for i in range(period - 1, len(prices)):
        std[i] = np.std(prices[i - period + 1:i + 1])
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return {
        'bb_upper': upper,
        'bb_middle': sma,
        'bb_lower': lower,
        'bb_width': upper - lower
    }


def calculate_returns(prices: np.array, periods: int = 1) -> np.array:
    """Calculate returns with specified period"""
    returns = np.full_like(prices, np.nan)
    for i in range(periods, len(prices)):
        returns[i] = (prices[i] / prices[i - periods]) - 1
    return returns


def calculate_zscore(values: np.array, period: int = 20) -> np.array:
    """Calculate rolling Z-score"""
    zscore = np.full_like(values, np.nan)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std != 0:
            zscore[i] = (values[i] - mean) / std
        else:
            zscore[i] = 0
    return zscore


@register("features.build")
def features_build(**kwargs) -> ToolResult:
    """
    Build technical features from OHLCV data
    
    Supports RSI, ATR, EMA, SMA, Bollinger Bands, returns, and Z-scores
    All features are calculated to prevent look-ahead bias
    """
    try:
        inputs = BuildFeaturesIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    warnings = []
    start_time = time.time()
    
    try:
        # First, get OHLCV data for the symbol
        from .data import data_load_ohlcv
        
        data_result = data_load_ohlcv(
            symbols=[inputs.symbol],
            interval=inputs.interval,
            start="2020-01-01",  # Use wide range for sufficient lookback
            end="2024-12-31",
            source=inputs.data_source or "mock"
        )
        
        if not data_result.ok:
            return ToolResult.error([f"Failed to load data: {data_result.errors}"])
        
        # Extract OHLCV data
        panel = data_result.data['panel']
        if inputs.symbol not in panel['aligned_data']:
            return ToolResult.error([f"No data found for symbol: {inputs.symbol}"])
        
        ohlcv_records = panel['aligned_data'][inputs.symbol]
        
        if len(ohlcv_records) < inputs.lookback:
            warnings.append(f"Limited data: {len(ohlcv_records)} rows, requested {inputs.lookback} lookback")
        
        # Use last N records based on lookback
        records = ohlcv_records[-inputs.lookback:] if len(ohlcv_records) > inputs.lookback else ohlcv_records
        
        # Convert to arrays for calculations
        timestamps = [r['timestamp'] for r in records]
        opens = np.array([float(r['open']) for r in records])
        highs = np.array([float(r['high']) for r in records])
        lows = np.array([float(r['low']) for r in records])
        closes = np.array([float(r['close']) for r in records])
        volumes = np.array([float(r['volume']) for r in records])
        
        # Build requested features
        features = {}
        feature_configs = []
        
        for feature_name in inputs.feature_set:
            try:
                if feature_name == "rsi":
                    features["rsi"] = calculate_rsi(closes).tolist()
                    feature_configs.append(FeatureConfig(
                        name="rsi", type="technical", params={"period": 14}, lookback_required=14
                    ))
                    
                elif feature_name == "rsi_21":
                    features["rsi_21"] = calculate_rsi(closes, 21).tolist()
                    feature_configs.append(FeatureConfig(
                        name="rsi_21", type="technical", params={"period": 21}, lookback_required=21
                    ))
                    
                elif feature_name == "atr":
                    features["atr"] = calculate_atr(highs, lows, closes).tolist()
                    feature_configs.append(FeatureConfig(
                        name="atr", type="technical", params={"period": 14}, lookback_required=14
                    ))
                    
                elif feature_name.startswith("ema_"):
                    period = int(feature_name.split("_")[1])
                    features[feature_name] = calculate_ema(closes, period).tolist()
                    feature_configs.append(FeatureConfig(
                        name=feature_name, type="technical", params={"period": period}, lookback_required=period
                    ))
                    
                elif feature_name.startswith("sma_"):
                    period = int(feature_name.split("_")[1])
                    features[feature_name] = calculate_sma(closes, period).tolist()
                    feature_configs.append(FeatureConfig(
                        name=feature_name, type="technical", params={"period": period}, lookback_required=period
                    ))
                    
                elif feature_name == "bbands":
                    bb_data = calculate_bbands(closes)
                    for bb_key, bb_values in bb_data.items():
                        features[bb_key] = bb_values.tolist()
                        feature_configs.append(FeatureConfig(
                            name=bb_key, type="technical", params={"period": 20, "std_dev": 2.0}, lookback_required=20
                        ))
                        
                elif feature_name == "returns_1d":
                    features["returns_1d"] = calculate_returns(closes, 1).tolist()
                    feature_configs.append(FeatureConfig(
                        name="returns_1d", type="price", params={"periods": 1}, lookback_required=1
                    ))
                    
                elif feature_name == "returns_5d":
                    features["returns_5d"] = calculate_returns(closes, 5).tolist()
                    feature_configs.append(FeatureConfig(
                        name="returns_5d", type="price", params={"periods": 5}, lookback_required=5
                    ))
                    
                elif feature_name == "zscore_close":
                    features["zscore_close"] = calculate_zscore(closes, 20).tolist()
                    feature_configs.append(FeatureConfig(
                        name="zscore_close", type="derived", params={"period": 20}, lookback_required=20
                    ))
                    
                elif feature_name == "volume_sma":
                    features["volume_sma"] = calculate_sma(volumes, 20).tolist()
                    feature_configs.append(FeatureConfig(
                        name="volume_sma", type="volume", params={"period": 20}, lookback_required=20
                    ))
                    
                else:
                    warnings.append(f"Unknown feature: {feature_name}")
                    
            except Exception as e:
                warnings.append(f"Failed to calculate {feature_name}: {e}")
        
        if not features:
            return ToolResult.error(["No features calculated successfully"])
        
        # Convert timestamps to ISO strings
        timestamps_iso = [convert_timestamps(ts) for ts in timestamps]
        
        # Create feature data object
        feature_data = FeatureData(
            symbol=inputs.symbol,
            interval=inputs.interval,
            features=features,
            timestamps=timestamps_iso,
            feature_configs=feature_configs,
            total_rows=len(records),
            lookback_used=len(records),
            metadata={
                "calculation_time": time.time() - start_time,
                "requested_features": inputs.feature_set,
                "calculated_features": list(features.keys())
            }
        )
        
        return ToolResult.success(
            data={
                'feature_data': feature_data.model_dump(),
                'summary': {
                    'features_calculated': len(features),
                    'total_rows': len(records),
                    'date_range': f"{timestamps[0]} to {timestamps[-1]}" if timestamps else "No data"
                }
            },
            warnings=warnings
        )
        
    except Exception as e:
        return ToolResult.error([f"Feature calculation failed: {e}"])


@register("features.validate_leakage")
def features_validate_leakage(**kwargs) -> ToolResult:
    """
    Validate features for look-ahead bias (data leakage)
    
    Checks correlations between features and future price movements
    """
    try:
        inputs = ValidateLeakageIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        violations = []
        correlations = {}
        
        # Get future price changes for comparison
        if 'close' not in inputs.price_data:
            return ToolResult.error(["Price data must include 'close' prices"])
        
        close_prices = np.array(inputs.price_data['close'])
        future_returns = np.roll(calculate_returns(close_prices, 1), -1)  # Next period return
        
        # Check each feature for leakage
        for feature_name, feature_values in inputs.features_data.items():
            feature_array = np.array(feature_values)
            
            # Remove NaN values for correlation calculation
            valid_indices = ~(np.isnan(feature_array) | np.isnan(future_returns))
            
            if np.sum(valid_indices) < 10:  # Need at least 10 points for meaningful correlation
                continue
                
            # Calculate correlation with future returns
            corr = np.corrcoef(
                feature_array[valid_indices], 
                future_returns[valid_indices]
            )[0, 1]
            
            if np.isnan(corr):
                corr = 0.0
                
            correlations[feature_name] = abs(corr)
            
            # Check for violations
            if abs(corr) > inputs.max_correlation_threshold:
                violations.append({
                    'feature': feature_name,
                    'correlation': corr,
                    'threshold': inputs.max_correlation_threshold,
                    'severity': 'high' if abs(corr) > 0.98 else 'medium'
                })
        
        # Generate recommendations
        recommendations = []
        if violations:
            recommendations.append("Review feature calculation for look-ahead bias")
            recommendations.append("Consider using lagged features instead of current period")
            recommendations.append("Validate feature engineering logic for temporal consistency")
        else:
            recommendations.append("No significant leakage detected")
            recommendations.append("Features appear temporally consistent")
        
        # Create validation result
        result = LeakageValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            correlations=correlations,
            summary={
                'features_tested': len(inputs.features_data),
                'violations_found': len(violations),
                'max_correlation': max(correlations.values()) if correlations else 0.0,
                'threshold_used': inputs.max_correlation_threshold
            },
            recommendations=recommendations
        )
        
        return ToolResult.success(
            data={
                'validation_result': result.model_dump(),
                'summary': {
                    'is_valid': result.is_valid,
                    'violations': len(violations),
                    'features_tested': len(inputs.features_data)
                }
            }
        )
        
    except Exception as e:
        return ToolResult.error([f"Leakage validation failed: {e}"])


if __name__ == "__main__":
    # Test feature building
    result = features_build(
        symbol="BTCUSDT",
        interval="1h",
        feature_set=["rsi", "ema_20", "atr", "returns_1d", "zscore_close"],
        lookback=1000
    )
    
    print(f"Features test: {result.ok}")
    if result.ok:
        summary = result.data['summary']
        print(f"Features calculated: {summary['features_calculated']}")
        print(f"Total rows: {summary['total_rows']}")
        print(f"Date range: {summary['date_range']}")
    else:
        print(f"Errors: {result.errors}")
    
    # Test leakage validation with mock data
    if result.ok:
        features_data = result.data['feature_data']['features']
        price_data = {'close': [100 + i * 0.1 for i in range(len(features_data['rsi']))]}
        
        leakage_result = features_validate_leakage(
            features_data=features_data,
            price_data=price_data,
            max_correlation_threshold=0.95
        )
        
        print(f"Leakage validation: {leakage_result.ok}")
        if leakage_result.ok:
            validation = leakage_result.data['validation_result']
            print(f"Validation passed: {validation['is_valid']}")
            print(f"Violations: {validation['summary']['violations_found']}")