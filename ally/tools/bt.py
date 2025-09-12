"""
Backtest tools for Ally - run backtests with metric normalization
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta

from ..tools import register
from ..schemas.base import ToolResult
from ..schemas.bt import BTRunIn, BTMetrics, BTResult
from ..utils.serialization import convert_timestamps


# Metric normalization mapping
METRIC_MAP = {
    "return_annualized": "annual_return",
    "cagr": "annual_return", 
    "sharpe": "sharpe_ratio",
    "sortino": "sortino_ratio",
    "max_dd": "max_drawdown",
    "max_drawdown_pct": "max_drawdown",
    "win_rate_pct": "win_rate",
    "avg_trade_return": "avg_trade_pnl",
    "total_return": "annual_return"  # Will be annualized
}


def normalize_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize metric keys to standard format
    
    Args:
        metrics_dict: Raw metrics dictionary
        
    Returns:
        Normalized metrics dictionary
    """
    normalized = {}
    warnings = []
    
    for key, value in metrics_dict.items():
        # Convert key to standard form
        standard_key = METRIC_MAP.get(key, key)
        normalized[standard_key] = value
        
        if key != standard_key:
            warnings.append(f"Mapped metric '{key}' to '{standard_key}'")
    
    # Ensure required metrics exist with defaults
    required_metrics = {
        "annual_return": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "profit_factor": 1.0,
        "avg_trade_pnl": 0.0,
        "total_trades": 0
    }
    
    for metric, default_value in required_metrics.items():
        if metric not in normalized:
            normalized[metric] = default_value
            warnings.append(f"Added missing metric '{metric}' with default value {default_value}")
    
    return normalized, warnings


def generate_mock_backtest_data(strategy_id: str, symbols: List[str], 
                               start: str, end: str) -> Dict[str, Any]:
    """Generate mock backtest data for testing"""
    
    # Generate mock equity curve
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate realistic equity curve with some volatility
    np.random.seed(hash(strategy_id) % 2**31)  # Deterministic but unique per strategy
    returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns ~20% annual vol
    equity_values = (1 + returns).cumprod() * 100000  # Start with $100k
    
    equity_curve = []
    for i, (date, value) in enumerate(zip(dates, equity_values)):
        equity_curve.append({
            "date": convert_timestamps(date),
            "equity": float(value),
            "drawdown": float(max(0, (max(equity_values[:i+1]) - value) / max(equity_values[:i+1])))
        })
    
    # Generate mock trades
    num_trades = min(50, len(dates) // 5)  # ~1 trade per week
    trade_dates = np.random.choice(len(dates), num_trades, replace=False)
    trade_dates.sort()
    
    trades = []
    for i, trade_idx in enumerate(trade_dates):
        symbol = np.random.choice(symbols)
        entry_date = dates[trade_idx]
        exit_date = dates[min(trade_idx + np.random.randint(1, 10), len(dates) - 1)]
        
        # Mock trade PnL
        trade_return = np.random.normal(0.002, 0.05)  # ~0.2% average with 5% vol
        
        trades.append({
            "trade_id": i + 1,
            "symbol": symbol,
            "entry_date": convert_timestamps(entry_date),
            "exit_date": convert_timestamps(exit_date),
            "entry_price": float(100 + np.random.normal(0, 10)),
            "exit_price": float(100 + np.random.normal(trade_return * 100, 10)),
            "quantity": float(np.random.randint(10, 100)),
            "pnl": float(trade_return * 10000),  # Mock P&L
            "return": float(trade_return)
        })
    
    # Calculate mock metrics
    total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
    days = len(dates)
    annual_return = (1 + total_return) ** (365.25 / days) - 1
    
    daily_returns = np.diff(equity_values) / equity_values[:-1]
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    
    # Sortino ratio (downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (np.mean(daily_returns) / np.std(downside_returns) * np.sqrt(252) 
                    if len(downside_returns) > 0 else sharpe_ratio)
    
    max_drawdown = max(equity_curve, key=lambda x: x["drawdown"])["drawdown"]
    
    win_rate = len([t for t in trades if t["return"] > 0]) / len(trades) if trades else 0.5
    avg_trade_pnl = np.mean([t["pnl"] for t in trades]) if trades else 0.0
    
    profit_trades = [t["pnl"] for t in trades if t["pnl"] > 0]
    loss_trades = [abs(t["pnl"]) for t in trades if t["pnl"] < 0]
    profit_factor = (sum(profit_trades) / sum(loss_trades) 
                    if loss_trades and sum(loss_trades) > 0 else 1.0)
    
    raw_metrics = {
        "cagr": annual_return,  # Will be normalized to annual_return
        "sharpe": sharpe_ratio,  # Will be normalized to sharpe_ratio
        "sortino": sortino_ratio,
        "max_dd": max_drawdown,  # Will be normalized to max_drawdown
        "win_rate_pct": win_rate,  # Will be normalized to win_rate
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_pnl,  # Will be normalized to avg_trade_pnl
        "total_trades": len(trades)
    }
    
    return {
        "raw_metrics": raw_metrics,
        "equity_curve": equity_curve,
        "trades": trades,
        "metadata": {
            "total_days": days,
            "start_equity": float(equity_values[0]),
            "end_equity": float(equity_values[-1])
        }
    }


@register("bt.run")
def bt_run(**kwargs) -> ToolResult:
    """
    Run backtest with metric normalization
    
    Executes a backtest and normalizes metrics to standard format
    """
    try:
        inputs = BTRunIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    start_time = time.time()
    warnings = []
    
    try:
        # Generate mock backtest data (replace with real backtest engine)
        backtest_data = generate_mock_backtest_data(
            inputs.strategy_id, inputs.symbols, inputs.start, inputs.end
        )
        
        warnings.append("Using mock backtest engine - integrate with real backtester in production")
        
        # Normalize metrics
        normalized_metrics, metric_warnings = normalize_metrics(backtest_data["raw_metrics"])
        warnings.extend(metric_warnings)
        
        # Create BTMetrics object
        metrics = BTMetrics(**normalized_metrics)
        
        # Create result
        result = BTResult(
            strategy_id=inputs.strategy_id,
            symbols=inputs.symbols,
            interval=inputs.interval,
            start_date=inputs.start,
            end_date=inputs.end,
            metrics=metrics,
            equity_curve=backtest_data["equity_curve"],
            trades=backtest_data["trades"],
            metadata={
                **backtest_data["metadata"],
                "backtest_duration_ms": int((time.time() - start_time) * 1000),
                "cost_bps": inputs.cost_bps,
                "slippage_bps": inputs.slippage_bps,
                "vol_target": inputs.vol_target,
                "walk_forward": inputs.walk_forward
            },
            warnings=warnings
        )
        
        return ToolResult.success(
            data={
                "backtest_result": result.model_dump(),
                "summary": {
                    "annual_return": metrics.annual_return,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "max_drawdown": metrics.max_drawdown,
                    "total_trades": metrics.total_trades,
                    "win_rate": metrics.win_rate
                }
            },
            warnings=warnings
        )
        
    except Exception as e:
        return ToolResult.error([f"Backtest execution failed: {str(e)}"])


@register("bt.normalize_metrics")
def bt_normalize_metrics(**kwargs) -> ToolResult:
    """
    Normalize backtest metrics to standard format
    
    Utility tool to normalize metric dictionaries
    """
    raw_metrics = kwargs.get("raw_metrics", {})
    
    if not isinstance(raw_metrics, dict):
        return ToolResult.error(["raw_metrics must be a dictionary"])
    
    try:
        normalized, warnings = normalize_metrics(raw_metrics)
        
        return ToolResult.success(
            data={
                "normalized_metrics": normalized,
                "mapping_applied": METRIC_MAP
            },
            warnings=warnings
        )
        
    except Exception as e:
        return ToolResult.error([f"Metric normalization failed: {str(e)}"])


if __name__ == "__main__":
    # Test backtest tools
    print("Testing backtest tools...")
    
    # Test bt.run
    result = bt_run(
        strategy_id="test_strategy_v1",
        symbols=["BTCUSDT", "ETHUSDT"],
        interval="1h",
        start="2023-01-01", 
        end="2023-02-01"
    )
    
    print(f"Backtest result: {result.ok}")
    if result.ok:
        summary = result.data["summary"]
        print(f"Annual return: {summary['annual_return']:.2%}")
        print(f"Sharpe ratio: {summary['sharpe_ratio']:.2f}")
        print(f"Max drawdown: {summary['max_drawdown']:.2%}")
        print(f"Total trades: {summary['total_trades']}")
    
    # Test metric normalization
    raw_metrics = {
        "cagr": 0.25,
        "sharpe": 1.5,
        "max_dd": 0.15,
        "win_rate_pct": 0.65
    }
    
    norm_result = bt_normalize_metrics(raw_metrics=raw_metrics)
    print(f"Normalization result: {norm_result.ok}")
    if norm_result.ok:
        normalized = norm_result.data["normalized_metrics"]
        print("Normalized metrics:", list(normalized.keys()))
    
    print("Backtest tools test complete!")