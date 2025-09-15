"""
Replication pipeline runner for strategy specs.
Orchestrates PIT universe → features → signals → weights → backtest → gates.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

from ally.utils.receipts import generate_receipt
from ally.utils.gating import check_live_mode_allowed
from ally.core.tool_registry import register, ToolResult
from ally.research.spec import StrategySpec


def generate_mock_universe(spec: StrategySpec, live: bool = False) -> pd.DataFrame:
    """Generate PIT-compliant mock universe"""
    if live:
        # In live mode, would connect to actual data provider
        raise NotImplementedError("Live universe generation not implemented")
    
    # Mock deterministic universe for CI
    np.random.seed(spec.backtest.seed)
    
    # Generate mock symbols based on universe spec
    if spec.universe.asset_class == "equities_us":
        symbols = [f"STOCK{i:03d}" for i in range(1, 501)]  # Mock S&P 500
    elif spec.universe.asset_class == "futures_global":
        symbols = ["CL1", "GC1", "ES1", "ZN1", "ZC1"]  # Mock futures
    else:
        symbols = [f"SYM{i:03d}" for i in range(1, 101)]
    
    # Generate date range
    start_date = pd.to_datetime(spec.backtest.start)
    end_date = pd.to_datetime(spec.backtest.end)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create PIT universe with delistings
    universe_data = []
    for date in dates:
        # Mock: 95% of universe active each day (simulates delistings)
        active_symbols = np.random.choice(symbols, size=int(len(symbols) * 0.95), replace=False)
        for symbol in active_symbols:
            universe_data.append({
                'date': date,
                'symbol': symbol,
                'active': True,
                'market_cap': np.random.lognormal(15, 2),  # Mock market cap
                'sector': f"SECTOR{hash(symbol) % 10}"
            })
    
    return pd.DataFrame(universe_data)


def generate_mock_ohlcv(universe_df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Generate PIT-compliant mock OHLCV data"""
    np.random.seed(spec.backtest.seed + 1)
    
    ohlcv_data = []
    for _, row in universe_df.iterrows():
        # Generate realistic OHLCV with trends and volatility
        base_price = 100 + np.random.randn() * 50
        base_price = max(base_price, 10)  # Floor at $10
        
        # Add momentum and volatility
        daily_ret = np.random.randn() * 0.02  # 2% daily vol
        volume = np.random.lognormal(10, 1)
        
        ohlcv_data.append({
            'date': row['date'],
            'symbol': row['symbol'],
            'open': base_price,
            'high': base_price * (1 + abs(daily_ret)),
            'low': base_price * (1 - abs(daily_ret)),
            'close': base_price * (1 + daily_ret),
            'volume': volume
        })
    
    return pd.DataFrame(ohlcv_data)


def generate_mock_fundamentals(universe_df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Generate PIT-compliant mock fundamental data"""
    np.random.seed(spec.backtest.seed + 2)
    
    fund_data = []
    for symbol in universe_df['symbol'].unique():
        # Generate quarterly fundamental data
        dates = pd.date_range(spec.backtest.start, spec.backtest.end, freq='QS')
        
        for date in dates:
            # Mock fundamental metrics
            market_cap = np.random.lognormal(15, 2)
            book_equity = market_cap * np.random.uniform(0.3, 2.0)  # Book-to-market variation
            
            fund_data.append({
                'date': date,
                'symbol': symbol,
                'market_cap': market_cap,
                'book_equity': book_equity,
                'book_to_market': book_equity / market_cap,
                'industry': f"IND{hash(symbol) % 20}"
            })
    
    return pd.DataFrame(fund_data)


def compute_signal(ohlcv_df: pd.DataFrame, fund_df: Optional[pd.DataFrame], 
                  spec: StrategySpec) -> pd.DataFrame:
    """Compute strategy signal from data"""
    if spec.signal.type == "cross_sectional" and "momentum" in spec.name:
        # XS-Momentum: 12m-1m returns
        signal_data = []
        
        for symbol in ohlcv_df['symbol'].unique():
            sym_data = ohlcv_df[ohlcv_df['symbol'] == symbol].sort_values('date')
            if len(sym_data) < 252:  # Need full year of data
                continue
                
            sym_data['ret_1d'] = sym_data['close'].pct_change()
            sym_data['ret_12m'] = sym_data['close'].pct_change(252)
            sym_data['ret_1m'] = sym_data['close'].pct_change(21)
            sym_data['momentum_signal'] = sym_data['ret_12m'] - sym_data['ret_1m']
            
            # Rebalance monthly
            monthly_dates = sym_data.groupby(pd.Grouper(key='date', freq='M')).last()
            
            for _, row in monthly_dates.iterrows():
                if not pd.isna(row['momentum_signal']):
                    signal_data.append({
                        'date': row['date'],
                        'symbol': symbol,
                        'signal': row['momentum_signal'],
                        'ret_12m': row['ret_12m'],
                        'ret_1m': row['ret_1m']
                    })
    
    elif spec.signal.type == "cross_sectional" and "value" in spec.name:
        # Value: Book-to-Market
        signal_data = []
        
        if fund_df is not None:
            for _, row in fund_df.iterrows():
                signal_data.append({
                    'date': row['date'],
                    'symbol': row['symbol'],
                    'signal': row['book_to_market'],
                    'book_equity': row['book_equity'],
                    'market_cap': row['market_cap']
                })
    
    elif spec.signal.type == "time_series" and "trend" in spec.name:
        # TS-Trend: SMA crossover
        signal_data = []
        
        for symbol in ohlcv_df['symbol'].unique():
            sym_data = ohlcv_df[ohlcv_df['symbol'] == symbol].sort_values('date')
            if len(sym_data) < 100:  # Need SMA data
                continue
                
            sym_data['sma_20'] = sym_data['close'].rolling(20).mean()
            sym_data['sma_100'] = sym_data['close'].rolling(100).mean()
            sym_data['trend_signal'] = np.where(sym_data['sma_20'] > sym_data['sma_100'], 1, -1)
            
            for _, row in sym_data.iterrows():
                if not pd.isna(row['trend_signal']):
                    signal_data.append({
                        'date': row['date'],
                        'symbol': symbol,
                        'signal': row['trend_signal'],
                        'sma_20': row['sma_20'],
                        'sma_100': row['sma_100']
                    })
    
    return pd.DataFrame(signal_data)


def compute_weights(signal_df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Compute portfolio weights from signals"""
    weights_data = []
    
    for date in signal_df['date'].unique():
        date_signals = signal_df[signal_df['date'] == date]
        
        if spec.portfolio.scheme == "equal_weight_top_k":
            # Equal weight top-K by signal
            top_k = date_signals.nlargest(spec.portfolio.k, 'signal')
            weight_per_stock = 1.0 / len(top_k)
            
            for _, row in top_k.iterrows():
                weights_data.append({
                    'date': date,
                    'symbol': row['symbol'],
                    'weight': weight_per_stock,
                    'signal_rank': len(top_k) - len(top_k[top_k['signal'] >= row['signal']]) + 1
                })
                
        elif spec.portfolio.scheme == "risk_parity":
            # Risk parity (simplified: equal vol contribution)
            # Mock: assume equal volatility, so equal weights
            n_assets = len(date_signals)
            if n_assets > 0:
                equal_weight = 1.0 / n_assets
                
                for _, row in date_signals.iterrows():
                    weights_data.append({
                        'date': date,
                        'symbol': row['symbol'],
                        'weight': equal_weight * row['signal'],  # Scale by signal
                        'signal_value': row['signal']
                    })
    
    return pd.DataFrame(weights_data)


def run_backtest(weights_df: pd.DataFrame, ohlcv_df: pd.DataFrame, 
                spec: StrategySpec) -> Dict[str, Any]:
    """Run deterministic backtest"""
    # Join weights with returns
    backtest_data = weights_df.merge(ohlcv_df, on=['date', 'symbol'], how='left')
    backtest_data['ret_1d'] = backtest_data.groupby('symbol')['close'].pct_change()
    
    # Compute portfolio returns
    backtest_data['contribution'] = backtest_data['weight'] * backtest_data['ret_1d']
    
    portfolio_returns = backtest_data.groupby('date')['contribution'].sum()
    
    # Apply transaction costs
    turnover_data = weights_df.groupby('date')['weight'].sum().diff().abs()
    cost_drag = turnover_data * (spec.costs.bps_per_turnover / 10000.0)
    
    net_returns = portfolio_returns - cost_drag.fillna(0)
    
    # Compute equity curve
    equity_curve = (1 + net_returns.fillna(0)).cumprod()
    
    # Summary statistics
    annual_return = (equity_curve.iloc[-1] ** (252 / len(equity_curve))) - 1
    annual_vol = net_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    max_dd = (equity_curve / equity_curve.expanding().max() - 1).min()
    
    return {
        "equity_curve": equity_curve.to_dict(),
        "portfolio_returns": portfolio_returns.to_dict(),
        "net_returns": net_returns.to_dict(),
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_return": equity_curve.iloc[-1] - 1,
        "backtest_start": spec.backtest.start,
        "backtest_end": spec.backtest.end
    }


@register("research.replication.run")
def research_replication_run(spec_path: str, live: bool = False, walkforward: bool = False, **kwargs) -> ToolResult:
    """
    Run complete replication pipeline for strategy spec
    
    Args:
        spec_path: Path to strategy specification YAML
        live: Whether to use live data
        walkforward: Whether to run walk-forward analysis
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Research Replication")
    
    # If walkforward requested, delegate to walk-forward analysis
    if walkforward:
        from ally.research.walkforward import research_walkforward_run
        return research_walkforward_run(
            spec_path=spec_path, 
            live=live, 
            **kwargs
        )
    
    try:
        # Load spec
        spec = StrategySpec.from_yaml(spec_path)
        
        # 1. Generate PIT universe
        universe_df = generate_mock_universe(spec, live)
        
        # 2. Generate data
        ohlcv_df = generate_mock_ohlcv(universe_df, spec)
        fund_df = generate_mock_fundamentals(universe_df, spec) if spec.data.fundamentals else None
        
        # 3. Compute signals
        signal_df = compute_signal(ohlcv_df, fund_df, spec)
        
        # 4. Compute weights
        weights_df = compute_weights(signal_df, spec)
        
        # 5. Run backtest
        backtest_results = run_backtest(weights_df, ohlcv_df, spec)
        
        # Generate receipts for each stage
        universe_receipt = generate_receipt("replication.universe", universe_df.to_dict())
        signal_receipt = generate_receipt("replication.signal", signal_df.to_dict()) 
        weights_receipt = generate_receipt("replication.weights", weights_df.to_dict())
        backtest_receipt = generate_receipt("replication.backtest", backtest_results)
        
        # Combined results
        result = {
            "spec_name": spec.name,
            "universe_count": len(universe_df),
            "signal_count": len(signal_df),
            "weights_count": len(weights_df),
            "backtest_results": backtest_results,
            "receipts": {
                "universe": universe_receipt[:16],
                "signal": signal_receipt[:16], 
                "weights": weights_receipt[:16],
                "backtest": backtest_receipt[:16]
            },
            "pipeline_timestamp": "2024-01-15T10:00:00Z"
        }
        
        # Main receipt for full pipeline
        pipeline_receipt = generate_receipt("research.replication.run", result)
        
        return ToolResult(
            ok=True,
            data=result,
            receipt_hash=pipeline_receipt
        )
        
    except Exception as e:
        error_dict = {"error": str(e), "spec_path": spec_path}
        receipt_hash = generate_receipt("research.replication.error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Replication pipeline failed: {e}"],
            receipt_hash=receipt_hash
        )