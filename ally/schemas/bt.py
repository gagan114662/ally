"""
Backtest tools schemas for Ally
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from ..schemas.base import ToolInput


class BTRunIn(ToolInput):
    """Input schema for bt.run tool"""
    strategy_id: str = Field(..., description="Strategy identifier")
    symbols: List[str] = Field(..., description="List of symbols to backtest")
    interval: str = Field(..., description="Time interval")
    start: str = Field(..., description="Start date")
    end: str = Field(..., description="End date")
    cost_bps: float = Field(2.0, description="Transaction cost in basis points")
    slippage_bps: float = Field(1.0, description="Slippage in basis points")
    vol_target: float = Field(0.1, description="Volatility target")
    walk_forward: bool = Field(True, description="Use walk-forward analysis")


class BTMetrics(BaseModel):
    """Normalized backtest metrics"""
    annual_return: float = Field(..., description="Annualized return")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: float = Field(..., description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    win_rate: float = Field(..., description="Win rate")
    profit_factor: float = Field(..., description="Profit factor")
    avg_trade_pnl: float = Field(..., description="Average trade P&L")
    total_trades: int = Field(..., description="Total number of trades")


class BTResult(BaseModel):
    """Backtest result"""
    strategy_id: str
    symbols: List[str]
    interval: str
    start_date: str
    end_date: str
    metrics: BTMetrics
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    trades: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)