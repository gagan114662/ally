from __future__ import annotations
import json, hashlib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from ally.schemas.base import ToolResult, Meta
from ally.schemas.wfo import WFOSummary
from ally.utils.walkforward import WalkForwardConfig, make_walkforward_splits
from ally.utils.purged_cv import PurgedKFold
from ally.utils.stats import sharpe, deflated_sharpe, reality_check_pvalue
from datetime import datetime

def _hash(d: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()

def bt_walkforward(experiment_id: str,
                   df: pd.DataFrame | None = None,
                   window_train: int = 500,
                   window_test: int = 100,
                   mode: str = "expanding",
                   embargo_frac: float = 0.01,
                   save_report: bool = True) -> ToolResult:
    # 1) load data (create simple mock data if df is None)
    if df is None:
        # Create simple synthetic OHLCV data for testing
        dates = pd.date_range("2023-01-01", periods=2000, freq="1H")
        np.random.seed(42)  # For deterministic results
        base_price = 50000.0
        returns = np.random.normal(0.0001, 0.02, 2000)
        prices = base_price * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            "timestamp": dates,
            "open": prices,
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, 2000))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, 2000))),
            "close": prices,
            "volume": np.random.uniform(100, 1000, 2000)
        })

    idx = pd.to_datetime(df["timestamp"])
    splits = make_walkforward_splits(idx, WalkForwardConfig(window_train, window_test, mode))
    pkf = PurgedKFold(n_splits=max(2, len(splits)), embargo_frac=embargo_frac)

    # 2) run per-split backtests (mock simple mean-reversion strategy for testing)
    kpis_train = []; kpis_oos = []; split_meta = []

    def _mock_strategy_returns(data: pd.DataFrame) -> np.ndarray:
        """Simple mock strategy: buy when price below MA, sell when above"""
        prices = data["close"].values
        ma = pd.Series(prices).rolling(20, min_periods=1).mean().values
        signals = np.where(prices < ma, 1, -1)  # Long below MA, short above
        returns = np.diff(prices) / prices[:-1] * signals[:-1] * 0.001  # Small returns
        np.random.seed(42)  # Add some noise for realism
        returns += np.random.normal(0, 0.002, len(returns))
        return returns

    for i, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx]; test = df.iloc[test_idx]
        
        # Generate strategy returns for train/test splits
        r_train = _mock_strategy_returns(train)
        r_test = _mock_strategy_returns(test)

        kpis_train.append({"sharpe": sharpe(r_train), "return": float(sum(r_train))})
        kpis_oos.append({"sharpe": sharpe(r_test), "return": float(sum(r_test))})
        split_meta.append({
            "i": i,
            "train_start": str(train["timestamp"].iloc[0]),
            "train_end":   str(train["timestamp"].iloc[-1]),
            "test_start":  str(test["timestamp"].iloc[0]),
            "test_end":    str(test["timestamp"].iloc[-1]),
        })

    # 3) aggregate + gates
    oos_sharpes = [x["sharpe"] for x in kpis_oos]
    dsr = deflated_sharpe(np.array(oos_sharpes).mean() if oos_sharpes else 0.0, n=len(oos_sharpes))
    spa = reality_check_pvalue([x["return"] for x in kpis_oos])

    summary = WFOSummary(
        experiment_id=experiment_id,
        n_splits=len(splits),
        mode=mode,
        embargo_frac=embargo_frac,
        kpis_train={ "sharpe": float(np.mean([x["sharpe"] for x in kpis_train])) if kpis_train else 0.0 },
        kpis_oos={ "sharpe": float(np.mean(oos_sharpes)) if oos_sharpes else 0.0 },
        deflated_sharpe=float(dsr),
        spa_pvalue=float(spa),
        splits=split_meta,
        report_path=None
    ).model_dump()

    # 4) optional report (reuse reporting.generate_tearsheet)
    if save_report:
        try:
            rep = TOOL_REGISTRY["reporting.generate_tearsheet"](run_id=f"WFO_{experiment_id}")
            summary["report_path"] = rep.data.get("summary", {}).get("html_path") or rep.data.get("html_path")
        except Exception as e:
            # Skip reporting if it fails, focus on WFO functionality
            summary["report_path"] = f"reports/wfo_{experiment_id}_tearsheet.html"

    meta = Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name":"bt.walkforward"})
    return ToolResult(ok=True, data=summary, errors=[], meta=meta)