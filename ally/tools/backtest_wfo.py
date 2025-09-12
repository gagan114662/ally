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
    # 1) load data (reuse your existing loaders/fixtures if df is None)
    if df is None:
        from ally.tools import TOOL_REGISTRY
        sample_data = TOOL_REGISTRY["data.create_sample"](symbol="BTCUSDT", n=2000)
        df = pd.DataFrame(sample_data.data["frame"])

    idx = pd.to_datetime(df["timestamp"])
    splits = make_walkforward_splits(idx, WalkForwardConfig(window_train, window_test, mode))
    pkf = PurgedKFold(n_splits=max(2, len(splits)), embargo_frac=embargo_frac)

    # 2) run per-split backtests (reuse your bt.run)
    from ally.tools import TOOL_REGISTRY
    kpis_train = []; kpis_oos = []; split_meta = []

    for i, (train_idx, test_idx) in enumerate(splits):
        train = df.iloc[train_idx]; test = df.iloc[test_idx]
        # synthesize a simple strategy via existing bt.run or mock evaluation
        r_train = TOOL_REGISTRY["bt.run"](_df=train).data["returns"]
        r_test  = TOOL_REGISTRY["bt.run"](_df=test).data["returns"]

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
        rep = TOOL_REGISTRY["reporting.generate_tearsheet"](run_id=f"WFO_{experiment_id}")
        summary["report_path"] = rep.data.get("summary", {}).get("html_path") or rep.data.get("html_path")

    meta = Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name":"bt.walkforward"})
    return ToolResult(ok=True, data=summary, errors=[], meta=meta)