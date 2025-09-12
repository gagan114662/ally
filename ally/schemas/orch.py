from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class OrchInput(BaseModel):
    experiment_id: str
    symbols: List[str] = ["BTCUSDT"]
    interval: str = "1h"
    lookback: int = 600
    targets: Dict[str, float] = {"annual_return": 0.10, "sharpe_ratio": 1.0}
    risk_policy_yaml: str = "max_leverage: 3.0\nmax_single_order_notional: 25000"
    save_run: bool = True
    make_report: bool = True

class OrchSummary(BaseModel):
    experiment_id: str
    run_id: str
    best_params: Dict[str, Any]
    kpis: Dict[str, float]
    report_path: Optional[str] = None