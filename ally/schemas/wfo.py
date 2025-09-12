from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List

class WFOConfigSchema(BaseModel):
    window_train: int
    window_test: int
    mode: str = Field("expanding", pattern="^(expanding|sliding)$")
    step: int | None = None

class WFOSummary(BaseModel):
    experiment_id: str
    n_splits: int
    mode: str
    embargo_frac: float
    kpis_train: Dict[str, float]
    kpis_oos: Dict[str, float]
    deflated_sharpe: float
    spa_pvalue: float
    splits: List[Dict[str, Any]]   # start/end timestamps per split
    report_path: str | None = None