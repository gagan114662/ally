from pydantic import BaseModel
from typing import Dict, List, Optional

class ReportSummary(BaseModel):
    run_id: str
    kpis: Dict[str, float]
    n_trades: int
    sections: List[str]
    html_path: str