from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional

class LogRunIn(BaseModel):
    run_id: str
    task: str
    code_hash: str
    inputs_hash: str
    ts: str                 # ISO-Z
    metrics: Dict[str, float] = {}
    events: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    notes: Optional[str] = None

class QueryIn(BaseModel):
    table: str              # runs|metrics|events|trades
    where: Optional[str] = None
    limit: int = 100

class QueryOut(BaseModel):
    rows: List[Dict[str, Any]]
    count: int