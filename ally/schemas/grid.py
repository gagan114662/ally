# ally/schemas/grid.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class GridJobStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DEDUPED = "DEDUPED"
    RESUMED = "RESUMED"

class GridJob(BaseModel):
    job_id: str
    strategy_config: Dict[str, Any]
    config_hash: str
    status: str = GridJobStatus.PENDING
    worker_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class GridBatch(BaseModel):
    batch_id: str
    jobs: List[GridJob]
    n_submitted: int = 0
    n_completed: int = 0
    n_failed: int = 0
    n_deduped: int = 0
    n_resumed: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))