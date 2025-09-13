# ally/schemas/promotion.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone

class PromotionDecision(str):
    PASS = "PASS"
    FAIL = "FAIL"

class PromotionBundleSummary(BaseModel):
    strategy_id: str
    selection_sha1: str
    code_hash: str
    params_hash: str
    receipts_sha1: List[str] = []
    metrics: Dict[str, Any] = {}
    decision: str = PromotionDecision.FAIL
    artifacts: Dict[str, str] = {}  # name -> relative path
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    bundle_sha1: Optional[str] = None