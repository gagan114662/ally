from pydantic import BaseModel, Field
from typing import List, Optional

class Candidate(BaseModel):
    sid: str
    spa_pvalue: float
    resid_alpha_t: float

class FdrInput(BaseModel):
    q: float = Field(0.05, ge=0, le=0.5)
    candidates: List[Candidate]
    promotion_budget: Optional[int] = None

class FdrResult(BaseModel):
    q: float
    total: int
    passed: List[str]
    psi_ok: bool
    det_hash: str