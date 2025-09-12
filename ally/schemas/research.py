from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class EvidenceGrade(str, Enum):
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"


class Evidence(BaseModel):
    source: str = Field(..., description="Source identifier (e.g., 'sec_filing', 'news_article')")
    content: str = Field(..., description="The evidence content or excerpt")
    grade: EvidenceGrade = Field(..., description="Quality/reliability grade")
    timestamp: datetime = Field(..., description="When this evidence was collected")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional source metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Claim(BaseModel):
    statement: str = Field(..., description="The claim being made")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Bayesian confidence score")
    evidence_ids: List[str] = Field(default_factory=list, description="References to supporting evidence")
    counter_evidence_ids: List[str] = Field(default_factory=list, description="References to contradicting evidence")
    
    
class ResearchSummary(BaseModel):
    query: str = Field(..., description="Original research query")
    claims: List[Claim] = Field(..., description="Key claims extracted from research")
    evidence: List[Evidence] = Field(..., description="All evidence collected")
    methodology: str = Field(..., description="Research methodology used")
    timestamp: datetime = Field(..., description="When research was conducted")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }