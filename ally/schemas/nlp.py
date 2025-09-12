"""
NLP Event Extraction Schemas for Ally
Extract structured financial events from text with ticker, date, category, and sentiment
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional

Sentiment = Literal["neg", "neu", "pos"]
Category = Literal["earnings", "guidance", "product", "litigation", "regulatory", "macro", "other"]

class NLPEventIn(BaseModel):
    """Input for NLP event extraction"""
    sources: List[str] = Field(..., description="List of sources: file:// paths or text:// content")
    tickers: Optional[List[str]] = Field(None, description="Filter events to these tickers only")
    window_days: int = Field(default=5, ge=0, le=30, description="Event study window size in days")

class NLPEvent(BaseModel):
    """A single extracted financial event"""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL)")
    date: str = Field(..., description="Event date in ISO-8601 with Z suffix")
    category: Category = Field(..., description="Event category")
    sentiment: Sentiment = Field(..., description="Sentiment classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0,1]")
    snippet: str = Field(..., description="Text snippet around the event (~120 chars)")
    source_path: str = Field(..., description="Source file or text:// identifier")

class NLPEventOut(BaseModel):
    """Output from NLP event extraction"""
    events: List[NLPEvent] = Field(default_factory=list, description="Extracted events")
    window_days: int = Field(..., description="Event window size used")
    audit_hash: str = Field(..., description="Deterministic hash for reproducibility")