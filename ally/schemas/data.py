"""
Data tools schemas for Ally
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from ..schemas.base import ToolInput


class LoadOHLCVIn(ToolInput):
    """Input schema for data.load_ohlcv tool"""
    symbols: List[str] = Field(..., description="List of symbols to load")
    interval: str = Field(..., description="Time interval (1m, 5m, 1h, 1d, etc.)")
    start: str = Field(..., description="Start date (YYYY-MM-DD or ISO format)")
    end: str = Field(..., description="End date (YYYY-MM-DD or ISO format)")
    source: str = Field("mock", description="Data source (alpha_vantage, polygon, finnhub, quandl, yfinance, csv, mock)")
    data_path: Optional[str] = Field(None, description="Path to data files")
    live: bool = Field(False, description="Enable live data fetching (requires ALLY_LIVE=1)")
    api_key: Optional[str] = Field(None, description="API key for live data sources")
    

class LoadDataIn(ToolInput):
    """Input schema for generic data loading"""
    source: str = Field(..., description="Data source identifier")
    query: Optional[str] = Field(None, description="SQL query or data filter")
    params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    limit: Optional[int] = Field(None, description="Maximum rows to return")
    

# Output schemas
class OHLCVData(BaseModel):
    """OHLCV data structure"""
    symbol: str
    interval: str
    data: List[Dict[str, Union[str, float, int]]]  # timestamp, open, high, low, close, volume
    start_date: datetime
    end_date: datetime
    total_rows: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataPanel(BaseModel):
    """Multi-symbol data panel"""
    symbols: List[str]
    interval: str
    start_date: datetime
    end_date: datetime
    total_rows: int
    aligned_data: Dict[str, List[Dict[str, Any]]]  # symbol -> OHLCV records
    index_column: str = "timestamp"
    metadata: Dict[str, Any] = Field(default_factory=dict)