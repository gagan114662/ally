"""
Memory schemas for Ally - logging and querying experimental runs
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from ..schemas.base import ToolInput


class LogRunIn(ToolInput):
    """Input schema for memory.log_run tool"""
    run_id: str = Field(..., description="Unique identifier for the run")
    task: str = Field(..., description="Task or tool name that generated this run")
    code_hash: str = Field(..., description="Hash of the code that generated the run")
    inputs_hash: str = Field(..., description="Hash of the inputs used")
    ts: str = Field(..., description="Timestamp in ISO-8601 format with Z suffix")
    metrics: Dict[str, Union[float, int]] = Field(default_factory=dict, description="Performance metrics")
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Events that occurred during the run")
    trades: List[Dict[str, Any]] = Field(default_factory=list, description="Trades executed during the run")
    notes: Optional[str] = Field(None, description="Additional notes about the run")


class QueryIn(ToolInput):
    """Input schema for memory.query tool"""
    query: Optional[str] = Field(None, description="SQL query to execute")
    table: Optional[str] = Field(None, description="Table name to query (shorthand for SELECT * FROM table)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    limit: Optional[int] = Field(None, description="Maximum number of rows to return")


class QueryOut(BaseModel):
    """Output schema for memory.query results"""
    rows: List[Dict[str, Any]] = Field(default_factory=list, description="Query result rows")
    count: int = Field(0, description="Number of rows returned")
    columns: List[str] = Field(default_factory=list, description="Column names")
    execution_time_ms: Optional[float] = Field(None, description="Query execution time in milliseconds")


class RunRecord(BaseModel):
    """Schema for a stored run record"""
    run_id: str
    task: str
    code_hash: str
    inputs_hash: str
    ts: datetime
    metrics: Dict[str, Union[float, int]]
    events: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    notes: Optional[str]
    created_at: datetime = Field(default_factory=datetime.utcnow)