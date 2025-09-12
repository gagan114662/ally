"""
Reporting schemas for Ally - tearsheet generation and report summaries
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..schemas.base import ToolInput


class GenerateTearsheetIn(ToolInput):
    """Input schema for reporting.generate_tearsheet tool"""
    run_id: str = Field(..., description="Run ID to generate report for")
    output_path: Optional[str] = Field(None, description="Output path for HTML report")
    include_trades: bool = Field(default=True, description="Include trades table in report")
    include_metrics: bool = Field(default=True, description="Include performance metrics")
    include_events: bool = Field(default=True, description="Include events timeline")


class ReportSummary(BaseModel):
    """Summary data for a generated report"""
    run_id: str = Field(..., description="Run ID the report was generated for")
    html_path: str = Field(..., description="Path to generated HTML file")
    kpis: Dict[str, Any] = Field(default_factory=dict, description="Key performance indicators")
    n_trades: int = Field(0, description="Number of trades in the report")
    n_events: int = Field(0, description="Number of events in the report")
    sections: List[str] = Field(default_factory=list, description="Report sections included")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Report generation timestamp")
    file_size_bytes: Optional[int] = Field(None, description="Size of generated HTML file")


class ReportSection(BaseModel):
    """A section of a report"""
    title: str = Field(..., description="Section title")
    content_type: str = Field(..., description="Type of content (table, chart, text, etc.)")
    content: Any = Field(..., description="Section content data")
    order: int = Field(0, description="Display order within report")