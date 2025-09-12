"""
Web tools schemas for Ally
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from ..schemas.base import ToolInput


class WebFetchIn(ToolInput):
    """Input schema for web.fetch tool"""
    url: str = Field(..., description="URL to fetch content from")
    as_pdf: bool = Field(False, description="Treat content as PDF")
    timeout_s: int = Field(30, description="Request timeout in seconds")
    user_agent: str = Field(
        "Ally-Bot/1.0 (Research Tool)",
        description="User agent string"
    )
    follow_redirects: bool = Field(True, description="Follow HTTP redirects")
    max_size_mb: float = Field(10.0, description="Maximum content size in MB")


class WebSearchIn(ToolInput):
    """Input schema for web.search tool"""
    query: str = Field(..., description="Search query")
    limit: int = Field(5, description="Maximum number of results")
    safe_search: bool = Field(True, description="Enable safe search")
    region: str = Field("us", description="Search region")
    time_filter: Optional[str] = Field(None, description="Time filter (day, week, month, year)")


class WebReadTablesIn(ToolInput):
    """Input schema for web.read_tables tool"""
    url: str = Field(..., description="URL or file path containing tables")
    format: str = Field("html", description="Content format (html, pdf)")
    table_index: Optional[int] = Field(None, description="Specific table index to extract")
    encoding: str = Field("utf-8", description="Text encoding")


# Output schemas
class WebPage(BaseModel):
    """Structured web page content"""
    url: str
    title: Optional[str] = None
    content: str
    clean_text: str
    links: List[Dict[str, str]] = Field(default_factory=list)
    images: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    content_type: str = "text/html"
    size_bytes: int = 0
    fetch_time: float = 0.0


class SearchResult(BaseModel):
    """Individual search result"""
    title: str
    url: str
    snippet: str
    rank: int
    domain: str


class TableData(BaseModel):
    """Extracted table data"""
    headers: List[str]
    rows: List[List[str]]
    table_index: int
    shape: tuple  # (rows, columns)
    metadata: Dict[str, Any] = Field(default_factory=dict)