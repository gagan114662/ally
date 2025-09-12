from __future__ import annotations
import os
from datetime import datetime
from ally.schemas.base import ToolResult, Meta

def web_search_tavily(q: str, recency_days: int = 30, live: bool = False) -> ToolResult:
    if not live and os.getenv("ALLY_LIVE","0") != "1":
        return ToolResult(ok=False, data={"error":"live disabled; set live=true or ALLY_LIVE=1"},
                          errors=["live_disabled"],
                          meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name":"web.search_tavily"}))
    # For this PR, live path not implemented
    return ToolResult(ok=False, data={"error":"tavily live path not implemented in this PR"},
                      errors=["not_implemented"],
                      meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name":"web.search_tavily"}))