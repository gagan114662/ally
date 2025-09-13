from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, Dict, Any

class CacheConfig(BaseModel):
    enabled: bool = True
    dir: str = "runs/cache"
    # TTL kept for future use; deterministic CI ignores expiry
    ttl_seconds: Optional[int] = None

class CacheEntry(BaseModel):
    key: str
    engine: str
    task: str
    prompt: str
    system: Optional[str] = None
    params: Dict[str, Any] = {}
    output: str