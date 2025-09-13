from pydantic import BaseModel
from typing import Dict, Optional, List

class OrchSummary(BaseModel):
    """Orchestrator execution summary with portfolio integration"""
    # Core execution fields
    symbols: List[str] = []
    backtest_start: Optional[str] = None
    backtest_end: Optional[str] = None
    execution_success: bool = False
    
    # Portfolio fields
    port_weights: Dict[str, float] = {}
    port_weights_sum: float = 0.0
    attribution_ok: bool = False
    port_det_hash: Optional[str] = None
    
    # Metadata
    timestamp: Optional[str] = None
    duration_ms: float = 0.0