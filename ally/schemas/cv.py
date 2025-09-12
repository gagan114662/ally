"""
Computer Vision schemas for Ally - chart pattern detection
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from ..schemas.base import ToolInput


PatternName = Literal[
    "engulfing_bull", "engulfing_bear",
    "pin_bar_bull", "pin_bar_bear", 
    "morning_star", "evening_star",
    "trendline_break", "channel_up", "channel_down", "flag_bull", "flag_bear"
]


class CVDetectIn(ToolInput):
    """Input schema for cv.detect_chart_patterns tool"""
    symbol: str = Field(..., description="Symbol to analyze")
    interval: Literal["1m", "5m", "15m", "1h", "4h", "1d", "1w"] = Field(..., description="Time interval")
    patterns: List[PatternName] = Field(..., description="List of patterns to detect")
    lookback: int = Field(default=600, ge=50, le=20000, description="Number of bars to analyze")
    confirm_with_rules: bool = Field(True, description="Apply numeric confirmation rules")
    return_image: bool = Field(False, description="Include base64 PNG of annotated chart")
    image_width: int = Field(900, description="Image width in pixels")
    image_height: int = Field(500, description="Image height in pixels")


class CVDetection(BaseModel):
    """A single chart pattern detection"""
    pattern: PatternName = Field(..., description="Pattern type detected")
    start_idx: int = Field(..., description="Start bar index")
    end_idx: int = Field(..., description="End bar index")
    strength: float = Field(..., ge=0.0, le=1.0, description="Confidence/strength (0-1)")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional pattern info")
    confirmed: bool = Field(..., description="Passed numeric confirmation rules")


class CVDetectOut(BaseModel):
    """Output schema for cv.detect_chart_patterns tool"""
    detections: List[CVDetection] = Field(default_factory=list, description="List of detected patterns")
    n_bars: int = Field(..., description="Number of bars analyzed")
    rendered: Optional[str] = Field(None, description="Base64 encoded PNG if requested")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Analysis metadata")