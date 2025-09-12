"""
Features tools schemas for Ally
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from ..schemas.base import ToolInput


class BuildFeaturesIn(ToolInput):
    """Input schema for features.build tool"""
    symbol: str = Field(..., description="Symbol to build features for")
    interval: str = Field(..., description="Time interval")
    feature_set: List[str] = Field(..., description="List of features to build")
    lookback: int = Field(5000, description="Maximum lookback period for features")
    data_source: Optional[str] = Field(None, description="Data source to use")
    

class FeatureConfig(BaseModel):
    """Configuration for a specific feature"""
    name: str
    type: str  # 'technical', 'price', 'volume', 'derived'
    params: Dict[str, Any] = Field(default_factory=dict)
    lookback_required: int = 0
    dependencies: List[str] = Field(default_factory=list)


class ValidateLeakageIn(ToolInput):
    """Input schema for features.validate_leakage tool"""
    features_data: Dict[str, List[float]] = Field(..., description="Features data to validate")
    price_data: Dict[str, List[float]] = Field(..., description="Price data for validation")
    max_correlation_threshold: float = Field(0.95, description="Maximum allowed correlation with future prices")


# Output schemas
class FeatureData(BaseModel):
    """Feature calculation results"""
    symbol: str
    interval: str
    features: Dict[str, List[float]]  # feature_name -> values
    timestamps: List[str]
    feature_configs: List[FeatureConfig]
    total_rows: int
    lookback_used: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LeakageValidationResult(BaseModel):
    """Results of leakage validation"""
    is_valid: bool
    violations: List[Dict[str, Any]]  # List of features with leakage issues
    correlations: Dict[str, float]  # feature -> max correlation with future prices
    summary: Dict[str, Any]
    recommendations: List[str] = Field(default_factory=list)