from pydantic import BaseModel, Field
from typing import Dict, List
from datetime import datetime

class FactorSetMeta(BaseModel):
    name: str = "FF5+Mom"
    frequency: str = "D"
    columns: List[str] = ["MKT","SMB","HML","RMW","CMA","MOM"]

class ExposureRow(BaseModel):
    factor: str
    beta: float
    tstat: float

class ExposuresOut(BaseModel):
    r2: float
    exposures: List[ExposureRow]
    method: str = "OLS-NeweyWest"
    lags: int = 5

class ResidualAlphaOut(BaseModel):
    alpha_bps: float     # annualized (bps)
    alpha_tstat: float
    r2: float
    window_days: int
    residual_series_path: str

class FactorLensSummary(BaseModel):
    meta: FactorSetMeta
    exposures: ExposuresOut
    residual: ResidualAlphaOut
    det_hash: str
    ts_utc: datetime