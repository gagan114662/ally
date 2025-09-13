from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class ReceiptRef(BaseModel):
    """Reference to a data receipt for provenance tracking."""
    content_sha1: str = Field(..., description="SHA1 hash linking to receipt")
    vendor: str = Field(..., description="Data provider name") 
    endpoint: str = Field(..., description="API endpoint called")
    ts_iso: str = Field(..., description="ISO timestamp of data fetch")
    cost_cents: Optional[int] = Field(None, description="Cost in cents")


class ReportSummary(BaseModel):
    """Extended report summary with end-to-end receipt provenance."""
    run_id: str = Field(..., description="Unique run identifier")
    task: str = Field(..., description="Task description")
    ts_iso: str = Field(..., description="Run timestamp")
    kpis: Dict[str, float] = Field(default_factory=dict, description="Key performance indicators")
    n_trades: int = Field(default=0, description="Number of trades executed")
    sections: List[str] = Field(default_factory=list, description="Report sections")
    html_path: str = Field(..., description="Path to HTML report")
    receipts: List[ReceiptRef] = Field(default_factory=list, description="Linked data receipts")
    receipt_cost_cents: int = Field(default=0, description="Total data cost")
    receipt_vendors: List[str] = Field(default_factory=list, description="Unique vendors used")
    audit_hash: str = Field(..., description="SHA256 of inputs + outputs + receipts")
    inputs_hash: str = Field(..., description="Hash of input parameters")
    code_hash: str = Field(..., description="Hash of function code")