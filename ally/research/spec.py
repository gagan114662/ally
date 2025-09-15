"""
Spec schema and validator for research strategies.
Defines YAML spec structure for PIT-compliant backtesting pipeline.
"""

import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from ally.utils.receipts import generate_receipt
from ally.utils.gating import check_live_mode_allowed
from ally.core.tool_registry import register, ToolResult


@dataclass
class UniverseSpec:
    """Universe definition with PIT compliance"""
    asset_class: str
    inclusion: Dict[str, Any]


@dataclass
class DataSpec:
    """Data requirements with PIT enforcement"""
    ohlcv: Optional[Dict[str, Any]] = None
    fundamentals: Optional[Dict[str, Any]] = None


@dataclass
class SignalSpec:
    """Signal definition with lookbacks and rebalancing"""
    type: str  # cross_sectional or time_series
    formula: str
    lookbacks: Optional[Dict[str, int]] = None
    winsorize: Optional[Dict[str, float]] = None
    neutralize: Optional[Dict[str, str]] = None
    rebalance: str = "monthly"


@dataclass
class PortfolioSpec:
    """Portfolio construction parameters"""
    scheme: str
    k: Optional[int] = None  # for top-K schemes
    vol_target_annual: Optional[float] = None  # for risk parity
    constraints: Dict[str, float] = None


@dataclass
class CostSpec:
    """Transaction cost modeling"""
    bps_per_turnover: float = 10.0


@dataclass
class BacktestSpec:
    """Backtest parameters"""
    start: str
    end: str
    benchmark: str
    seed: int = 42


@dataclass
class GateSpec:
    """Research gate definitions"""
    factorlens: Optional[Dict[str, Any]] = None
    fdr: Optional[Dict[str, Any]] = None
    promotion: Optional[Dict[str, Any]] = None


@dataclass
class ProofSpec:
    """PROOF line emission settings"""
    emit: bool = True


@dataclass
class StrategySpec:
    """Complete strategy specification"""
    name: str
    universe: UniverseSpec
    data: DataSpec
    signal: SignalSpec
    portfolio: PortfolioSpec
    costs: CostSpec
    backtest: BacktestSpec
    gates: GateSpec
    proof: ProofSpec
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'StrategySpec':
        """Load spec from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            name=data['name'],
            universe=UniverseSpec(**data['universe']),
            data=DataSpec(**data.get('data', {})),
            signal=SignalSpec(**data['signal']),
            portfolio=PortfolioSpec(**data['portfolio']),
            costs=CostSpec(**data.get('costs', {})),
            backtest=BacktestSpec(**data['backtest']),
            gates=GateSpec(**data.get('gates', {})),
            proof=ProofSpec(**data.get('proof', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing"""
        return {
            'name': self.name,
            'universe': self.universe.__dict__,
            'data': self.data.__dict__,
            'signal': self.signal.__dict__,
            'portfolio': self.portfolio.__dict__,
            'costs': self.costs.__dict__,
            'backtest': self.backtest.__dict__,
            'gates': self.gates.__dict__,
            'proof': self.proof.__dict__
        }


@register("research.spec.validate")
def research_spec_validate(spec_path: str, live: bool = False, **kwargs) -> ToolResult:
    """
    Validate strategy specification YAML
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Research Spec Validation")
    
    try:
        # Load and validate spec
        spec = StrategySpec.from_yaml(spec_path)
        
        # Generate receipt
        spec_dict = spec.to_dict()
        receipt_hash = generate_receipt("research.spec.validate", spec_dict)
        
        result = {
            "spec_valid": True,
            "spec_name": spec.name,
            "spec_hash": receipt_hash[:16],
            "validation_timestamp": "2024-01-15T10:00:00Z"
        }
        
        return ToolResult(
            ok=True,
            data=result,
            receipt_hash=receipt_hash
        )
        
    except Exception as e:
        error_dict = {"error": str(e), "spec_path": spec_path}
        receipt_hash = generate_receipt("research.spec.validate_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Spec validation failed: {e}"],
            receipt_hash=receipt_hash
        )


@register("research.spec.load")
def research_spec_load(spec_path: str, live: bool = False, **kwargs) -> ToolResult:
    """
    Load strategy specification from YAML
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Research Spec Loading")
    
    try:
        # Load spec
        spec = StrategySpec.from_yaml(spec_path)
        
        # Generate receipt
        spec_dict = spec.to_dict()
        receipt_hash = generate_receipt("research.spec.load", spec_dict)
        
        result = {
            "spec_loaded": True,
            "spec_name": spec.name,
            "spec_dict": spec_dict,
            "spec_hash": receipt_hash[:16],
            "load_timestamp": "2024-01-15T10:00:00Z"
        }
        
        return ToolResult(
            ok=True,
            data=result,
            receipt_hash=receipt_hash
        )
        
    except Exception as e:
        error_dict = {"error": str(e), "spec_path": spec_path}
        receipt_hash = generate_receipt("research.spec.load_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Spec loading failed: {e}"],
            receipt_hash=receipt_hash
        )