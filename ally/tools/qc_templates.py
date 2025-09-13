"""
QuantConnect algorithm template generation tools
"""

from __future__ import annotations
import os
import json
import hashlib
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ally.schemas.base import ToolResult, Meta
from ally.qc.runtime_asserts import inject_asserts_into_template


def _norm_hash(text: str) -> str:
    """Stable content hash independent of trailing spaces/line endings"""
    norm = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    return hashlib.sha1(norm.encode()).hexdigest()


def parse_date(date_str: str) -> tuple[int, int, int]:
    """Parse date string into year, month, day tuple"""
    if isinstance(date_str, str):
        parsed = datetime.strptime(date_str, "%Y-%m-%d").date()
    elif isinstance(date_str, date):
        parsed = date_str
    else:
        raise ValueError(f"Invalid date format: {date_str}")
    return parsed.year, parsed.month, parsed.day


def qc_generate_python(
    class_name: str,
    symbols: List[str],
    start: str = "2020-01-01",
    end: str = "2020-01-03", 
    initial_cash: int = 100000,
    warmup_bars: int = 10,
    trade_logic: str = "pass",
    use_template_v2: bool = True,
    # Legacy parameters for backward compatibility
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    resolution: str = "Minute",
    indicators: Optional[List[Dict[str, Any]]] = None,
    rebalance_logic: Optional[str] = None,
    description: str = "Algorithmic trading strategy",
    strategy: str = "Buy and Hold",
    output_dir: str = "build/qc"
) -> ToolResult:
    """
    Generate QC-safe Python algorithm from template
    
    Args:
        class_name: Algorithm class name (must match filename)
        symbols: List of symbols to trade
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format  
        initial_cash: Starting capital
        warmup_bars: Number of bars for warmup
        trade_logic: Custom trading logic code
        use_template_v2: Use safer v2 template (default True)
        
    Returns:
        ToolResult with generated file path, template hash, and metadata
    """
    try:
        # Handle legacy parameter names
        start_val = start_date if start_date else start
        end_val = end_date if end_date else end
        
        # Validate inputs
        if not class_name or not class_name.isidentifier():
            raise ValueError(f"Invalid class name: {class_name}")
        
        if not symbols:
            raise ValueError("At least one symbol is required")
            
        # Parse dates  
        start_year, start_month, start_day = parse_date(start_val)
        end_year, end_month, end_day = parse_date(end_val)
        
        # Split symbols into equities/cryptos with default markets/resolutions
        equities, cryptos = [], []
        for s in symbols:
            if s.upper().endswith("USD") or s.upper().endswith("USDT"):
                # assume crypto; Market default = Coinbase unless caller overrides later
                cryptos.append({"ticker": s, "market": "Coinbase", "resolution": "Minute"})
            else:
                equities.append({"ticker": s, "resolution": "Minute"})
        
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent.parent / "qc" / "templates"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        
        template_name = "py_algo_v2.j2" if use_template_v2 else "py_algo.j2"
        template = env.get_template(template_name)
        
        # Render template context
        ctx = {
            "class_name": class_name,
            "start_y": start_year, "start_m": start_month, "start_d": start_day,
            "end_y": end_year, "end_m": end_month, "end_d": end_day,
            "initial_cash": initial_cash,
            "equities": equities,
            "cryptos": cryptos,
            "warmup_bars": warmup_bars,
            "default_resolution": "Minute",
            "trade_logic": trade_logic,
        }
        
        code = template.render(**ctx)
        
        # Inject runtime assertions by default (M-QC Runtime Assertions)
        if use_template_v2:
            code = inject_asserts_into_template(code)
        
        # Write to file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{class_name}.py"
        
        with open(file_path, 'w') as f:
            f.write(code)
        
        return ToolResult(
            ok=True,
            data={
                "algo_path": str(file_path),
                "template": template_name,
                "template_hash": _norm_hash(code),
                "class_name": class_name,
                "symbols": symbols,
                "date_range": f"{start_val} to {end_val}",
            },
            errors=[],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.generate_python"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.generate_python"})
        )


def qc_list_templates() -> ToolResult:
    """List available QC algorithm templates"""
    try:
        template_dir = Path(__file__).parent.parent / "qc" / "templates"
        templates = []
        
        if template_dir.exists():
            for template_file in template_dir.glob("*.j2"):
                templates.append({
                    "name": template_file.stem,
                    "file": template_file.name,
                    "path": str(template_file)
                })
        
        return ToolResult(
            ok=True,
            data={"templates": templates, "count": len(templates)},
            errors=[],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.list_templates"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.list_templates"})
        )