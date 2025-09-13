"""
QuantConnect algorithm template generation tools
"""

from __future__ import annotations
import os
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional
from jinja2 import Environment, FileSystemLoader

from ally.schemas.base import ToolResult, Meta


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
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-03", 
    initial_cash: int = 100000,
    resolution: str = "Minute",
    warmup_bars: int = 10,
    indicators: Optional[List[Dict[str, Any]]] = None,
    trade_logic: Optional[str] = None,
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
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        initial_cash: Starting capital
        resolution: Data resolution (Minute, Hour, Daily)
        warmup_bars: Number of bars for warmup
        indicators: List of indicator configs
        trade_logic: Custom trading logic code
        rebalance_logic: Custom rebalancing logic code
        description: Algorithm description
        strategy: Strategy name
        output_dir: Output directory
        
    Returns:
        ToolResult with generated file path and metadata
    """
    try:
        # Validate inputs
        if not class_name or not class_name.isidentifier():
            raise ValueError(f"Invalid class name: {class_name}")
        
        if not symbols:
            raise ValueError("At least one symbol is required")
            
        # Parse dates
        start_year, start_month, start_day = parse_date(start_date)
        end_year, end_month, end_day = parse_date(end_date)
        
        # Default indicators
        if indicators is None:
            indicators = [
                {"name": "sma_50", "type": "SMA", "period": 50, "selector": None},
                {"name": "rsi", "type": "RSI", "period": 14, "selector": None}
            ]
        
        # Setup Jinja2 environment
        template_dir = Path(__file__).parent.parent / "qc" / "templates"
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("py_algo.j2")
        
        # Render template
        code = template.render(
            class_name=class_name,
            symbols=symbols,
            start_year=start_year,
            start_month=start_month, 
            start_day=start_day,
            end_year=end_year,
            end_month=end_month,
            end_day=end_day,
            initial_cash=initial_cash,
            resolution=resolution,
            warmup_bars=warmup_bars,
            indicators=indicators,
            trade_logic=trade_logic,
            rebalance_logic=rebalance_logic,
            description=description,
            strategy=strategy,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        # Write to file
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{class_name}.py"
        
        with open(file_path, 'w') as f:
            f.write(code)
        
        # Generate metadata
        metadata = {
            "class_name": class_name,
            "file_path": str(file_path),
            "symbols": symbols,
            "date_range": f"{start_date} to {end_date}",
            "indicators": [ind["name"] for ind in indicators],
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "line_count": len(code.splitlines())
        }
        
        return ToolResult(
            ok=True,
            data={
                "file_path": str(file_path),
                "class_name": class_name,
                "metadata": metadata,
                "preview": code.split('\n')[:20]  # First 20 lines
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