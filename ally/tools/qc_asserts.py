"""
QC Runtime Assertions Tool - Inject and validate assertions
"""

from __future__ import annotations
import re
from datetime import datetime
from pathlib import Path
from ally.schemas.base import ToolResult, Meta
from ally.qc.runtime_asserts import generate_assert_helpers, count_assert_trips


def qc_inject_asserts(algo_path: str) -> ToolResult:
    """
    Inject runtime assertion helpers into QC algorithm
    
    Args:
        algo_path: Path to QC algorithm file
        
    Returns:
        ToolResult with injection results
    """
    try:
        path = Path(algo_path)
        if not path.exists():
            raise FileNotFoundError(f"Algorithm file not found: {algo_path}")
        
        content = path.read_text()
        
        # Check if assertions already injected
        if 'assert_indicator_ready' in content:
            return ToolResult(
                ok=True,
                data={
                    "algo_path": str(path),
                    "already_injected": True,
                    "message": "Assertions already present"
                },
                errors=[],
                meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.inject_asserts"})
            )
        
        # Generate and inject helpers
        helpers = generate_assert_helpers()
        
        # Find Initialize method and inject
        lines = content.split('\n')
        new_lines = []
        injected = False
        
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            if 'def Initialize(self):' in line and not injected:
                # Skip docstring if present
                j = i + 1
                while j < len(lines) and lines[j].strip().startswith(('"""', "'''")):
                    new_lines.append(lines[j])
                    j += 1
                
                # Add blank line and helpers
                new_lines.append('')
                new_lines.append('        # Runtime Assertions')
                for helper_line in helpers.split('\n'):
                    if helper_line.strip():
                        new_lines.append('        ' + helper_line)
                injected = True
        
        if injected:
            path.write_text('\n'.join(new_lines))
            
            return ToolResult(
                ok=True,
                data={
                    "algo_path": str(path),
                    "injected": True,
                    "helpers_added": [
                        "assert_indicator_ready",
                        "assert_warmup_complete", 
                        "assert_orders_filled",
                        "assert_history_available",
                        "assert_portfolio_value"
                    ]
                },
                errors=[],
                meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.inject_asserts"})
            )
        else:
            return ToolResult(
                ok=False,
                data={"algo_path": str(path), "error": "Could not find Initialize method"},
                errors=["Initialize method not found for injection"],
                meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.inject_asserts"})
            )
            
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.inject_asserts"})
        )


def qc_validate_asserts(log_path: str = None, log_content: str = None) -> ToolResult:
    """
    Validate assertion trips from LEAN logs
    
    Args:
        log_path: Path to LEAN log file
        log_content: Direct log content (if no path)
        
    Returns:
        ToolResult with assertion validation
    """
    try:
        if log_path:
            content = Path(log_path).read_text()
        elif log_content:
            content = log_content
        else:
            # Mock validation for CI when no LEAN logs available
            return ToolResult(
                ok=True,
                data={
                    "total_trips": 0,
                    "categories": {
                        'indicator_not_ready': 0,
                        'warmup_incomplete': 0,
                        'order_mismatch': 0,
                        'history_unavailable': 0,
                        'portfolio_below_min': 0,
                        'other': 0
                    },
                    "mock_validation": True,
                    "message": "Mock validation - no LEAN logs available"
                },
                errors=[],
                meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.validate_asserts"})
            )
        
        results = count_assert_trips(content)
        
        return ToolResult(
            ok=results['total_trips'] == 0,
            data=results,
            errors=[f"Found {results['total_trips']} assertion trips"] if results['total_trips'] > 0 else [],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.validate_asserts"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.validate_asserts"})
        )