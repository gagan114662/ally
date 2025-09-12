"""
QC Universe/Data Guard - symbol availability checks and fallbacks
"""

from __future__ import annotations
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any
from ally.schemas.base import ToolResult, Meta


def _load_symbol_maps() -> Dict[str, Any]:
    """Load symbol mapping database"""
    maps_path = Path(__file__).parent.parent / "qc" / "symbol_maps.yaml"
    try:
        return yaml.safe_load(maps_path.read_text())
    except FileNotFoundError:
        return {"markets": {}, "symbol_mappings": {}, "resolution_mappings": {}}


def qc_universe_check(symbols: List[str], resolution: str = "Minute") -> ToolResult:
    """
    Check symbol availability in QC universe
    
    Args:
        symbols: List of symbols to check (e.g., ["SPY", "BTCUSDT"])
        resolution: Data resolution to check
        
    Returns:
        ToolResult with availability status
    """
    try:
        maps = _load_symbol_maps()
        markets = maps.get("markets", {})
        
        supported = []
        unsupported = []
        
        for symbol in symbols:
            found = False
            for market_key, market_data in markets.items():
                if symbol in market_data.get("symbols", []):
                    if resolution in market_data.get("resolutions", []):
                        supported.append({
                            "symbol": symbol,
                            "market": market_key,
                            "resolution": resolution
                        })
                        found = True
                        break
            
            if not found:
                unsupported.append({
                    "symbol": symbol,
                    "resolution": resolution,
                    "reason": "Symbol or resolution not available"
                })
        
        return ToolResult(
            ok=len(unsupported) == 0,
            data={
                "supported": supported,
                "unsupported": unsupported,
                "total_symbols": len(symbols),
                "success_rate": len(supported) / len(symbols) if symbols else 0
            },
            errors=[f"Unsupported: {item['symbol']}" for item in unsupported],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.universe_check"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.universe_check"})
        )


def qc_normalize_symbols(symbols: List[str]) -> ToolResult:
    """
    Normalize symbols to supported alternatives
    
    Args:
        symbols: List of symbols to normalize
        
    Returns:
        ToolResult with normalization mappings
    """
    try:
        maps = _load_symbol_maps()
        symbol_mappings = maps.get("symbol_mappings", {})
        
        normalized = []
        unchanged = []
        
        for symbol in symbols:
            if symbol in symbol_mappings:
                normalized.append({
                    "original": symbol,
                    "normalized": symbol_mappings[symbol],
                    "reason": "Mapped to supported alternative"
                })
            else:
                unchanged.append(symbol)
        
        return ToolResult(
            ok=True,
            data={
                "normalized": normalized,
                "unchanged": unchanged,
                "mapping_count": len(normalized),
                "total_symbols": len(symbols)
            },
            errors=[],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.normalize_symbols"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.normalize_symbols"})
        )


def qc_universe_guard(symbols: List[str], resolution: str = "Minute") -> ToolResult:
    """
    Comprehensive universe guard: check availability and normalize
    
    Args:
        symbols: List of symbols to validate
        resolution: Data resolution to check
        
    Returns:
        ToolResult with guard results
    """
    try:
        # First normalize symbols
        norm_result = qc_normalize_symbols(symbols)
        if not norm_result.ok:
            return norm_result
        
        # Apply normalizations
        final_symbols = []
        normalization_map = {}
        
        for symbol in symbols:
            normalized = None
            for mapping in norm_result.data.get("normalized", []):
                if mapping["original"] == symbol:
                    normalized = mapping["normalized"]
                    normalization_map[symbol] = normalized
                    break
            
            final_symbols.append(normalized if normalized else symbol)
        
        # Check availability of final symbols
        check_result = qc_universe_check(final_symbols, resolution)
        
        return ToolResult(
            ok=check_result.ok,
            data={
                "original_symbols": symbols,
                "final_symbols": final_symbols,
                "normalizations": normalization_map,
                "availability": check_result.data,
                "guard_passed": check_result.ok,
                "normalized_count": len(normalization_map)
            },
            errors=check_result.errors,
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.universe_guard"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.universe_guard"})
        )