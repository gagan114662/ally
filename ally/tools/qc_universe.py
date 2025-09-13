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


def canonical_equity(ticker: str) -> str:
    """
    Normalize equity ticker to QC canonical form
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Canonical ticker form
    """
    t = ticker.upper().replace(' ', '')
    
    # Punctuation-agnostic match for alias lookup
    no_punct = t.replace('.', '')
    
    if no_punct in {'BRKA', 'BRK.A'}:
        return 'BRK.A'
    if no_punct in {'BRKB', 'BRK.B'}:
        return 'BRK.B'
    if t in {'GOOG', 'GOOGL'}:
        return t  # Don't force remap - both are valid share classes
    if t in {'FB', 'META'}:
        return 'META'  # Post-rename canonical
    
    return t


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
            # First try exact mapping
            if symbol in symbol_mappings:
                normalized.append({
                    "original": symbol,
                    "normalized": symbol_mappings[symbol],
                    "reason": "Mapped to supported alternative"
                })
            else:
                # Try canonical equity normalization
                canonical = canonical_equity(symbol)
                if canonical != symbol:
                    normalized.append({
                        "original": symbol,
                        "normalized": canonical,
                        "reason": "Canonical equity form"
                    })
                else:
                    unchanged.append(symbol)
        
        # Count total aliases in database
        alias_count = len(symbol_mappings)
        
        return ToolResult(
            ok=True,
            data={
                "normalized": normalized,
                "unchanged": unchanged,
                "mapping_count": len(normalized),
                "total_symbols": len(symbols),
                "alias_count": alias_count
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


def qc_history_smoke(symbols: List[str], resolution: str = "Daily") -> ToolResult:
    """
    Smoke test history data availability for symbols
    
    Args:
        symbols: List of symbols to test
        resolution: Data resolution to test
        
    Returns:
        ToolResult with history smoke results
    """
    try:
        # For now, simulate history checks since we don't have LEAN running
        # In real implementation, this would use LEAN's History() method
        smoke_results = []
        
        for symbol in symbols:
            # Mock successful history fetch for known good symbols
            if symbol in ["SPY", "QQQ", "AAPL", "MSFT", "BTCUSD", "ETHUSD", "BNBUSDT"]:
                smoke_results.append({
                    "symbol": symbol,
                    "resolution": resolution,
                    "history_available": True,
                    "bars_count": 1  # Mock 1 bar
                })
            else:
                smoke_results.append({
                    "symbol": symbol,
                    "resolution": resolution,  
                    "history_available": False,
                    "bars_count": 0
                })
        
        success_count = sum(1 for r in smoke_results if r["history_available"])
        
        return ToolResult(
            ok=success_count == len(symbols),
            data={
                "smoke_results": smoke_results,
                "success_count": success_count,
                "total_symbols": len(symbols),
                "success_rate": success_count / len(symbols) if symbols else 0
            },
            errors=[f"History unavailable: {r['symbol']}" for r in smoke_results if not r["history_available"]],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.history_smoke"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.history_smoke"})
        )


def qc_resolution_matrix(symbols: List[str]) -> ToolResult:
    """
    Generate resolution availability matrix for symbols
    
    Args:
        symbols: List of symbols to check
        
    Returns:
        ToolResult with resolution matrix
    """
    try:
        maps = _load_symbol_maps()
        markets = maps.get("markets", {})
        
        matrix = {}
        
        for symbol in symbols:
            symbol_resolutions = {}
            
            # Find which market supports this symbol
            for market_key, market_data in markets.items():
                if symbol in market_data.get("symbols", []):
                    available_resolutions = market_data.get("resolutions", [])
                    for res in ["Second", "Minute", "Hour", "Daily"]:
                        symbol_resolutions[res] = res in available_resolutions
                    break
            
            # If not found in any market, assume no resolutions available
            if not symbol_resolutions:
                symbol_resolutions = {res: False for res in ["Second", "Minute", "Hour", "Daily"]}
            
            matrix[symbol] = symbol_resolutions
        
        return ToolResult(
            ok=True,
            data={
                "resolution_matrix": matrix,
                "symbols_count": len(symbols)
            },
            errors=[],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.resolution_matrix"})
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0, provenance={"tool_name": "qc.resolution_matrix"})
        )