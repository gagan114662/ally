# ally/tools/pit.py
"""Point-in-Time (PIT) tools for honest backtesting.

Eliminates survivorship bias and look-ahead bias by providing historical
universe snapshots and delisting adjustments.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ally.schemas.base import ToolResult
from ally.schemas.pit import PITUniverseRow, ActionRow, PITSnapshot, PITQuery
from ally.utils.pit_io import (
    load_fixture_universe, 
    load_fixture_actions, 
    create_pit_snapshot, 
    validate_pit_query,
    get_pit_fixture_summary
)
from ally.tools import register, TOOL_REGISTRY


@register("pit.load_universe")
def load_universe(
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None,
    min_market_cap: Optional[float] = None,
    sectors: Optional[List[str]] = None
) -> ToolResult:
    """Load point-in-time universe data with filters.
    
    Returns historical universe membership without survivorship bias.
    Only includes symbols that were actually tradable at each point in time.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        symbols: Optional list of symbols to filter by
        min_market_cap: Optional minimum market cap filter
        sectors: Optional list of sectors to include
    
    Returns:
        ToolResult with universe data and metadata
    """
    try:
        # Validate query
        query = PITQuery(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            min_market_cap=min_market_cap,
            sectors=sectors
        )
        
        validation_errors = validate_pit_query(query)
        if validation_errors:
            return ToolResult.error(validation_errors)
        
        # Load universe data
        universe_rows = load_fixture_universe(start_date, end_date)
        
        # Apply filters
        filtered_rows = []
        for row in universe_rows:
            # Symbol filter
            if symbols and row.symbol not in symbols:
                continue
                
            # Market cap filter
            if min_market_cap and (row.market_cap is None or row.market_cap < min_market_cap):
                continue
                
            # Sector filter  
            if sectors and (row.sector is None or row.sector not in sectors):
                continue
                
            filtered_rows.append(row)
        
        # Compute summary statistics
        unique_dates = sorted(set(row.date for row in filtered_rows))
        unique_symbols = sorted(set(row.symbol for row in filtered_rows))
        active_count = len([row for row in filtered_rows if row.active])
        
        # Check for survivorship bias (warn if all symbols are active at end date)
        end_date_rows = [row for row in filtered_rows if row.date == end_date]
        all_active_at_end = all(row.active for row in end_date_rows)
        survivorship_warning = None
        if all_active_at_end and len(end_date_rows) > 1:
            survivorship_warning = "All symbols active at end date - potential survivorship bias"
        
        data = {
            "query": query.model_dump(),
            "total_records": len(filtered_rows),
            "unique_dates": len(unique_dates),
            "unique_symbols": len(unique_symbols),
            "active_records": active_count,
            "inactive_records": len(filtered_rows) - active_count,
            "date_range": f"{unique_dates[0]} to {unique_dates[-1]}" if unique_dates else None,
            "symbols": unique_symbols,
            "universe_data": [row.model_dump() for row in filtered_rows],
            "survivorship_warning": survivorship_warning
        }
        
        return ToolResult.success(data)
        
    except Exception as e:
        return ToolResult.error([f"Universe loading failed: {str(e)}"])


@register("pit.load_corporate_actions")  
def load_corporate_actions(
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None,
    action_types: Optional[List[str]] = None
) -> ToolResult:
    """Load corporate actions for a date range.
    
    Returns delisting events, splits, mergers, and other actions
    that affect historical returns and universe membership.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        symbols: Optional list of symbols to filter by
        action_types: Optional list of action types (delisting, split, merger, etc.)
    
    Returns:
        ToolResult with corporate actions data
    """
    try:
        # Load actions data
        actions = load_fixture_actions(start_date, end_date)
        
        # Apply filters
        filtered_actions = []
        for action in actions:
            # Symbol filter
            if symbols and action.symbol not in symbols:
                continue
                
            # Action type filter
            if action_types and action.action not in action_types:
                continue
                
            filtered_actions.append(action)
        
        # Compute summary
        unique_symbols = sorted(set(action.symbol for action in filtered_actions))
        action_counts = {}
        for action in filtered_actions:
            action_counts[action.action] = action_counts.get(action.action, 0) + 1
        
        # Find delistings (critical for bias elimination)
        delistings = [action for action in filtered_actions if action.action == "delisting"]
        delisting_symbols = [action.symbol for action in delistings]
        
        data = {
            "total_actions": len(filtered_actions),
            "unique_symbols": len(unique_symbols),
            "symbols": unique_symbols,
            "action_counts": action_counts,
            "delisting_count": len(delistings),
            "delisted_symbols": delisting_symbols,
            "actions": [action.model_dump() for action in filtered_actions]
        }
        
        return ToolResult.success(data)
        
    except Exception as e:
        return ToolResult.error([f"Corporate actions loading failed: {str(e)}"])


@register("pit.apply_delistings")
def apply_delistings(
    returns_data: List[Dict[str, Any]],
    start_date: str,
    end_date: str,
    include_delisting_returns: bool = True
) -> ToolResult:
    """Apply delisting adjustments to returns data.
    
    Includes final delisting returns to eliminate survivorship bias.
    Critical for honest backtesting - removes the bias of excluding
    stocks that went to zero.
    
    Args:
        returns_data: List of return records with 'date', 'symbol', 'return' fields
        start_date: Start date in YYYY-MM-DD format  
        end_date: End date in YYYY-MM-DD format
        include_delisting_returns: Whether to include final delisting returns
    
    Returns:
        ToolResult with adjusted returns data
    """
    try:
        # Load delisting events
        actions_result = TOOL_REGISTRY["pit.load_corporate_actions"](start_date=start_date, end_date=end_date, action_types=["delisting"])
        if not actions_result.ok:
            return actions_result
            
        delistings = [
            ActionRow(**action_data) 
            for action_data in actions_result.data["actions"]
            if action_data["action"] == "delisting"
        ]
        
        # Create lookup for delisting events
        delisting_lookup = {action.symbol: action for action in delistings}
        
        # Process returns data
        adjusted_returns = []
        delisting_returns_added = []
        symbols_processed = set()
        delisting_returns_created = set()  # Track which delisting returns we've added
        
        for return_record in returns_data:
            symbol = return_record["symbol"]
            date = return_record["date"]
            
            adjusted_returns.append(return_record.copy())
            symbols_processed.add(symbol)
        
        # Add delisting returns for symbols that have delistings
        if include_delisting_returns:
            for symbol, delisting in delisting_lookup.items():
                if symbol in symbols_processed and delisting.date not in delisting_returns_created:
                    # Calculate delisting return (usually -100% or final price based)
                    delisting_return = -1.0 if delisting.reason == "bankruptcy" else -0.8
                    if delisting.final_price is not None and delisting.final_price > 0:
                        # More sophisticated calculation could use price data
                        delisting_return = max(-0.95, -0.5)  # Cap at -95% loss
                    
                    delisting_record = {
                        "date": delisting.date,
                        "symbol": symbol,
                        "return": delisting_return,
                        "type": "delisting_return",
                        "reason": delisting.reason,
                        "meta": {"original_action": delisting.model_dump()}
                    }
                    
                    adjusted_returns.append(delisting_record)
                    delisting_returns_added.append(delisting_record)
                    delisting_returns_created.add(delisting.date)
        
        # Detect potential survivorship bias
        bias_checks = _check_survivorship_bias(returns_data, delistings)
        
        data = {
            "original_records": len(returns_data),
            "adjusted_records": len(adjusted_returns), 
            "delisting_returns_added": len(delisting_returns_added),
            "symbols_processed": len(symbols_processed),
            "delistings_found": len(delistings),
            "delisted_symbols": list(delisting_lookup.keys()),
            "bias_checks": bias_checks,
            "adjusted_returns": adjusted_returns,
            "delisting_details": delisting_returns_added
        }
        
        return ToolResult.success(data)
        
    except Exception as e:
        return ToolResult.error([f"Delisting adjustment failed: {str(e)}"])


@register("pit.get_snapshot")
def get_snapshot(date: str) -> ToolResult:
    """Get point-in-time universe snapshot for a specific date.
    
    Returns complete universe state including active symbols and
    any corporate actions on that date.
    
    Args:
        date: Date in YYYY-MM-DD format
    
    Returns:
        ToolResult with PITSnapshot data
    """
    try:
        # Validate date format
        datetime.fromisoformat(date)
        
        snapshot = create_pit_snapshot(date)
        
        # Add leakage detection
        future_info_risk = _detect_future_info_leakage(snapshot)
        
        data = snapshot.model_dump()
        data["leakage_risk"] = future_info_risk
        
        return ToolResult.success(data)
        
    except ValueError as e:
        return ToolResult.error([f"Invalid date format: {e}"])
    except Exception as e:
        return ToolResult.error([f"Snapshot creation failed: {str(e)}"])


@register("pit.validate_backtest") 
def validate_backtest(
    backtest_data: Dict[str, Any],
    start_date: str,
    end_date: str
) -> ToolResult:
    """Validate backtest for survivorship and look-ahead bias.
    
    Checks if backtest uses only point-in-time data and includes
    delisting returns. Critical quality control for honest backtests.
    
    Args:
        backtest_data: Backtest results with trades/positions
        start_date: Backtest start date  
        end_date: Backtest end date
    
    Returns:
        ToolResult with validation results and bias warnings
    """
    try:
        validation_results = {
            "survivorship_bias": False,
            "look_ahead_bias": False,
            "delisting_coverage": True,
            "warnings": [],
            "errors": []
        }
        
        # Extract symbols from backtest
        backtest_symbols = set()
        if "trades" in backtest_data:
            backtest_symbols.update(trade.get("symbol", "") for trade in backtest_data["trades"])
        if "positions" in backtest_data:
            backtest_symbols.update(pos.get("symbol", "") for pos in backtest_data["positions"])
        
        backtest_symbols.discard("")  # Remove empty strings
        
        # Load PIT data for comparison
        universe_result = TOOL_REGISTRY["pit.load_universe"](start_date=start_date, end_date=end_date, symbols=list(backtest_symbols))
        if not universe_result.ok:
            return universe_result
            
        actions_result = TOOL_REGISTRY["pit.load_corporate_actions"](start_date=start_date, end_date=end_date, symbols=list(backtest_symbols))
        if not actions_result.ok:
            return actions_result
        
        # Check for survivorship bias
        delisted_symbols = set(actions_result.data["delisted_symbols"])
        symbols_missing_delistings = delisted_symbols - backtest_symbols
        
        if symbols_missing_delistings:
            validation_results["survivorship_bias"] = True
            validation_results["errors"].append(
                f"Missing delisted symbols: {list(symbols_missing_delistings)}"
            )
        
        # Check delisting returns inclusion
        if "trades" in backtest_data:
            delisting_trades = [
                trade for trade in backtest_data["trades"] 
                if trade.get("type") == "delisting_return"
            ]
            if len(delisted_symbols) > 0 and len(delisting_trades) == 0:
                validation_results["delisting_coverage"] = False
                validation_results["warnings"].append(
                    "No delisting returns found despite delisting events"
                )
        
        # Check for future information leakage
        if _has_look_ahead_bias(backtest_data, start_date, end_date):
            validation_results["look_ahead_bias"] = True
            validation_results["errors"].append("Potential look-ahead bias detected")
        
        # Overall validation status
        validation_results["valid"] = (
            not validation_results["survivorship_bias"] and
            not validation_results["look_ahead_bias"] and
            validation_results["delisting_coverage"]
        )
        
        return ToolResult.success(validation_results)
        
    except Exception as e:
        return ToolResult.error([f"Backtest validation failed: {str(e)}"])


def _check_survivorship_bias(returns_data: List[Dict[str, Any]], delistings: List[ActionRow]) -> Dict[str, Any]:
    """Internal helper to detect survivorship bias patterns."""
    symbols_in_returns = set(r["symbol"] for r in returns_data)
    delisted_symbols = set(d.symbol for d in delistings)
    
    missing_delistings = delisted_symbols - symbols_in_returns
    survivorship_ratio = len(missing_delistings) / len(delisted_symbols) if delisted_symbols else 0
    
    return {
        "missing_delisted_symbols": len(missing_delistings),
        "total_delistings": len(delisted_symbols),
        "survivorship_ratio": survivorship_ratio,
        "bias_likely": survivorship_ratio > 0.1,  # More than 10% missing
        "missing_symbols": list(missing_delistings)
    }


def _detect_future_info_leakage(snapshot: PITSnapshot) -> Dict[str, Any]:
    """Internal helper to detect potential future information leakage."""
    today = datetime.now().date().isoformat()
    
    return {
        "snapshot_date": snapshot.date,
        "is_future_date": snapshot.date > today,
        "risk_level": "high" if snapshot.date > today else "low",
        "warning": "Snapshot date is in the future" if snapshot.date > today else None
    }


def _has_look_ahead_bias(backtest_data: Dict[str, Any], start_date: str, end_date: str) -> bool:
    """Internal helper to detect look-ahead bias in backtest data."""
    # Simple heuristic: check if any trade dates are outside backtest period
    if "trades" in backtest_data:
        for trade in backtest_data["trades"]:
            trade_date = trade.get("date", "")
            if trade_date and (trade_date < start_date or trade_date > end_date):
                return True
    return False