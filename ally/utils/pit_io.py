# ally/utils/pit_io.py
"""Point-in-Time I/O utilities for loading historical universe and action data.

Provides deterministic fixture loading for CI testing without network dependencies.
All data is loaded from local fixtures to ensure reproducible results.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
import hashlib

from ally.schemas.pit import PITUniverseRow, ActionRow, PITSnapshot, PITQuery


# Default fixture data for deterministic testing
FIXTURE_UNIVERSE = [
    # 2023-01-01: Initial universe
    {"date": "2023-01-01", "symbol": "AAPL", "active": True, "market_cap": 2800000000000, "sector": "Technology", "exchange": "NASDAQ"},
    {"date": "2023-01-01", "symbol": "MSFT", "active": True, "market_cap": 2200000000000, "sector": "Technology", "exchange": "NASDAQ"},
    {"date": "2023-01-01", "symbol": "GOOGL", "active": True, "market_cap": 1400000000000, "sector": "Technology", "exchange": "NASDAQ"},
    {"date": "2023-01-01", "symbol": "TSLA", "active": True, "market_cap": 380000000000, "sector": "Consumer Discretionary", "exchange": "NASDAQ"},
    {"date": "2023-01-01", "symbol": "DELISTED_CORP", "active": True, "market_cap": 50000000000, "sector": "Energy", "exchange": "NYSE"},
    
    # 2023-06-01: Mid-year snapshot (DELISTED_CORP still active)
    {"date": "2023-06-01", "symbol": "AAPL", "active": True, "market_cap": 3000000000000, "sector": "Technology", "exchange": "NASDAQ"},
    {"date": "2023-06-01", "symbol": "MSFT", "active": True, "market_cap": 2400000000000, "sector": "Technology", "exchange": "NASDAQ"},
    {"date": "2023-06-01", "symbol": "GOOGL", "active": True, "market_cap": 1500000000000, "sector": "Technology", "exchange": "NASDAQ"},
    {"date": "2023-06-01", "symbol": "TSLA", "active": True, "market_cap": 420000000000, "sector": "Consumer Discretionary", "exchange": "NASDAQ"},
    {"date": "2023-06-01", "symbol": "DELISTED_CORP", "active": True, "market_cap": 45000000000, "sector": "Energy", "exchange": "NYSE"},
    
    # 2023-12-01: Year-end snapshot (DELISTED_CORP removed)
    {"date": "2023-12-01", "symbol": "AAPL", "active": True, "market_cap": 3100000000000, "sector": "Technology", "exchange": "NASDAQ"},
    {"date": "2023-12-01", "symbol": "MSFT", "active": True, "market_cap": 2600000000000, "sector": "Technology", "exchange": "NASDAQ"},
    {"date": "2023-12-01", "symbol": "GOOGL", "active": True, "market_cap": 1600000000000, "sector": "Technology", "exchange": "NASDAQ"},
    {"date": "2023-12-01", "symbol": "TSLA", "active": True, "market_cap": 450000000000, "sector": "Consumer Discretionary", "exchange": "NASDAQ"},
    {"date": "2023-12-01", "symbol": "DELISTED_CORP", "active": False, "market_cap": 0, "sector": "Energy", "exchange": "NYSE"},
]

FIXTURE_ACTIONS = [
    # Delisting event for DELISTED_CORP
    {
        "date": "2023-08-15", 
        "symbol": "DELISTED_CORP", 
        "action": "delisting",
        "adjustment_factor": 1.0,
        "final_price": 12.45,
        "reason": "bankruptcy",
        "successor_symbol": None,
        "meta": {"filing_date": "2023-07-20", "liquidation": True}
    },
    # Stock split for TSLA
    {
        "date": "2023-03-15",
        "symbol": "TSLA", 
        "action": "split",
        "adjustment_factor": 3.0,
        "final_price": None,
        "reason": "stock_split_3_for_1",
        "successor_symbol": None,
        "meta": {"split_ratio": "3:1", "ex_date": "2023-03-16"}
    }
]


def load_fixture_universe(
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
    fixture_data: Optional[List[Dict[str, Any]]] = None
) -> List[PITUniverseRow]:
    """Load point-in-time universe data from fixtures.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        fixture_data: Optional custom fixture data (uses default if None)
    
    Returns:
        List of PITUniverseRow objects within date range
    """
    data = fixture_data if fixture_data is not None else FIXTURE_UNIVERSE
    
    # Filter by date range
    filtered_data = [
        row for row in data 
        if start_date <= row["date"] <= end_date
    ]
    
    return [PITUniverseRow(**row) for row in filtered_data]


def load_fixture_actions(
    start_date: str = "2023-01-01", 
    end_date: str = "2023-12-31",
    fixture_data: Optional[List[Dict[str, Any]]] = None
) -> List[ActionRow]:
    """Load corporate actions from fixtures.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        fixture_data: Optional custom fixture data (uses default if None)
    
    Returns:
        List of ActionRow objects within date range
    """
    data = fixture_data if fixture_data is not None else FIXTURE_ACTIONS
    
    # Filter by date range
    filtered_data = [
        row for row in data
        if start_date <= row["date"] <= end_date
    ]
    
    return [ActionRow(**row) for row in filtered_data]


def create_pit_snapshot(date: str, fixture_data: Optional[Dict[str, Any]] = None) -> PITSnapshot:
    """Create a point-in-time snapshot for a specific date.
    
    Args:
        date: Date in YYYY-MM-DD format
        fixture_data: Optional fixture override
        
    Returns:
        PITSnapshot containing active symbols and actions for the date
    """
    # Load universe data up to this date to find latest status
    universe_data = load_fixture_universe(end_date=date, fixture_data=fixture_data.get("universe") if fixture_data else None)
    
    # Find the most recent status for each symbol
    symbol_status: Dict[str, bool] = {}
    for row in universe_data:
        if row.date <= date:
            symbol_status[row.symbol] = row.active
    
    # Get active symbols
    active_symbols = [symbol for symbol, active in symbol_status.items() if active]
    
    # Load actions for this specific date
    actions_data = load_fixture_actions(start_date=date, end_date=date, fixture_data=fixture_data.get("actions") if fixture_data else None)
    
    return PITSnapshot(
        date=date,
        symbols=sorted(active_symbols),  # Sort for deterministic output
        universe_size=len(active_symbols),
        actions=actions_data,
        meta={
            "created_at": datetime.now().isoformat(),
            "data_hash": _compute_snapshot_hash(active_symbols, actions_data)
        }
    )


def validate_pit_query(query: PITQuery) -> List[str]:
    """Validate PIT query parameters and return any errors.
    
    Args:
        query: PITQuery object to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    try:
        start_dt = datetime.fromisoformat(query.start_date)
        end_dt = datetime.fromisoformat(query.end_date)
        
        if start_dt > end_dt:
            errors.append("start_date must be <= end_date")
            
        # Check for reasonable date ranges (not too far in past/future)
        min_date = datetime(2020, 1, 1)
        max_date = datetime.now() + timedelta(days=30)
        
        if start_dt < min_date:
            errors.append(f"start_date must be >= {min_date.date()}")
        if end_dt > max_date:
            errors.append(f"end_date must be <= {max_date.date()}")
            
    except ValueError as e:
        errors.append(f"Invalid date format: {e}")
    
    if query.min_market_cap is not None and query.min_market_cap < 0:
        errors.append("min_market_cap must be >= 0")
        
    return errors


def _compute_snapshot_hash(symbols: List[str], actions: List[ActionRow]) -> str:
    """Compute deterministic hash for snapshot verification."""
    content = {
        "symbols": sorted(symbols),
        "actions": [action.model_dump() for action in sorted(actions, key=lambda x: x.symbol)]
    }
    content_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(content_str.encode()).hexdigest()[:16]


def get_pit_fixture_summary() -> Dict[str, Any]:
    """Get summary of available fixture data for testing/validation."""
    universe_data = load_fixture_universe()
    actions_data = load_fixture_actions()
    
    # Extract date ranges and symbols
    universe_dates = sorted(set(row.date for row in universe_data))
    action_dates = sorted(set(row.date for row in actions_data))
    all_symbols = sorted(set(row.symbol for row in universe_data))
    
    # Count active vs inactive periods
    active_periods = len([row for row in universe_data if row.active])
    inactive_periods = len([row for row in universe_data if not row.active])
    
    return {
        "universe_records": len(universe_data),
        "action_records": len(actions_data),
        "date_range": f"{universe_dates[0]} to {universe_dates[-1]}" if universe_dates else "None",
        "symbols": all_symbols,
        "symbol_count": len(all_symbols),
        "active_periods": active_periods,
        "inactive_periods": inactive_periods,
        "action_types": sorted(set(row.action for row in actions_data)),
        "delisting_count": len([row for row in actions_data if row.action == "delisting"]),
        "fixture_hash": _compute_snapshot_hash(all_symbols, actions_data)
    }