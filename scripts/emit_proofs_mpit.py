#!/usr/bin/env python3
"""Emit M-PIT proof lines for CI validation."""

from ally.tools import TOOL_REGISTRY
from ally.utils.pit_io import get_pit_fixture_summary


def main():
    """Generate M-PIT proof lines for CI."""
    
    # PROOF 1: LEAKAGE_TRIPS should be 0 (no future information leakage)
    leakage_trips = 0
    test_dates = ["2023-01-01", "2023-06-01", "2023-12-01"]
    
    try:
        for date in test_dates:
            result = TOOL_REGISTRY["pit.get_snapshot"](date=date)
            if result.ok and result.data["leakage_risk"]["is_future_date"]:
                leakage_trips += 1
    except Exception:
        leakage_trips = -1  # Error condition
    
    # PROOF 2: DELISTINGS_INCLUDED should be true (delisting returns included)
    delistings_included = False
    try:
        result = TOOL_REGISTRY["pit.apply_delistings"](
            returns_data=[{"date": "2023-07-01", "symbol": "DELISTED_CORP", "return": 0.01}],
            start_date="2023-01-01",
            end_date="2023-12-31", 
            include_delisting_returns=True
        )
        if result.ok:
            delistings_included = result.data["delisting_returns_added"] > 0
    except Exception:
        delistings_included = False
    
    # PROOF 3: PIT_SNAPSHOTS should be ok (snapshots work correctly)  
    pit_snapshots_ok = False
    try:
        result = TOOL_REGISTRY["pit.get_snapshot"](date="2023-06-01")
        if result.ok:
            pit_snapshots_ok = (
                result.data["universe_size"] > 0 and
                len(result.data["symbols"]) == result.data["universe_size"] and
                "data_hash" in result.data["meta"]
            )
    except Exception:
        pit_snapshots_ok = False
    
    # Emit proof lines in expected format
    print(f"PROOF:LEAKAGE_TRIPS: {leakage_trips}")
    print(f"PROOF:DELISTINGS_INCLUDED: {str(delistings_included).lower()}")  
    print(f"PROOF:PIT_SNAPSHOTS: {'ok' if pit_snapshots_ok else 'fail'}")


if __name__ == "__main__":
    main()