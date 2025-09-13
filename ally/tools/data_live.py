"""
Live data tools for M-RealData Gate system
Provides receipt-backed, anti-fabrication data fetching
"""

from __future__ import annotations
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from ally.schemas.base import ToolResult, Meta
from ally.schemas.receipts import Receipt, QuorumVerdict, LiveAccessError, BudgetExceededError, QuorumFailureError, LiveDataSession
from ally.tools import register
from ally.utils.receipts import write_payload_and_receipt, quorum_check, enforce_live_mode_or_die
from ally.utils.providers import PROVIDER_REGISTRY


# Session tracking for budget enforcement
_current_session: Optional[LiveDataSession] = None


def _get_or_create_session(budget_cents: int) -> LiveDataSession:
    """Get current session or create new one"""
    global _current_session
    
    if _current_session is None:
        _current_session = LiveDataSession(
            session_id=f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            started_at=datetime.utcnow().isoformat() + "Z",
            budget_cents=budget_cents
        )
    
    return _current_session


def _check_budget(session: LiveDataSession, cost_cents: int) -> None:
    """Check if cost would exceed budget"""
    if session.spent_cents + cost_cents > session.budget_cents:
        raise BudgetExceededError(
            f"Cost {cost_cents}¢ would exceed budget {session.budget_cents}¢ "
            f"(already spent {session.spent_cents}¢)"
        )


@register("data.live_fetch")
def live_fetch(
    vendor: str,
    endpoint: str, 
    params: Dict[str, Any],
    live: bool = False,
    quorum: Optional[Dict[str, Any]] = None,
    budget_cents: int = 100
) -> ToolResult:
    """
    Fetch live data with receipt attestation and optional quorum verification
    
    Args:
        vendor: Provider name (polygon, alphavantage, etc.)
        endpoint: API endpoint to call
        params: Request parameters
        live: Enable live fetching (requires ALLY_LIVE=1)
        quorum: Optional cross-provider verification config
        budget_cents: Maximum cost allowed for this session
    
    Returns:
        ToolResult with data and receipt, or live_denied if gates not met
    """
    try:
        # Enforce double gate
        enforce_live_mode_or_die("data.live_fetch", live)
        
        # Get session and check budget
        session = _get_or_create_session(budget_cents)
        provider_config = PROVIDER_REGISTRY.get_provider(vendor)
        _check_budget(session, provider_config.cost_per_call_cents)
        
        if quorum:
            # Multi-vendor quorum check
            return _fetch_with_quorum(endpoint, params, quorum, session)
        else:
            # Single vendor fetch
            return _fetch_single_vendor(vendor, endpoint, params, session)
            
    except LiveAccessError:
        # Live access denied - return controlled response
        return ToolResult(
            ok=False,
            data={"live_denied": True, "reason": "live=False or ALLY_LIVE!=1"},
            errors=["Live data access denied"],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0)
        )
    except (BudgetExceededError, QuorumFailureError) as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0)
        )
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": f"Fetch failed: {e}"},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0)
        )


@register("data.live_history")
def live_history(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    vendor: str = "polygon",
    live: bool = False,
    budget_cents: int = 100,
    quorum: Optional[Dict[str, Any]] = None
) -> ToolResult:
    """
    Fetch historical data with receipt tracking
    
    Args:
        symbol: Stock/crypto symbol (AAPL, BTC-USD)
        start: Start date (ISO format)
        end: End date (ISO format) 
        interval: Data interval (1d, 1h, etc.)
        vendor: Primary data vendor
        live: Enable live fetching
        budget_cents: Maximum session cost
        quorum: Optional cross-provider verification
    
    Returns:
        ToolResult with OHLCV data and receipt_ids
    """
    try:
        enforce_live_mode_or_die("data.live_history", live)
        
        # Build endpoint and params based on vendor
        if vendor == "polygon":
            endpoint = f"/v2/aggs/ticker/{symbol}/range/1/{interval}/{start}/{end}"
            params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        elif vendor == "alphavantage":
            endpoint = "/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED" if interval == "1d" else "TIME_SERIES_INTRADAY",
                "symbol": symbol,
                "outputsize": "full",
                "datatype": "json"
            }
            if interval != "1d":
                params["interval"] = interval
        else:
            raise ValueError(f"Unsupported vendor for history: {vendor}")
        
        return live_fetch(
            vendor=vendor,
            endpoint=endpoint,
            params=params,
            live=live,
            quorum=quorum,
            budget_cents=budget_cents
        )
        
    except Exception as e:
        return ToolResult(
            ok=False,
            data={"error": str(e)},
            errors=[str(e)],
            meta=Meta(ts=datetime.utcnow(), duration_ms=0)
        )


def _fetch_single_vendor(vendor: str, endpoint: str, params: Dict[str, Any], 
                        session: LiveDataSession) -> ToolResult:
    """Fetch from single vendor with receipt"""
    provider_config = PROVIDER_REGISTRY.get_provider(vendor)
    
    # Fetch data
    payload_bytes = PROVIDER_REGISTRY.fetch_with_auth(vendor, endpoint, params)
    
    # Write receipt
    receipt = write_payload_and_receipt(
        vendor=vendor,
        endpoint=endpoint, 
        params=params,
        payload_bytes=payload_bytes,
        cost_cents=provider_config.cost_per_call_cents
    )
    
    # Update session
    session.spent_cents += provider_config.cost_per_call_cents
    session.receipts.append(receipt.content_sha1)
    
    # Parse JSON response
    try:
        response_json = json.loads(payload_bytes.decode('utf-8'))
    except Exception:
        response_json = {"raw_bytes": len(payload_bytes)}
    
    return ToolResult(
        ok=True,
        data={
            "receipt": receipt.model_dump(),
            "json": response_json,
            "session": session.model_dump()
        },
        errors=[],
        meta=Meta(ts=datetime.utcnow(), duration_ms=0)
    )


def _fetch_with_quorum(endpoint: str, params: Dict[str, Any], quorum_config: Dict[str, Any],
                      session: LiveDataSession) -> ToolResult:
    """Fetch from multiple vendors and verify agreement"""
    vendors = quorum_config["vendors"]
    metric = quorum_config["metric"] 
    tolerance_bps = quorum_config["tolerance_bps"]
    
    receipts = []
    measurements = []
    responses = []
    
    # Fetch from each vendor
    for vendor in vendors:
        try:
            result = _fetch_single_vendor(vendor, endpoint, params, session)
            if result.ok:
                receipts.append(result.data["receipt"])
                responses.append(result.data["json"])
                
                # Extract measurement for quorum check
                measurement = _extract_measurement(result.data["json"], metric, vendor)
                measurements.append(measurement)
                
        except Exception as e:
            # Continue with other vendors if one fails
            continue
    
    if len(measurements) < len(vendors):
        raise RuntimeError(f"Failed to fetch from all vendors: got {len(measurements)}/{len(vendors)}")
    
    # Verify quorum
    verdict = quorum_check(measurements, tolerance_bps, vendors, metric)
    
    if not verdict.ok:
        raise QuorumFailureError(
            f"Quorum failed: {metric} variance {verdict.variance_bps:.2f} bps "
            f"exceeds tolerance {tolerance_bps} bps"
        )
    
    return ToolResult(
        ok=True,
        data={
            "receipts": receipts,
            "responses": responses,
            "quorum_verdict": verdict.model_dump(),
            "session": session.model_dump()
        },
        errors=[],
        meta=Meta(ts=datetime.utcnow(), duration_ms=0)
    )


def _extract_measurement(response: Dict[str, Any], metric: str, vendor: str) -> float:
    """Extract measurement from vendor response for quorum check"""
    try:
        if vendor == "polygon":
            # Polygon format: {"results": [{"c": close_price, ...}]}
            if "results" in response and response["results"]:
                last_bar = response["results"][-1]
                if metric == "close":
                    return float(last_bar.get("c", 0))
                elif metric == "volume":
                    return float(last_bar.get("v", 0))
                    
        elif vendor == "alphavantage":
            # AlphaVantage format: {"Time Series (Daily)": {"2023-01-01": {"4. close": "150.00"}}}
            time_series_key = None
            for key in response.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break
                    
            if time_series_key:
                time_series = response[time_series_key]
                latest_date = max(time_series.keys())
                latest_data = time_series[latest_date]
                
                if metric == "close":
                    return float(latest_data.get("4. close", 0))
                elif metric == "volume":
                    return float(latest_data.get("6. volume", 0))
        
        # Default fallback
        return 0.0
        
    except Exception:
        return 0.0