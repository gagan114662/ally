"""
Gating utilities to control live vs offline mode
Implements the double-gate: live=True AND ALLY_LIVE=1 required for network access
"""

import os
from typing import Optional


class LiveModeError(Exception):
    """Raised when live mode requirements are not met"""
    pass


def check_live_mode_allowed(live: bool = False, api_key: Optional[str] = None, 
                           service_name: str = "service") -> None:
    """
    Check if live mode is allowed for network operations
    
    Args:
        live: Whether live mode is requested by the tool
        api_key: API key for the service (if None, will be checked via env)
        service_name: Name of the service for error messages
    
    Raises:
        LiveModeError: If live mode requirements are not met
    """
    if not live:
        # Offline mode - always allowed
        return
    
    # Live mode requested - check double gate
    ally_live = os.getenv("ALLY_LIVE", "0")
    
    if ally_live != "1":
        raise LiveModeError(
            f"Live mode requested (live=True) but ALLY_LIVE={ally_live}. "
            f"Set ALLY_LIVE=1 to enable network access for {service_name}."
        )
    
    if api_key is None:
        raise LiveModeError(
            f"Live mode enabled but no API key provided for {service_name}. "
            f"Set the appropriate API key environment variable."
        )
    
    if not api_key or api_key.startswith("your_") or api_key == "":
        raise LiveModeError(
            f"Live mode enabled but invalid API key for {service_name}. "
            f"API key appears to be a placeholder or empty."
        )


def is_live_mode_enabled() -> bool:
    """
    Check if live mode is enabled via ALLY_LIVE environment variable
    
    Returns:
        True if ALLY_LIVE=1, False otherwise
    """
    return os.getenv("ALLY_LIVE", "0") == "1"


def require_offline_mode(operation_name: str = "operation") -> None:
    """
    Ensure we are in offline mode - raise error if ALLY_LIVE=1
    Used for operations that should never run in CI
    
    Args:
        operation_name: Name of the operation for error messages
    
    Raises:
        LiveModeError: If ALLY_LIVE=1
    """
    if is_live_mode_enabled():
        raise LiveModeError(
            f"Operation '{operation_name}' is not allowed in live mode (ALLY_LIVE=1). "
            f"Set ALLY_LIVE=0 or unset ALLY_LIVE to run offline operations."
        )


def get_live_mode_status() -> dict:
    """
    Get current live mode status for debugging
    
    Returns:
        Dictionary with live mode status information
    """
    ally_live = os.getenv("ALLY_LIVE", "0")
    return {
        "ally_live_env": ally_live,
        "is_live_enabled": ally_live == "1",
        "is_offline_mode": ally_live != "1"
    }