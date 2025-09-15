#!/usr/bin/env python3
"""
CI safety guards to prevent live operations in continuous integration

Ensures that CI runs remain deterministic and cannot accidentally
perform live trades, API calls, or other external operations.
"""

import os
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def assert_ci_read_only(operation_name: str = "operation"):
    """
    Assert that we're not attempting live operations in CI

    Args:
        operation_name: Description of the operation being guarded

    Raises:
        AssertionError: If attempting live operation in CI
    """
    is_ci = os.getenv("ALLY_LIVE") == "0" or os.getenv("CI") == "true"

    if is_ci:
        raise AssertionError(
            f"Live {operation_name} disabled in CI. "
            f"Set ALLY_LIVE=1 for live operations (local only)."
        )


def ci_read_only_guard(func):
    """
    Decorator to guard functions from executing in CI

    Usage:
        @ci_read_only_guard
        def trade_execution(symbol, qty):
            # This will fail in CI
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        assert_ci_read_only(func.__name__)
        return func(*args, **kwargs)

    return wrapper


def ensure_deterministic_mode():
    """
    Ensure we're running in deterministic mode for CI

    Returns:
        bool: True if in deterministic mode
    """
    ally_live = os.getenv("ALLY_LIVE", "0")
    is_deterministic = ally_live == "0"

    if not is_deterministic:
        logger.warning(
            f"Running in LIVE mode (ALLY_LIVE={ally_live}). "
            f"Set ALLY_LIVE=0 for deterministic CI behavior."
        )

    return is_deterministic


def get_safe_mode_config():
    """
    Get configuration for safe CI execution

    Returns:
        dict: Configuration dict with safe defaults
    """
    return {
        "live": False,
        "deterministic": True,
        "network_enabled": False,
        "trading_enabled": False,
        "fixture_mode": True,
        "ally_live": os.getenv("ALLY_LIVE", "0"),
        "is_ci": os.getenv("CI") == "true"
    }


# Example usage in chat controller
def safe_chat_command(command: str, live: bool = False):
    """
    Execute chat command with CI safety checks

    Args:
        command: Chat command to execute
        live: Whether this requires live data access

    Returns:
        Command result
    """
    if live:
        assert_ci_read_only(f"chat command '{command}'")

    # Execute read-only command safely
    return f"Executing '{command}' in safe mode"


# Example usage in trading execution
@ci_read_only_guard
def execute_trade(symbol: str, side: str, quantity: float):
    """
    Execute a trade (blocked in CI)

    This function will raise AssertionError if called in CI
    """
    logger.info(f"Executing {side} {quantity} shares of {symbol}")
    # Trade execution logic here
    return {"status": "executed", "symbol": symbol}


if __name__ == "__main__":
    # Test the guards
    config = get_safe_mode_config()
    print(f"Safe mode config: {config}")

    print(f"Deterministic mode: {ensure_deterministic_mode()}")

    try:
        # This should work in CI
        result = safe_chat_command("show status", live=False)
        print(f"Safe command: {result}")

        # This should fail in CI
        if not ensure_deterministic_mode():
            result = safe_chat_command("execute live trade", live=True)
            print(f"Live command: {result}")
        else:
            print("Live commands blocked in deterministic mode ✅")

    except AssertionError as e:
        print(f"CI guard triggered: {e}")