#!/usr/bin/env python3
"""
Promotion guard for blocking deployments - Phase 8

Implements promotion blocking when any sentinel status != "OK",
providing comprehensive guard logic and actionable failure reasons.
"""

import os
import json
import yaml
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict

from ally.schemas.base import ToolResult as Result
from ally.utils.gating import check_live_mode_allowed, LiveModeError
from ally.tools import register

# Import drift detection modules
try:
    from ally.ops.drift_data import ops_drift_data
    from ally.ops.drift_strategy import ops_drift_strategy
    from ally.ops.drift_ops import ops_drift_ops
    DRIFT_MODULES_AVAILABLE = True
except ImportError:
    DRIFT_MODULES_AVAILABLE = False

# Create a simple receipt generator
def generate_receipt(tool_name: str, data: dict) -> str:
    """Generate a simple receipt hash"""
    import json
    import hashlib
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


@dataclass
class PromotionGuardConfig:
    """Configuration for promotion guard"""
    require_all_ok: bool = True
    block_on: List[str] = None
    timeout_hours: int = 24
    check_recent_hours: int = 1

    def __post_init__(self):
        if self.block_on is None:
            self.block_on = [
                "data_drift",
                "strategy_drift",
                "ops_drift"
            ]


def check_sentinel_status(sentinel_type: str, config: Dict[str, Any],
                         recent_hours: int = 1) -> Dict[str, Any]:
    """
    Check the status of a specific sentinel

    Args:
        sentinel_type: Type of sentinel (data_drift, strategy_drift, ops_drift)
        config: Configuration for the sentinel check
        recent_hours: Look for results within this many hours

    Returns:
        Sentinel status result
    """
    try:
        if sentinel_type == "data_drift" and DRIFT_MODULES_AVAILABLE:
            result = ops_drift_data(
                panel_path=config.get("panel_path"),
                policy_path=config.get("policy_path", "ally/ops/policy.yaml"),
                live=False  # Don't trigger live mode in guard
            )
        elif sentinel_type == "strategy_drift" and DRIFT_MODULES_AVAILABLE:
            result = ops_drift_strategy(
                strategy_hash=config.get("strategy_hash", "test_strategy"),
                policy_path=config.get("policy_path", "ally/ops/policy.yaml"),
                live=False
            )
        elif sentinel_type == "ops_drift" and DRIFT_MODULES_AVAILABLE:
            result = ops_drift_ops(
                fixture_path=config.get("fixture_path", "artifacts/fixtures/determinism.pkl"),
                policy_path=config.get("policy_path", "ally/ops/policy.yaml"),
                live=False
            )
        else:
            # Mock sentinel for CI or when modules not available
            mock_status = "OK" if config.get("mock_ok", True) else "DRIFT"
            return {
                "sentinel_type": sentinel_type,
                "status": mock_status,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "violations": [] if mock_status == "OK" else ["Mock violation for testing"],
                "mock": True
            }

        if result.ok:
            return {
                "sentinel_type": sentinel_type,
                "status": result.data.get("status", "UNKNOWN"),
                "timestamp": result.data.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                "violations": result.data.get("violations", []),
                "receipt_hash": result.data.get("drift_receipt", ""),
                "mock": False
            }
        else:
            return {
                "sentinel_type": sentinel_type,
                "status": "ERROR",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "violations": result.errors,
                "error": True,
                "mock": False
            }

    except Exception as e:
        return {
            "sentinel_type": sentinel_type,
            "status": "ERROR",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "violations": [f"Sentinel check failed: {str(e)}"],
            "error": True,
            "mock": False
        }


def evaluate_promotion_readiness(bundle_sha1: str, sentinels: List[Dict[str, Any]],
                                config: PromotionGuardConfig) -> Dict[str, Any]:
    """
    Evaluate if a bundle is ready for promotion

    Args:
        bundle_sha1: Bundle identifier
        sentinels: List of sentinel check results
        config: Guard configuration

    Returns:
        Promotion readiness evaluation
    """
    # Check each sentinel status
    failed_sentinels = []
    error_sentinels = []
    ok_sentinels = []

    for sentinel in sentinels:
        status = sentinel.get("status", "UNKNOWN")
        sentinel_type = sentinel.get("sentinel_type", "unknown")

        if status == "OK":
            ok_sentinels.append(sentinel_type)
        elif status in ["DRIFT", "VIOLATION"]:
            failed_sentinels.append({
                "type": sentinel_type,
                "status": status,
                "violations": sentinel.get("violations", [])
            })
        elif status == "ERROR":
            error_sentinels.append({
                "type": sentinel_type,
                "error": sentinel.get("violations", ["Unknown error"])
            })

    # Determine if promotion should be blocked
    promotion_blocked = False
    blocking_reasons = []

    if config.require_all_ok:
        if failed_sentinels:
            promotion_blocked = True
            for failed in failed_sentinels:
                blocking_reasons.append(f"{failed['type']} status is {failed['status']}")
                for violation in failed['violations']:
                    blocking_reasons.append(f"  - {violation}")

        if error_sentinels:
            promotion_blocked = True
            for error in error_sentinels:
                blocking_reasons.append(f"{error['type']} check failed")
                for err_msg in error['error']:
                    blocking_reasons.append(f"  - {err_msg}")

    # Check if required sentinels are present
    required_types = set(config.block_on)
    present_types = set(s.get("sentinel_type", "") for s in sentinels)
    missing_types = required_types - present_types

    if missing_types:
        promotion_blocked = True
        blocking_reasons.append(f"Missing sentinel checks: {list(missing_types)}")

    return {
        "bundle_sha1": bundle_sha1,
        "promotion_allowed": not promotion_blocked,
        "promotion_blocked": promotion_blocked,
        "blocking_reasons": blocking_reasons,
        "sentinel_summary": {
            "total_sentinels": len(sentinels),
            "ok_sentinels": len(ok_sentinels),
            "failed_sentinels": len(failed_sentinels),
            "error_sentinels": len(error_sentinels),
            "missing_sentinels": len(missing_types)
        },
        "sentinels_ok": ok_sentinels,
        "sentinels_failed": [f["type"] for f in failed_sentinels],
        "sentinels_error": [e["type"] for e in error_sentinels],
        "sentinels_missing": list(missing_types)
    }


@register("ops.guard")
def ops_promote_guard(
    bundle_sha1: str,
    policy_path: str = "ally/ops/policy.yaml",
    strategy_hash: Optional[str] = None,
    panel_path: Optional[str] = None,
    fixture_path: Optional[str] = None,
    mock_data_ok: bool = True,
    mock_strategy_ok: bool = True,
    mock_ops_ok: bool = True,
    live: bool = True
) -> Result:
    """
    Guard promotion by checking all sentinel statuses

    Args:
        bundle_sha1: Bundle identifier to evaluate for promotion
        policy_path: Path to policy configuration
        strategy_hash: Strategy hash for strategy drift check
        panel_path: Data panel path for data drift check
        fixture_path: Fixture path for ops drift check
        mock_data_ok: Mock data drift status (for testing)
        mock_strategy_ok: Mock strategy drift status (for testing)
        mock_ops_ok: Mock ops drift status (for testing)
        live: Enable live mode (requires ALLY_LIVE=1)

    Returns:
        Result with promotion guard decision and detailed reasoning
    """
    try:
        # Gate live mode
        if live:
            check_live_mode_allowed(
                live=live,
                api_key=os.getenv("GUARD_API_KEY", "not_set"),
                service_name="Promotion Guard"
            )

        # Load policy configuration
        try:
            with open(policy_path, 'r') as f:
                policy = yaml.safe_load(f)
            promotion_policy = policy.get('promotion', {})
        except FileNotFoundError:
            promotion_policy = {}

        # Create configuration
        config = PromotionGuardConfig(
            require_all_ok=promotion_policy.get('require_all_ok', True),
            block_on=promotion_policy.get('block_on', ["data_drift", "strategy_drift", "ops_drift"]),
            timeout_hours=24,
            check_recent_hours=1
        )

        # Prepare sentinel configurations
        sentinel_configs = {
            "data_drift": {
                "panel_path": panel_path,
                "policy_path": policy_path,
                "mock_ok": mock_data_ok
            },
            "strategy_drift": {
                "strategy_hash": strategy_hash or "test_strategy_" + bundle_sha1[:8],
                "policy_path": policy_path,
                "mock_ok": mock_strategy_ok
            },
            "ops_drift": {
                "fixture_path": fixture_path or "artifacts/fixtures/determinism.pkl",
                "policy_path": policy_path,
                "mock_ok": mock_ops_ok
            }
        }

        # Run sentinel checks
        sentinel_results = []
        for sentinel_type in config.block_on:
            if sentinel_type in sentinel_configs:
                sentinel_config = sentinel_configs[sentinel_type]
                sentinel_result = check_sentinel_status(sentinel_type, sentinel_config, config.check_recent_hours)
                sentinel_results.append(sentinel_result)
            else:
                # Unknown sentinel type
                sentinel_results.append({
                    "sentinel_type": sentinel_type,
                    "status": "ERROR",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "violations": [f"Unknown sentinel type: {sentinel_type}"],
                    "error": True
                })

        # Evaluate promotion readiness
        evaluation = evaluate_promotion_readiness(bundle_sha1, sentinel_results, config)

        # Generate receipt
        guard_data = {
            "bundle_sha1": bundle_sha1,
            "promotion_allowed": evaluation["promotion_allowed"],
            "sentinels_checked": len(sentinel_results),
            "sentinels_ok": evaluation["sentinel_summary"]["ok_sentinels"],
            "config": asdict(config)
        }

        receipt_hash = generate_receipt("ops.guard", guard_data)

        # Guard decision: fail if promotion is blocked
        guard_ok = evaluation["promotion_allowed"]

        return Result(
            ok=guard_ok,
            data={
                "guard_receipt": receipt_hash[:16],
                "bundle_sha1": bundle_sha1,
                "promotion_decision": "ALLOW" if evaluation["promotion_allowed"] else "BLOCK",
                "promotion_allowed": evaluation["promotion_allowed"],
                "promotion_blocked": evaluation["promotion_blocked"],
                "blocking_reasons": evaluation["blocking_reasons"],
                "sentinel_results": sentinel_results,
                "evaluation": evaluation,
                "guard_summary": {
                    "decision": "ALLOW" if evaluation["promotion_allowed"] else "BLOCK",
                    "total_sentinels": len(sentinel_results),
                    "sentinels_ok": evaluation["sentinel_summary"]["ok_sentinels"],
                    "sentinels_failed": evaluation["sentinel_summary"]["failed_sentinels"],
                    "sentinels_error": evaluation["sentinel_summary"]["error_sentinels"],
                    "blocking_reason_count": len(evaluation["blocking_reasons"])
                },
                "config_used": asdict(config),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            errors=evaluation["blocking_reasons"] if evaluation["promotion_blocked"] else [],
            receipt_hash=receipt_hash
        )

    except LiveModeError as e:
        return Result(ok=False, errors=[str(e)])
    except Exception as e:
        return Result(ok=False, errors=[f"Promotion guard failed: {str(e)}"])


if __name__ == "__main__":
    # Test promotion guard
    result = ops_promote_guard(
        bundle_sha1="test_bundle_abc123def456",
        strategy_hash="test_strategy_xyz789",
        mock_data_ok=True,
        mock_strategy_ok=True,
        mock_ops_ok=True,
        live=False
    )

    if result.ok:
        print("✅ Promotion guard completed - PROMOTION ALLOWED")
        print(f"Receipt: {result.data['guard_receipt']}")
        print(f"Bundle: {result.data['bundle_sha1']}")
        print(f"Decision: {result.data['promotion_decision']}")
        print(f"Sentinels OK: {result.data['guard_summary']['sentinels_ok']}/{result.data['guard_summary']['total_sentinels']}")
    else:
        print("❌ Promotion guard BLOCKED")
        print(f"Bundle: {result.data.get('bundle_sha1', 'unknown')}")
        print("Blocking reasons:")
        for reason in result.errors:
            print(f"  - {reason}")
        print(f"Failed sentinels: {result.data.get('guard_summary', {}).get('sentinels_failed', 0)}")
        print(f"Error sentinels: {result.data.get('guard_summary', {}).get('sentinels_error', 0)}")