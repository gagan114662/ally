#!/usr/bin/env python3
"""
System heartbeat and status reporting - Phase 8

Implements structured status reporting for dashboards and daily narratives,
aggregating drift detection results and system health metrics.
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

# Create a simple receipt generator
def generate_receipt(tool_name: str, data: dict) -> str:
    """Generate a simple receipt hash"""
    import json
    import hashlib
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


@dataclass
class HeartbeatConfig:
    """Configuration for heartbeat reporting"""
    since_hours: int = 24
    include_details: bool = True
    timezone: str = "UTC"
    aggregate_by_hour: bool = True


def get_mock_receipts_data(since_hours: int) -> List[Dict[str, Any]]:
    """
    Generate mock receipts data for CI/testing

    Args:
        since_hours: Hours to look back

    Returns:
        List of mock receipt records
    """
    now = datetime.utcnow()
    receipts = []

    # Generate mock drift detection receipts
    tools = ["ops.drift.data", "ops.drift.strategy", "ops.drift.ops", "ops.guard"]
    statuses = ["OK", "DRIFT", "OK", "OK"]  # One drift for testing

    for i, (tool, status) in enumerate(zip(tools, statuses)):
        timestamp = now - timedelta(hours=i, minutes=30)
        receipt = {
            "tool": tool,
            "timestamp": timestamp.isoformat() + "Z",
            "receipt_hash": f"mock_hash_{i:03d}",
            "params_hash": f"params_{i:03d}",
            "extra": {
                "status": status,
                "violations": [] if status == "OK" else ["Mock violation for testing"],
                "timestamp": timestamp.isoformat() + "Z"
            }
        }
        receipts.append(receipt)

    # Add some portfolio optimization receipts
    for i in range(3):
        timestamp = now - timedelta(hours=i*2, minutes=45)
        receipt = {
            "tool": "portfolio.optimize",
            "timestamp": timestamp.isoformat() + "Z",
            "receipt_hash": f"portfolio_hash_{i:03d}",
            "params_hash": f"portfolio_params_{i:03d}",
            "extra": {
                "method": "erc",
                "constraints_ok": True,
                "ex_ante_vol": 0.12,
                "ex_ante_sr": 0.85
            }
        }
        receipts.append(receipt)

    return receipts


def analyze_drift_status(receipts: List[Dict[str, Any]], since_hours: int) -> Dict[str, Any]:
    """
    Analyze drift detection status from receipts

    Args:
        receipts: List of receipt records
        since_hours: Hours to analyze

    Returns:
        Drift analysis summary
    """
    since_time = datetime.utcnow() - timedelta(hours=since_hours)

    # Filter drift-related receipts
    drift_tools = ["ops.drift.data", "ops.drift.strategy", "ops.drift.ops"]
    drift_receipts = [
        r for r in receipts
        if r.get("tool") in drift_tools
        and datetime.fromisoformat(r.get("timestamp", "").replace("Z", "")) >= since_time
    ]

    # Aggregate by tool
    tool_status = {}
    for tool in drift_tools:
        tool_receipts = [r for r in drift_receipts if r.get("tool") == tool]

        if tool_receipts:
            # Get latest status for each tool
            latest_receipt = max(tool_receipts, key=lambda x: x.get("timestamp", ""))
            status = latest_receipt.get("extra", {}).get("status", "UNKNOWN")
            violations = latest_receipt.get("extra", {}).get("violations", [])

            tool_status[tool] = {
                "status": status,
                "latest_check": latest_receipt.get("timestamp"),
                "violations_count": len(violations),
                "violations": violations,
                "checks_in_period": len(tool_receipts)
            }
        else:
            tool_status[tool] = {
                "status": "NO_DATA",
                "latest_check": None,
                "violations_count": 0,
                "violations": [],
                "checks_in_period": 0
            }

    # Overall health assessment
    all_statuses = [t["status"] for t in tool_status.values()]
    overall_health = "HEALTHY" if all(s == "OK" for s in all_statuses) else "DEGRADED"

    if any(s == "DRIFT" for s in all_statuses):
        overall_health = "DRIFT_DETECTED"
    elif any(s in ["ERROR", "NO_DATA"] for s in all_statuses):
        overall_health = "DATA_ISSUES"

    return {
        "overall_health": overall_health,
        "tool_status": tool_status,
        "total_drift_checks": len(drift_receipts),
        "tools_with_issues": len([t for t in tool_status.values() if t["status"] != "OK"]),
        "analysis_period_hours": since_hours
    }


def analyze_portfolio_performance(receipts: List[Dict[str, Any]], since_hours: int) -> Dict[str, Any]:
    """
    Analyze portfolio optimization performance from receipts

    Args:
        receipts: List of receipt records
        since_hours: Hours to analyze

    Returns:
        Portfolio performance summary
    """
    since_time = datetime.utcnow() - timedelta(hours=since_hours)

    # Filter portfolio receipts
    portfolio_receipts = [
        r for r in receipts
        if r.get("tool") == "portfolio.optimize"
        and datetime.fromisoformat(r.get("timestamp", "").replace("Z", "")) >= since_time
    ]

    if not portfolio_receipts:
        return {
            "optimizations_count": 0,
            "avg_sharpe": None,
            "avg_volatility": None,
            "constraints_violations": 0,
            "methods_used": []
        }

    # Extract metrics
    sharpe_ratios = []
    volatilities = []
    constraints_violations = 0
    methods_used = set()

    for receipt in portfolio_receipts:
        extra = receipt.get("extra", {})

        # Extract performance metrics
        sharpe = extra.get("ex_ante_sr")
        vol = extra.get("ex_ante_vol")
        method = extra.get("method")
        constraints_ok = extra.get("constraints_ok", True)

        if sharpe is not None:
            sharpe_ratios.append(sharpe)
        if vol is not None:
            volatilities.append(vol)
        if method:
            methods_used.add(method)
        if not constraints_ok:
            constraints_violations += 1

    # Calculate averages
    avg_sharpe = sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else None
    avg_volatility = sum(volatilities) / len(volatilities) if volatilities else None

    return {
        "optimizations_count": len(portfolio_receipts),
        "avg_sharpe": avg_sharpe,
        "avg_volatility": avg_volatility,
        "constraints_violations": constraints_violations,
        "methods_used": list(methods_used),
        "latest_optimization": portfolio_receipts[0].get("timestamp") if portfolio_receipts else None
    }


def generate_system_summary(drift_analysis: Dict[str, Any],
                          portfolio_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate overall system health summary

    Args:
        drift_analysis: Drift detection analysis
        portfolio_analysis: Portfolio performance analysis

    Returns:
        System summary
    """
    # Determine overall system status
    drift_health = drift_analysis["overall_health"]
    portfolio_health = "HEALTHY" if portfolio_analysis["constraints_violations"] == 0 else "ISSUES"

    if drift_health == "DRIFT_DETECTED":
        system_status = "DRIFT_DETECTED"
    elif drift_health == "DATA_ISSUES":
        system_status = "DATA_ISSUES"
    elif portfolio_health == "ISSUES":
        system_status = "PORTFOLIO_ISSUES"
    elif drift_health == "HEALTHY" and portfolio_health == "HEALTHY":
        system_status = "HEALTHY"
    else:
        system_status = "DEGRADED"

    # Generate action items
    action_items = []

    if drift_analysis["tools_with_issues"] > 0:
        action_items.append(f"Investigate {drift_analysis['tools_with_issues']} drift detection tools with issues")

    if portfolio_analysis["constraints_violations"] > 0:
        action_items.append(f"Review {portfolio_analysis['constraints_violations']} constraint violations in portfolio optimization")

    if portfolio_analysis["optimizations_count"] == 0:
        action_items.append("No portfolio optimizations found in the analysis period")

    return {
        "system_status": system_status,
        "health_score": calculate_health_score(drift_analysis, portfolio_analysis),
        "action_items": action_items,
        "uptime_indicators": {
            "drift_monitoring": drift_health == "HEALTHY",
            "portfolio_optimization": portfolio_analysis["optimizations_count"] > 0,
            "constraint_compliance": portfolio_analysis["constraints_violations"] == 0
        }
    }


def calculate_health_score(drift_analysis: Dict[str, Any],
                         portfolio_analysis: Dict[str, Any]) -> float:
    """
    Calculate overall system health score (0-100)

    Args:
        drift_analysis: Drift detection analysis
        portfolio_analysis: Portfolio performance analysis

    Returns:
        Health score between 0 and 100
    """
    score = 100.0

    # Drift detection penalties
    if drift_analysis["overall_health"] == "DRIFT_DETECTED":
        score -= 30
    elif drift_analysis["overall_health"] == "DATA_ISSUES":
        score -= 20
    elif drift_analysis["overall_health"] == "DEGRADED":
        score -= 10

    # Portfolio performance penalties
    if portfolio_analysis["constraints_violations"] > 0:
        score -= min(20, portfolio_analysis["constraints_violations"] * 5)

    if portfolio_analysis["optimizations_count"] == 0:
        score -= 15

    return max(0.0, score)


@register("ops.heartbeat")
def ops_heartbeat(
    since_hours: int = 24,
    policy_path: str = "ally/ops/policy.yaml",
    include_details: bool = True,
    live: bool = True
) -> Result:
    """
    Generate system heartbeat and status report

    Args:
        since_hours: Hours to look back for analysis
        policy_path: Path to policy configuration
        include_details: Include detailed analysis
        live: Enable live mode (requires ALLY_LIVE=1)

    Returns:
        Result with comprehensive system status report
    """
    try:
        # Gate live mode
        if live:
            check_live_mode_allowed(
                live=live,
                api_key=os.getenv("HEARTBEAT_API_KEY", "not_set"),
                service_name="System Heartbeat"
            )

        # Load policy configuration
        try:
            with open(policy_path, 'r') as f:
                policy = yaml.safe_load(f)
            reporting_policy = policy.get('reporting', {})
        except FileNotFoundError:
            reporting_policy = {}

        # Create configuration
        config = HeartbeatConfig(
            since_hours=since_hours or reporting_policy.get('heartbeat_since_hours', 24),
            include_details=include_details,
            timezone=reporting_policy.get('timezone', 'UTC')
        )

        # Load receipts data (mock for CI)
        receipts_data = get_mock_receipts_data(config.since_hours)

        # Analyze drift detection status
        drift_analysis = analyze_drift_status(receipts_data, config.since_hours)

        # Analyze portfolio performance
        portfolio_analysis = analyze_portfolio_performance(receipts_data, config.since_hours)

        # Generate system summary
        system_summary = generate_system_summary(drift_analysis, portfolio_analysis)

        # Generate receipt
        heartbeat_data = {
            "system_status": system_summary["system_status"],
            "health_score": system_summary["health_score"],
            "analysis_period_hours": config.since_hours,
            "total_receipts": len(receipts_data),
            "config": asdict(config)
        }

        receipt_hash = generate_receipt("ops.heartbeat", heartbeat_data)

        return Result(
            ok=True,
            data={
                "heartbeat_receipt": receipt_hash[:16],
                "system_summary": system_summary,
                "drift_analysis": drift_analysis,
                "portfolio_analysis": portfolio_analysis,
                "reporting_config": asdict(config),
                "snapshot": {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "analysis_period": f"{config.since_hours} hours",
                    "total_receipts_analyzed": len(receipts_data),
                    "system_status": system_summary["system_status"],
                    "health_score": system_summary["health_score"]
                },
                "daily_narrative": {
                    "status": system_summary["system_status"],
                    "summary": f"System health score: {system_summary['health_score']:.1f}/100",
                    "drift_status": drift_analysis["overall_health"],
                    "portfolio_optimizations": portfolio_analysis["optimizations_count"],
                    "action_items": system_summary["action_items"],
                    "next_review": (datetime.utcnow() + timedelta(hours=24)).isoformat() + "Z"
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            },
            receipt_hash=receipt_hash
        )

    except LiveModeError as e:
        return Result(ok=False, errors=[str(e)])
    except Exception as e:
        return Result(ok=False, errors=[f"Heartbeat generation failed: {str(e)}"])


if __name__ == "__main__":
    # Test heartbeat generation
    result = ops_heartbeat(
        since_hours=24,
        include_details=True,
        live=False
    )

    if result.ok:
        print("✅ System heartbeat completed")
        print(f"Receipt: {result.data['heartbeat_receipt']}")
        print(f"System status: {result.data['snapshot']['system_status']}")
        print(f"Health score: {result.data['snapshot']['health_score']:.1f}/100")
        print(f"Drift status: {result.data['drift_analysis']['overall_health']}")
        print(f"Portfolio optimizations: {result.data['portfolio_analysis']['optimizations_count']}")

        if result.data['system_summary']['action_items']:
            print("Action items:")
            for item in result.data['system_summary']['action_items']:
                print(f"  - {item}")
    else:
        print("❌ Heartbeat generation failed")
        for error in result.errors:
            print(f"Error: {error}")