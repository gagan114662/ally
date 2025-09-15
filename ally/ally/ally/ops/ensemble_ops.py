#!/usr/bin/env python3
"""
Phase 9: Meta-Ops and Rebalance Planning
Translates ensemble weights into executable rebalance plans
"""

import os
import json
import yaml
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Local imports for receipts
try:
    from .receipts import write_tool_receipt
except ImportError:
    # Fallback receipt system
    import hashlib
    def write_tool_receipt(tool_name: str, params: dict, status: str, result_data: dict = None):
        data = {'tool': tool_name, 'params': params, 'status': status}
        receipt_hash = hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
        print(f"RECEIPT: {tool_name}:{receipt_hash}")
        return receipt_hash


def load_policy(policy_path: str) -> dict:
    """Load policy configuration"""
    try:
        with open(policy_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            'phase9': {
                'turnover': {'max_rebalance_turnover': 0.35}
            }
        }


def load_cost_model(cost_model_path: str) -> dict:
    """Load cost model configuration"""
    try:
        with open(cost_model_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            'model': 'bps=10,impact=k*sqrt(q),k=6.0,borrow_bps=50',
            'capacity_usd': 50000000,
            'max_turnover': 0.35
        }


def load_last_weights(last_weights_path: str) -> dict:
    """Load last portfolio weights"""
    try:
        with open(last_weights_path, 'r') as f:
            data = json.load(f)
            return data.get('weights', {})
    except FileNotFoundError:
        return {}


def calculate_weight_deltas(target_weights: dict, last_weights: dict) -> dict:
    """Calculate weight changes from last to target"""
    deltas = {}

    # Get all strategies (union of current and target)
    all_strategies = set(target_weights.keys()) | set(last_weights.keys())

    for strategy in all_strategies:
        current = last_weights.get(strategy, 0.0)
        target = target_weights.get(strategy, 0.0)
        deltas[strategy] = target - current

    return deltas


def calculate_turnover(weight_deltas: dict) -> float:
    """Calculate total turnover from weight deltas"""
    return sum(abs(delta) for delta in weight_deltas.values()) / 2.0


def estimate_transaction_costs(weight_deltas: dict, cost_model: dict, portfolio_value: float = 10_000_000) -> dict:
    """Estimate transaction costs using simple model"""

    # Parse cost model (simplified)
    model_str = cost_model.get('model', 'bps=10')
    base_bps = 10  # Default

    if 'bps=' in model_str:
        bps_part = model_str.split('bps=')[1].split(',')[0]
        try:
            base_bps = float(bps_part)
        except ValueError:
            base_bps = 10

    total_cost = 0.0
    strategy_costs = {}

    for strategy, delta in weight_deltas.items():
        if abs(delta) > 0.001:  # Only calculate for meaningful changes
            notional = abs(delta) * portfolio_value
            cost_bps = base_bps + (abs(delta) * 50)  # Impact scaling
            cost_usd = notional * (cost_bps / 10000)

            strategy_costs[strategy] = {
                "delta_weight": delta,
                "notional_usd": notional,
                "cost_bps": cost_bps,
                "cost_usd": cost_usd
            }
            total_cost += cost_usd

    return {
        "total_cost_usd": total_cost,
        "total_cost_bps": (total_cost / portfolio_value) * 10000,
        "strategy_costs": strategy_costs
    }


def generate_rebalance_orders(weight_deltas: dict, portfolio_value: float = 10_000_000) -> List[dict]:
    """Generate rebalance orders from weight deltas"""

    orders = []
    order_id = 1

    for strategy, delta in weight_deltas.items():
        if abs(delta) > 0.001:  # Only generate orders for meaningful changes
            notional = delta * portfolio_value
            side = "BUY" if delta > 0 else "SELL"

            order = {
                "order_id": f"ORD_{order_id:04d}",
                "strategy": strategy,
                "side": side,
                "notional_usd": abs(notional),
                "weight_delta": delta,
                "status": "PENDING"
            }

            orders.append(order)
            order_id += 1

    return orders


def simulate_order_fills(orders: List[dict], backend: str = "simulator") -> List[dict]:
    """Simulate order execution"""

    filled_orders = []

    for order in orders:
        filled_order = order.copy()
        filled_order.update({
            "status": "FILLED",
            "fill_price": 1.0,  # Simplified
            "fill_time": datetime.now().isoformat() + "Z",
            "fill_notional": order["notional_usd"],
            "execution_cost_bps": 12.0,  # Simplified
            "backend": backend
        })
        filled_orders.append(filled_order)

    return filled_orders


def ensemble_ops_apply(
    target_weights: dict,
    last_weights_path: str = "artifacts/fixtures/phase9/last_weights.json",
    cost_model_path: str = "artifacts/fixtures/phase9/cost_model.yaml",
    policy_path: str = "ally/ops/policy.yaml",
    backend: str = "simulator",
    live: bool = False
) -> dict:
    """Apply ensemble weights and create rebalance plan"""

    try:
        # Load inputs
        policy = load_policy(policy_path)
        cost_model = load_cost_model(cost_model_path)
        last_weights = load_last_weights(last_weights_path)

        # Calculate rebalance plan
        weight_deltas = calculate_weight_deltas(target_weights, last_weights)
        turnover = calculate_turnover(weight_deltas)

        # Check turnover constraints
        max_turnover = policy['phase9']['turnover']['max_rebalance_turnover']
        turnover_ok = turnover <= max_turnover

        turnover_violations = []
        if not turnover_ok:
            turnover_violations.append(f"turnover={turnover:.3f}>{max_turnover}")

        # Estimate costs
        cost_analysis = estimate_transaction_costs(weight_deltas, cost_model)

        # Generate orders
        orders = generate_rebalance_orders(weight_deltas)

        # Execute orders (simulate)
        if backend == "simulator":
            filled_orders = simulate_order_fills(orders, backend)
        else:
            # For other backends, would integrate with actual execution
            filled_orders = orders  # Placeholder

        # Build result
        result = {
            "timestamp": datetime.now().isoformat() + "Z",
            "target_weights": target_weights,
            "last_weights": last_weights,
            "weight_deltas": weight_deltas,
            "turnover": turnover,
            "turnover_ok": turnover_ok,
            "turnover_violations": turnover_violations,
            "cost_analysis": cost_analysis,
            "orders_generated": len(orders),
            "orders_filled": len(filled_orders),
            "backend": backend,
            "rebalance_plan": {
                "orders": orders,
                "filled_orders": filled_orders,
                "summary": {
                    "strategies_rebalanced": len([d for d in weight_deltas.values() if abs(d) > 0.001]),
                    "total_turnover": turnover,
                    "projected_cost_bps": cost_analysis["total_cost_bps"],
                    "projected_cost_usd": cost_analysis["total_cost_usd"]
                }
            }
        }

        # Write artifacts
        os.makedirs("artifacts/ops/rebalance", exist_ok=True)
        os.makedirs("artifacts/ops/orders", exist_ok=True)

        # Rebalance plan
        timestamp_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        with open(f"artifacts/ops/rebalance/plan_{timestamp_str}.json", "w") as f:
            json.dump(result, f, indent=2)

        # Orders (JSONL format)
        with open(f"artifacts/ops/orders/{backend}_{timestamp_str}.jsonl", "w") as f:
            for order in filled_orders:
                f.write(json.dumps(order) + "\n")

        # Write receipt
        params = {
            "target_weights": target_weights,
            "last_weights_path": last_weights_path,
            "cost_model_path": cost_model_path,
            "policy_path": policy_path,
            "backend": backend,
            "live": live
        }

        status = "OK" if turnover_ok else "TURNOVER_VIOLATION"
        receipt_hash = write_tool_receipt("ensemble.ops", params, status, result)

        result["receipt_hash"] = receipt_hash
        return result

    except Exception as e:
        error_result = {
            "timestamp": datetime.now().isoformat() + "Z",
            "turnover_ok": False,
            "error": str(e),
            "turnover_violations": [f"ops_error: {str(e)}"],
            "orders_generated": 0,
            "orders_filled": 0
        }
        receipt_hash = write_tool_receipt("ensemble.ops", {}, "ERROR", error_result)
        error_result["receipt_hash"] = receipt_hash
        return error_result


if __name__ == "__main__":
    # Test meta-ops with sample weights
    test_weights = {
        "XS_Momentum_v1": 0.35,
        "Value_BTM_v1": 0.35,
        "TS_Trend_v1": 0.30
    }

    result = ensemble_ops_apply(
        target_weights=test_weights,
        backend="simulator",
        live=False
    )

    print("✅ Ensemble ops completed")
    print(f"Receipt: {result['receipt_hash']}")
    print(f"Turnover OK: {result['turnover_ok']}")
    print(f"Turnover: {result['turnover']:.3f}")
    print(f"Orders generated: {result['orders_generated']}")
    print(f"Projected cost: {result['cost_analysis']['total_cost_bps']:.2f} bps")