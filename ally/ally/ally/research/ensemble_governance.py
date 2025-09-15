#!/usr/bin/env python3
"""
Phase 9: Ensemble Governance & Weight Allocation
Pure Python implementation for CI compatibility
"""

import os
import json
import yaml
import math
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Local imports for receipts
try:
    from ..ops.receipts import write_tool_receipt
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
                'caps': {
                    'weight_per_strategy_max': 0.20,
                    'weight_per_family_max': 0.40,
                    'gross_exposure_max': 1.00,
                    'net_exposure_band': 0.20
                },
                'correlation': {
                    'pairwise_max': 0.65,
                    'factor_exposure_max': 1.5
                },
                'risk_budget': {
                    'ctr_max': 0.15,
                    'drawdown_scale': True
                },
                'turnover': {
                    'max_rebalance_turnover': 0.35
                },
                'novelty': {
                    'min_n_strategies': 3,
                    'novelty_quota_min': 0.10
                },
                'drift': {
                    'require_sentinel_ok': True,
                    'deweight_on_warn': 0.50
                }
            }
        }


def load_approved_bundles(bundles_path: str) -> dict:
    """Load approved strategy bundles"""
    with open(bundles_path, 'r') as f:
        return json.load(f)


def load_correlation_matrix(corr_path: str) -> dict:
    """Load pairwise correlation matrix from CSV"""
    corr_matrix = {}
    with open(corr_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategy = row['strategy']
            corr_matrix[strategy] = {}
            for other_strategy in row:
                if other_strategy != 'strategy':
                    corr_matrix[strategy][other_strategy] = float(row[other_strategy])
    return corr_matrix


def load_factor_exposures(factors_path: str) -> dict:
    """Load factor exposures from CSV"""
    exposures = {}
    with open(factors_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            strategy = row['strategy']
            exposures[strategy] = {}
            for factor in row:
                if factor != 'strategy':
                    exposures[strategy][factor] = float(row[factor])
    return exposures


def check_weight_caps(weights: dict, strategies: list, policy: dict) -> Tuple[bool, List[str]]:
    """Check weight caps per strategy and family"""
    violations = []
    caps_config = policy['phase9']['caps']

    strategy_max = caps_config['weight_per_strategy_max']
    family_max = caps_config['weight_per_family_max']

    # Check per-strategy caps
    for strategy_id, weight in weights.items():
        if weight > strategy_max:
            violations.append(f"strategy_weight[{strategy_id}]={weight:.3f}>{strategy_max}")

    # Check per-family caps
    family_weights = {}
    strategy_lookup = {s['id']: s for s in strategies}

    for strategy_id, weight in weights.items():
        if strategy_id in strategy_lookup:
            family = strategy_lookup[strategy_id]['family']
            family_weights[family] = family_weights.get(family, 0) + weight

    for family, total_weight in family_weights.items():
        if total_weight > family_max:
            violations.append(f"family_weight[{family}]={total_weight:.3f}>{family_max}")

    return len(violations) == 0, violations


def check_correlation_caps(weights: dict, corr_matrix: dict, policy: dict) -> Tuple[bool, List[str]]:
    """Check pairwise correlation caps"""
    violations = []
    pairwise_max = policy['phase9']['correlation']['pairwise_max']

    strategies = list(weights.keys())

    for i, strat1 in enumerate(strategies):
        for j, strat2 in enumerate(strategies[i+1:], i+1):
            if strat1 in corr_matrix and strat2 in corr_matrix[strat1]:
                corr = corr_matrix[strat1][strat2]
                weight_product = weights[strat1] * weights[strat2]

                # Only flag if both strategies have meaningful weight AND high correlation
                if weight_product > 0.01 and corr > pairwise_max:
                    violations.append(f"pairwise_corr[{strat1},{strat2}]={corr:.2f}>{pairwise_max}")

    return len(violations) == 0, violations


def check_exposure_caps(weights: dict, policy: dict) -> Tuple[bool, List[str]]:
    """Check gross and net exposure caps"""
    violations = []
    caps_config = policy['phase9']['caps']

    gross_exposure = sum(abs(w) for w in weights.values())
    net_exposure = sum(weights.values())

    gross_max = caps_config['gross_exposure_max']
    net_band = caps_config['net_exposure_band']

    if gross_exposure > gross_max:
        violations.append(f"gross_exposure={gross_exposure:.3f}>{gross_max}")

    if abs(net_exposure - 1.0) > net_band:
        violations.append(f"net_exposure_deviation={abs(net_exposure - 1.0):.3f}>{net_band}")

    return len(violations) == 0, violations


def check_novelty_requirements(strategies: list, policy: dict) -> Tuple[bool, List[str]]:
    """Check minimum number of strategies and novelty quota"""
    violations = []
    novelty_config = policy['phase9']['novelty']

    min_n = novelty_config['min_n_strategies']
    min_quota = novelty_config['novelty_quota_min']

    if len(strategies) < min_n:
        violations.append(f"strategy_count={len(strategies)}<{min_n}")

    # Check for new/satellite strategies (simplified heuristic)
    satellite_weight = sum(s['weight_hint'] for s in strategies if s.get('risk_tag') == 'satellite')
    if satellite_weight < min_quota:
        violations.append(f"novelty_quota={satellite_weight:.3f}<{min_quota}")

    return len(violations) == 0, violations


def apply_drift_deweighting(weights: dict, strategies: list, policy: dict) -> Tuple[dict, List[str]]:
    """Apply drift-based deweighting"""
    violations = []
    drift_config = policy['phase9']['drift']

    require_ok = drift_config['require_sentinel_ok']
    deweight_factor = drift_config['deweight_on_warn']

    adjusted_weights = weights.copy()
    strategy_lookup = {s['id']: s for s in strategies}

    for strategy_id in weights:
        if strategy_id in strategy_lookup:
            drift_status = strategy_lookup[strategy_id].get('drift_status', 'UNKNOWN')

            if require_ok and drift_status != 'OK':
                if drift_status == 'DRIFT':
                    # Deweight on drift
                    original_weight = adjusted_weights[strategy_id]
                    adjusted_weights[strategy_id] *= deweight_factor
                    violations.append(f"drift_deweight[{strategy_id}]: {original_weight:.3f}→{adjusted_weights[strategy_id]:.3f}")
                else:
                    # Exclude on error
                    adjusted_weights[strategy_id] = 0.0
                    violations.append(f"drift_exclude[{strategy_id}]: status={drift_status}")

    # Renormalize weights
    total_weight = sum(adjusted_weights.values())
    if total_weight > 0:
        adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

    return adjusted_weights, violations


def calculate_initial_weights(strategies: list) -> dict:
    """Calculate initial weights from hints"""
    # Use weight hints as starting point
    initial_weights = {s['id']: s['weight_hint'] for s in strategies}

    # Normalize to sum to 1
    total_hint = sum(initial_weights.values())
    if total_hint > 0:
        initial_weights = {k: v / total_hint for k, v in initial_weights.items()}

    return initial_weights


def govern_ensemble(
    approved_path: str = "artifacts/fixtures/phase9/approved_bundles.json",
    corr_path: str = "artifacts/fixtures/phase9/pairwise_corr.csv",
    factors_path: str = "artifacts/fixtures/phase9/factor_exposures.csv",
    policy_path: str = "ally/ops/policy.yaml",
    snapshot_path: str = None,
    asof: str = None,
    live: bool = False
) -> dict:
    """Main ensemble governance function"""

    try:
        # Load inputs
        policy = load_policy(policy_path)
        bundles_data = load_approved_bundles(approved_path)
        corr_matrix = load_correlation_matrix(corr_path)
        factor_exposures = load_factor_exposures(factors_path)

        strategies = bundles_data['strategies']

        # Calculate initial weights
        initial_weights = calculate_initial_weights(strategies)

        # Apply drift deweighting
        adjusted_weights, drift_violations = apply_drift_deweighting(initial_weights, strategies, policy)

        # Check all governance rules
        all_violations = drift_violations.copy()

        # Weight caps
        weight_caps_ok, weight_violations = check_weight_caps(adjusted_weights, strategies, policy)
        all_violations.extend(weight_violations)

        # Correlation caps
        corr_caps_ok, corr_violations = check_correlation_caps(adjusted_weights, corr_matrix, policy)
        all_violations.extend(corr_violations)

        # Exposure caps
        exposure_caps_ok, exposure_violations = check_exposure_caps(adjusted_weights, policy)
        all_violations.extend(exposure_violations)

        # Novelty requirements
        novelty_ok, novelty_violations = check_novelty_requirements(strategies, policy)
        all_violations.extend(novelty_violations)

        # Overall governance status
        governance_ok = (weight_caps_ok and corr_caps_ok and exposure_caps_ok and novelty_ok)

        # Build result
        result = {
            "timestamp": asof or datetime.now().isoformat() + "Z",
            "governance_ok": governance_ok,
            "violations": all_violations,
            "initial_weights": initial_weights,
            "final_weights": adjusted_weights,
            "weight_sum": sum(adjusted_weights.values()),
            "gross_exposure": sum(abs(w) for w in adjusted_weights.values()),
            "net_exposure": sum(adjusted_weights.values()),
            "strategy_count": len([w for w in adjusted_weights.values() if w > 0.001]),
            "drift_enforced": len(drift_violations) > 0,
            "pairwise_breaches": len(corr_violations),
            "binding_caps": {
                "weight_caps": not weight_caps_ok,
                "correlation_caps": not corr_caps_ok,
                "exposure_caps": not exposure_caps_ok,
                "novelty_reqs": not novelty_ok
            }
        }

        # Write artifact
        os.makedirs("artifacts/research/ensemble", exist_ok=True)
        timestamp_str = asof.replace(":", "").replace("-", "") if asof else datetime.now().strftime("%Y%m%dT%H%M%S")
        with open(f"artifacts/research/ensemble/weights_{timestamp_str}.json", "w") as f:
            json.dump(result, f, indent=2)

        # Write receipt
        params = {
            "approved_path": approved_path,
            "corr_path": corr_path,
            "factors_path": factors_path,
            "policy_path": policy_path,
            "asof": asof,
            "live": live
        }

        status = "OK" if governance_ok else "VIOLATIONS"
        receipt_hash = write_tool_receipt("ensemble.govern", params, status, result)

        result["receipt_hash"] = receipt_hash
        return result

    except Exception as e:
        error_result = {
            "timestamp": asof or datetime.now().isoformat() + "Z",
            "governance_ok": False,
            "error": str(e),
            "violations": [f"governance_error: {str(e)}"],
            "weight_sum": 0.0,
            "pairwise_breaches": 0,
            "drift_enforced": False
        }
        receipt_hash = write_tool_receipt("ensemble.govern", {}, "ERROR", error_result)
        error_result["receipt_hash"] = receipt_hash
        return error_result


if __name__ == "__main__":
    # Test ensemble governance
    result = govern_ensemble(
        asof="2025-09-15T12:00:00Z",
        live=False
    )

    print("✅ Ensemble governance completed")
    print(f"Receipt: {result['receipt_hash']}")
    print(f"Governance OK: {result['governance_ok']}")
    print(f"Violations: {len(result['violations'])}")
    print(f"Weight sum: {result['weight_sum']:.6f}")
    print(f"Strategy count: {result['strategy_count']}")

    if result['violations']:
        print("Violations found:")
        for violation in result['violations']:
            print(f"  - {violation}")