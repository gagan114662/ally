#!/usr/bin/env python3
"""
Phase 9: Portfolio CLI for ensemble governance and rebalancing
"""

import sys
import os
import argparse
from pathlib import Path

# Add the nested ally path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from research.ensemble_governance import govern_ensemble
    from ops.ensemble_ops import ensemble_ops_apply
    from ops.receipts import write_tool_receipt
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def cmd_govern(args):
    """Run ensemble governance"""
    print("Running ensemble governance...")

    result = govern_ensemble(
        approved_path=args.approved if hasattr(args, 'approved') else "artifacts/fixtures/phase9/approved_bundles.json",
        corr_path=args.corr if hasattr(args, 'corr') else "artifacts/fixtures/phase9/pairwise_corr.csv",
        factors_path=args.factors if hasattr(args, 'factors') else "artifacts/fixtures/phase9/factor_exposures.csv",
        policy_path=args.policy,
        snapshot_path=args.snapshot if hasattr(args, 'snapshot') else None,
        asof=args.asof if hasattr(args, 'asof') else None,
        live=args.live
    )

    if result['governance_ok']:
        print(f"✅ Governance: PASSED")
    else:
        print(f"❌ Governance: FAILED ({len(result['violations'])} violations)")

    print(f"PROOF:run:ensemble.govern:{result['receipt_hash']}")

    # Write policy check receipt
    policy_params = {
        "governance_ok": result['governance_ok'],
        "violations_count": len(result['violations']),
        "weight_sum": result['weight_sum'],
        "pairwise_breaches": result['pairwise_breaches']
    }
    policy_status = "OK" if result['governance_ok'] else "VIOLATIONS"
    policy_receipt = write_tool_receipt("ensemble.policy", policy_params, policy_status, result)
    print(f"PROOF:run:ensemble.policy:{policy_receipt}")

    return result


def cmd_rebalance(args):
    """Run portfolio rebalancing"""
    print("Running portfolio rebalancing...")

    # First, get current weights from governance (or use provided)
    if hasattr(args, 'weights_file') and args.weights_file:
        import json
        with open(args.weights_file, 'r') as f:
            weights_data = json.load(f)
            target_weights = weights_data.get('final_weights', weights_data.get('weights', {}))
    else:
        # Run governance to get target weights
        gov_result = govern_ensemble(
            approved_path="artifacts/fixtures/phase9/approved_bundles.json",
            corr_path="artifacts/fixtures/phase9/pairwise_corr.csv",
            factors_path="artifacts/fixtures/phase9/factor_exposures.csv",
            policy_path=args.policy,
            asof=args.asof if hasattr(args, 'asof') else None,
            live=args.live
        )
        target_weights = gov_result['final_weights']

    # Apply ensemble operations
    result = ensemble_ops_apply(
        target_weights=target_weights,
        last_weights_path=args.last if hasattr(args, 'last') else "artifacts/fixtures/phase9/last_weights.json",
        cost_model_path=args.cost_model if hasattr(args, 'cost_model') else "artifacts/fixtures/phase9/cost_model.yaml",
        policy_path=args.policy,
        backend=args.backend,
        live=args.live
    )

    if result.get('turnover_ok', False):
        print(f"✅ Rebalance: OK (turnover: {result.get('turnover', 0):.3f})")
    else:
        if 'error' in result:
            print(f"❌ Rebalance: ERROR ({result['error']})")
        else:
            print(f"❌ Rebalance: FAILED (turnover: {result.get('turnover', 0):.3f})")

    print(f"Orders: {result.get('orders_generated', 0)} generated, {result.get('orders_filled', 0)} filled")
    if 'cost_analysis' in result:
        print(f"Cost: {result['cost_analysis']['total_cost_bps']:.2f} bps")
    else:
        print("Cost: N/A (error occurred)")

    # Generate PROOF lines for different tool names based on backend
    if args.backend == "simulator":
        proof_tool = "portfolio.rebalance@orders_sim"
    elif args.backend == "qc_paper":
        proof_tool = "portfolio.rebalance@orders_qc_paper"
    else:
        proof_tool = "portfolio.rebalance@orders"

    print(f"PROOF:run:{proof_tool}:{result['receipt_hash']}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Ally Portfolio CLI')
    parser.add_argument('--policy', default='ally/ops/policy.yaml', help='Policy file path')
    parser.add_argument('--live', action='store_true', help='Enable live mode')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # portfolio govern
    govern_parser = subparsers.add_parser('govern', help='Ensemble governance')
    govern_parser.add_argument('--approved', help='Approved bundles JSON')
    govern_parser.add_argument('--corr', help='Correlation matrix CSV')
    govern_parser.add_argument('--factors', help='Factor exposures CSV')
    govern_parser.add_argument('--snapshot', help='Sentinel snapshot JSON')
    govern_parser.add_argument('--asof', help='As-of timestamp (ISO format)')

    # portfolio rebalance
    rebalance_parser = subparsers.add_parser('rebalance', help='Portfolio rebalancing')
    rebalance_parser.add_argument('--backend', default='simulator', choices=['simulator', 'qc_paper'], help='Execution backend')
    rebalance_parser.add_argument('--last', help='Last weights JSON')
    rebalance_parser.add_argument('--cost-model', help='Cost model YAML')
    rebalance_parser.add_argument('--weights-file', help='Target weights file (skip governance)')
    rebalance_parser.add_argument('--asof', help='As-of timestamp (ISO format)')

    args = parser.parse_args()

    if args.command == 'govern':
        return cmd_govern(args)
    elif args.command == 'rebalance':
        return cmd_rebalance(args)
    else:
        parser.print_help()
        return None


if __name__ == '__main__':
    main()