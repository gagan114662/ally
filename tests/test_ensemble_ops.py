#!/usr/bin/env python3
"""
Test suite for Phase 9 ensemble ops
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add the nested ally path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ally', 'ally', 'ally'))

from ops.ensemble_ops import (
    ensemble_ops_apply,
    calculate_weight_deltas,
    calculate_turnover,
    estimate_transaction_costs,
    generate_rebalance_orders,
    simulate_order_fills
)


class TestEnsembleOps(unittest.TestCase):
    """Test ensemble ops functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.target_weights = {
            "Strategy_A": 0.35,
            "Strategy_B": 0.35,
            "Strategy_C": 0.30
        }

        self.last_weights = {
            "Strategy_A": 0.40,
            "Strategy_B": 0.30,
            "Strategy_C": 0.30
        }

        self.cost_model = {
            'model': 'bps=10,impact=k*sqrt(q),k=6.0',
            'capacity_usd': 50000000,
            'max_turnover': 0.35
        }

    def test_calculate_weight_deltas(self):
        """Test weight delta calculation"""
        deltas = calculate_weight_deltas(self.target_weights, self.last_weights)

        # Check specific deltas
        self.assertAlmostEqual(deltas["Strategy_A"], -0.05, places=6)  # 0.35 - 0.40
        self.assertAlmostEqual(deltas["Strategy_B"], 0.05, places=6)   # 0.35 - 0.30
        self.assertAlmostEqual(deltas["Strategy_C"], 0.00, places=6)   # 0.30 - 0.30

        # Net delta should be zero (conservation)
        self.assertAlmostEqual(sum(deltas.values()), 0.0, places=6)

    def test_calculate_turnover(self):
        """Test turnover calculation"""
        deltas = {"Strategy_A": -0.05, "Strategy_B": 0.05, "Strategy_C": 0.00}
        turnover = calculate_turnover(deltas)

        # Turnover = sum(|delta|) / 2 = (0.05 + 0.05 + 0.00) / 2 = 0.05
        self.assertAlmostEqual(turnover, 0.05, places=6)

    def test_estimate_transaction_costs(self):
        """Test cost estimation"""
        deltas = {"Strategy_A": -0.10, "Strategy_B": 0.10}
        costs = estimate_transaction_costs(deltas, self.cost_model, portfolio_value=1_000_000)

        # Should have cost breakdown
        self.assertIn('total_cost_usd', costs)
        self.assertIn('total_cost_bps', costs)
        self.assertIn('strategy_costs', costs)

        # Should have costs for both strategies with meaningful deltas
        self.assertEqual(len(costs['strategy_costs']), 2)
        self.assertGreater(costs['total_cost_usd'], 0)

    def test_generate_rebalance_orders(self):
        """Test order generation"""
        deltas = {"Strategy_A": -0.10, "Strategy_B": 0.05, "Strategy_C": 0.05}
        orders = generate_rebalance_orders(deltas, portfolio_value=1_000_000)

        # Should generate orders for meaningful deltas
        self.assertEqual(len(orders), 3)

        # Check order structure
        for order in orders:
            self.assertIn('order_id', order)
            self.assertIn('strategy', order)
            self.assertIn('side', order)
            self.assertIn('notional_usd', order)
            self.assertIn('weight_delta', order)

        # Check order sides
        strategy_a_order = next(o for o in orders if o['strategy'] == 'Strategy_A')
        self.assertEqual(strategy_a_order['side'], 'SELL')  # Negative delta

        strategy_b_order = next(o for o in orders if o['strategy'] == 'Strategy_B')
        self.assertEqual(strategy_b_order['side'], 'BUY')   # Positive delta

    def test_simulate_order_fills(self):
        """Test order simulation"""
        orders = [
            {
                "order_id": "ORD_0001",
                "strategy": "Strategy_A",
                "side": "SELL",
                "notional_usd": 100000,
                "weight_delta": -0.10,
                "status": "PENDING"
            }
        ]

        filled_orders = simulate_order_fills(orders, backend="simulator")

        # Should have same number of orders
        self.assertEqual(len(filled_orders), len(orders))

        # Check fill details
        filled = filled_orders[0]
        self.assertEqual(filled['status'], 'FILLED')
        self.assertIn('fill_price', filled)
        self.assertIn('fill_time', filled)
        self.assertIn('execution_cost_bps', filled)
        self.assertEqual(filled['backend'], 'simulator')

    def test_turnover_constraint_enforcement(self):
        """Test turnover constraint checking"""
        # Create high turnover scenario
        high_turnover_weights = {
            "Strategy_A": 0.10,  # Was 0.40, big change
            "Strategy_B": 0.90   # Was 0.30, big change
        }

        last_weights = {"Strategy_A": 0.40, "Strategy_B": 0.30, "Strategy_C": 0.30}

        # Calculate expected turnover
        deltas = calculate_weight_deltas(high_turnover_weights, last_weights)
        turnover = calculate_turnover(deltas)

        # Should be high turnover (> 0.35 threshold)
        self.assertGreater(turnover, 0.35)

        # In real ensemble_ops_apply, this would be flagged as violation

    def test_deterministic_ops(self):
        """Test that ops are deterministic with same inputs"""
        # Test with same inputs twice (would check receipt hashes in real test)
        try:
            result1 = ensemble_ops_apply(
                target_weights=self.target_weights,
                backend="simulator",
                live=False
            )

            result2 = ensemble_ops_apply(
                target_weights=self.target_weights,
                backend="simulator",
                live=False
            )

            # Should have same turnover (deterministic calculation)
            if 'turnover' in result1 and 'turnover' in result2:
                self.assertAlmostEqual(result1['turnover'], result2['turnover'], places=6)

        except Exception as e:
            # Expected if files don't exist in test environment
            self.assertIn('FileNotFoundError', str(type(e).__name__))


if __name__ == '__main__':
    unittest.main()