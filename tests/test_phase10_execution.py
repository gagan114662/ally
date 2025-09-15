#!/usr/bin/env python3
"""
Phase 10: Comprehensive Execution Test Suite
Tests for deterministic execution, slippage models, order lifecycle, and CLI integration
"""

import unittest
import json
import os
import sys
import tempfile
import csv
from pathlib import Path

# Add execution path
sys.path.append('./ally/ally/ally')

class TestSlippageModels(unittest.TestCase):
    """Test slippage and latency models"""

    def setUp(self):
        """Set up test fixtures"""
        from slippage import LinearSlippage, SquareRootSlippage, LatencyModel

        self.test_order = {
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 1000,
            'notional_usd': 176000
        }

        self.test_market = {
            'adv_usd': 50_000_000,
            'bid': 175.90,
            'ask': 176.10,
            'mid': 176.00
        }

    def test_linear_slippage_deterministic(self):
        """Test linear slippage model with deterministic behavior"""
        from slippage import LinearSlippage

        # Create two models with same seed
        model1 = LinearSlippage(base_bps=10, linear_factor=50, seed=42)
        model2 = LinearSlippage(base_bps=10, linear_factor=50, seed=42)

        # Should produce identical results
        slip1 = model1.calculate_slippage(self.test_order, self.test_market)
        slip2 = model2.calculate_slippage(self.test_order, self.test_market)

        self.assertEqual(slip1, slip2, "Linear slippage should be deterministic with same seed")
        self.assertGreater(slip1, 0, "Slippage should be positive")

    def test_sqrt_slippage_deterministic(self):
        """Test square-root slippage model with deterministic behavior"""
        from slippage import SquareRootSlippage

        # Create two models with same seed
        model1 = SquareRootSlippage(base_bps=10, impact_k=6.0, seed=42)
        model2 = SquareRootSlippage(base_bps=10, impact_k=6.0, seed=42)

        # Should produce identical results
        slip1 = model1.calculate_slippage(self.test_order, self.test_market)
        slip2 = model2.calculate_slippage(self.test_order, self.test_market)

        self.assertEqual(slip1, slip2, "Square-root slippage should be deterministic with same seed")
        self.assertGreater(slip1, 0, "Slippage should be positive")

    def test_latency_model(self):
        """Test latency model execution"""
        from slippage import LatencyModel

        prices = [
            {'timestamp': '2025-09-16T09:30:00Z', 'bid': 176.00, 'ask': 176.02},
            {'timestamp': '2025-09-16T09:31:00Z', 'bid': 176.05, 'ask': 176.07},
            {'timestamp': '2025-09-16T09:32:00Z', 'bid': 176.10, 'ask': 176.12}
        ]

        # Test immediate execution
        latency_model = LatencyModel(latency_bars=0)
        price, timestamp = latency_model.get_execution_price(prices, 0, 'BUY')
        self.assertEqual(price, 176.02, "Should use ask price for BUY orders")

        # Test with latency
        latency_model = LatencyModel(latency_bars=1)
        price, timestamp = latency_model.get_execution_price(prices, 0, 'BUY')
        self.assertEqual(price, 176.07, "Should use ask price from next bar with latency")

    def test_adv_constraint_checking(self):
        """Test ADV constraint validation"""
        from slippage import check_adv_constraint, split_order_for_adv

        # Order within ADV limit
        small_order = {'notional_usd': 1_000_000}  # 2% of 50M ADV
        can_execute, reason = check_adv_constraint(small_order, self.test_market, 0.10)
        self.assertTrue(can_execute, "Small order should pass ADV constraint")

        # Order exceeding ADV limit
        large_order = {'notional_usd': 10_000_000, 'side': 'BUY', 'quantity': 1000}  # 20% of 50M ADV
        can_execute, reason = check_adv_constraint(large_order, self.test_market, 0.10)
        self.assertFalse(can_execute, "Large order should fail ADV constraint")

        # Test order splitting
        slices = split_order_for_adv(large_order, self.test_market, 0.10)
        self.assertGreater(len(slices), 1, "Large order should be split into multiple slices")


class TestOrderLifecycle(unittest.TestCase):
    """Test order lifecycle and journaling"""

    def setUp(self):
        """Set up test environment"""
        from orders import Order, OrderJournal, TradeJournal
        import tempfile

        # Create temporary journal files
        self.temp_dir = tempfile.mkdtemp()
        self.orders_file = os.path.join(self.temp_dir, "orders.jsonl")
        self.trades_file = os.path.join(self.temp_dir, "trades.jsonl")

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_order_creation_and_fills(self):
        """Test order creation and fill lifecycle"""
        from orders import Order, OrderStatus

        order = Order(
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            order_type="MARKET",
            strategy="TEST"
        )

        # Test initial state
        self.assertEqual(order.status, OrderStatus.NEW)
        self.assertEqual(order.filled_quantity, 0.0)
        self.assertEqual(len(order.fills), 0)

        # Test partial fill
        fill = order.add_fill(600, 176.05, "2025-09-16T09:30:00Z")
        self.assertEqual(order.status, OrderStatus.PARTIALLY_FILLED)
        self.assertEqual(order.filled_quantity, 600)
        self.assertEqual(len(order.fills), 1)

        # Test complete fill
        order.add_fill(400, 176.10, "2025-09-16T09:30:30Z")
        self.assertEqual(order.status, OrderStatus.FILLED)
        self.assertEqual(order.filled_quantity, 1000)
        self.assertGreater(order.avg_fill_price, 0)

    def test_order_cancellation(self):
        """Test order cancellation"""
        from orders import Order, OrderStatus

        order = Order("AAPL", "BUY", 1000)

        # Should be able to cancel new order
        success = order.cancel("User requested")
        self.assertTrue(success)
        self.assertEqual(order.status, OrderStatus.CANCELED)

        # Should not be able to cancel again
        success = order.cancel("Already canceled")
        self.assertFalse(success)

    def test_order_journaling(self):
        """Test order journaling to JSONL"""
        from orders import Order, OrderJournal

        journal = OrderJournal(self.orders_file)

        order = Order("AAPL", "BUY", 1000, strategy="TEST")
        order_id = journal.add_order(order)

        # Update with fill
        journal.update_order(order_id, "FILL", {
            'quantity': 1000,
            'price': 176.05,
            'timestamp': '2025-09-16T09:30:00Z'
        })

        # Check journal file was created and contains data
        self.assertTrue(os.path.exists(self.orders_file))

        with open(self.orders_file, 'r') as f:
            lines = f.readlines()

        self.assertGreater(len(lines), 0, "Journal should contain entries")

        # Verify JSON format
        for line in lines:
            if line.strip():
                event = json.loads(line)
                self.assertIn('event', event)
                self.assertIn('timestamp', event)

    def test_trade_journaling(self):
        """Test trade journaling"""
        from orders import TradeJournal

        journal = TradeJournal(self.trades_file)

        trade = {
            'order_id': 'ORD_123456789012',
            'symbol': 'AAPL',
            'side': 'BUY',
            'quantity': 1000,
            'price': 176.05,
            'notional': 176050,
            'slippage_bps': 5.0
        }

        trade_id = journal.add_trade(trade)
        self.assertIsNotNone(trade_id)

        # Check journal file
        self.assertTrue(os.path.exists(self.trades_file))

        summary = journal.get_summary()
        self.assertEqual(summary['total_trades'], 1)
        self.assertGreater(summary['total_notional'], 0)


class TestExecutionSimulator(unittest.TestCase):
    """Test execution simulator backend"""

    def setUp(self):
        """Set up test fixtures"""
        # Ensure fixture files exist
        self.fixture_files = {
            'target_weights': 'artifacts/fixtures/phase10/target_weights.json',
            'positions': 'artifacts/fixtures/phase10/last_positions.json',
            'symbols': 'artifacts/fixtures/phase10/symbols_meta.csv',
            'prices': 'artifacts/fixtures/phase10/prices_intraday.csv',
            'cost_model': 'artifacts/fixtures/phase10/cost_model.yaml'
        }

        for name, path in self.fixture_files.items():
            self.assertTrue(os.path.exists(path), f"Fixture file missing: {path}")

    def test_simulator_deterministic_behavior(self):
        """Test that simulator produces identical results with same seed"""
        from execution.backends.simulator import simulate_execution

        # Run simulation twice with same parameters
        result1 = simulate_execution(
            target_weights_path=self.fixture_files['target_weights'],
            last_positions_path=self.fixture_files['positions'],
            symbols_path=self.fixture_files['symbols'],
            prices_path=self.fixture_files['prices'],
            cost_model_path=self.fixture_files['cost_model'],
            slippage_model='sqrt',
            impact_k=6.0,
            latency_bars=0,
            live=False
        )

        result2 = simulate_execution(
            target_weights_path=self.fixture_files['target_weights'],
            last_positions_path=self.fixture_files['positions'],
            symbols_path=self.fixture_files['symbols'],
            prices_path=self.fixture_files['prices'],
            cost_model_path=self.fixture_files['cost_model'],
            slippage_model='sqrt',
            impact_k=6.0,
            latency_bars=0,
            live=False
        )

        # Results should be identical for deterministic execution
        self.assertEqual(result1['orders_placed'], result2['orders_placed'])
        self.assertEqual(result1['orders_filled'], result2['orders_filled'])
        self.assertEqual(result1['avg_slippage_bps'], result2['avg_slippage_bps'])

    def test_simulator_receipt_generation(self):
        """Test that simulator generates proper receipts"""
        from execution.backends.simulator import simulate_execution

        result = simulate_execution(
            target_weights_path=self.fixture_files['target_weights'],
            last_positions_path=self.fixture_files['positions'],
            symbols_path=self.fixture_files['symbols'],
            prices_path=self.fixture_files['prices'],
            cost_model_path=self.fixture_files['cost_model'],
            slippage_model='sqrt',
            impact_k=6.0,
            latency_bars=0,
            live=False
        )

        # Check receipts are generated
        self.assertIn('orders_receipt', result)
        self.assertIn('trades_receipt', result)
        self.assertIsNotNone(result['orders_receipt'])
        self.assertIsNotNone(result['trades_receipt'])

        # Check receipt format (16 char hex)
        orders_receipt = result['orders_receipt']
        trades_receipt = result['trades_receipt']

        self.assertEqual(len(orders_receipt), 16, "Orders receipt should be 16 characters")
        self.assertEqual(len(trades_receipt), 16, "Trades receipt should be 16 characters")

        # Should be valid hex
        int(orders_receipt, 16)
        int(trades_receipt, 16)

    def test_simulator_slippage_models(self):
        """Test simulator with different slippage models"""
        from execution.backends.simulator import simulate_execution

        # Test linear slippage
        result_linear = simulate_execution(
            target_weights_path=self.fixture_files['target_weights'],
            last_positions_path=self.fixture_files['positions'],
            symbols_path=self.fixture_files['symbols'],
            prices_path=self.fixture_files['prices'],
            cost_model_path=self.fixture_files['cost_model'],
            slippage_model='linear',
            impact_k=6.0,
            latency_bars=0,
            live=False
        )

        # Test square-root slippage
        result_sqrt = simulate_execution(
            target_weights_path=self.fixture_files['target_weights'],
            last_positions_path=self.fixture_files['positions'],
            symbols_path=self.fixture_files['symbols'],
            prices_path=self.fixture_files['prices'],
            cost_model_path=self.fixture_files['cost_model'],
            slippage_model='sqrt',
            impact_k=6.0,
            latency_bars=0,
            live=False
        )

        # Both should succeed and produce results
        self.assertGreater(result_linear['orders_placed'], 0)
        self.assertGreater(result_sqrt['orders_placed'], 0)

        # Slippage values should be different between models
        self.assertNotEqual(result_linear['avg_slippage_bps'], result_sqrt['avg_slippage_bps'])

    def test_kill_switch_functionality(self):
        """Test kill switch functionality"""
        from execution.backends.simulator import SimulatorBackend

        simulator = SimulatorBackend()
        result = simulator.kill_all_orders()

        # Should complete successfully even with no orders
        self.assertIn('canceled_count', result)
        self.assertIn('status', result)
        self.assertIn('receipt_hash', result)
        self.assertEqual(result['status'], 'KILL_SWITCH_ACTIVATED')


class TestFixtureIntegrity(unittest.TestCase):
    """Test fixture file integrity"""

    def test_target_weights_format(self):
        """Test target weights JSON format"""
        with open('artifacts/fixtures/phase10/target_weights.json', 'r') as f:
            data = json.load(f)

        self.assertIn('weights', data)
        self.assertIn('notional_usd', data)

        weights = data['weights']
        self.assertIsInstance(weights, dict)
        self.assertGreater(len(weights), 0)

        # Check weight values are numeric
        for symbol, weight in weights.items():
            self.assertIsInstance(weight, (int, float))

    def test_symbols_metadata_format(self):
        """Test symbols metadata CSV format"""
        with open('artifacts/fixtures/phase10/symbols_meta.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertGreater(len(rows), 0, "Symbols metadata should not be empty")

        required_cols = ['symbol', 'tick_size', 'lot_size', 'adv_usd']
        for col in required_cols:
            self.assertIn(col, rows[0], f"Missing required column: {col}")

        # Check data types
        for row in rows:
            self.assertIsNotNone(row['symbol'])
            float(row['tick_size'])  # Should be convertible to float
            int(row['lot_size'])     # Should be convertible to int
            float(row['adv_usd'])    # Should be convertible to float

    def test_prices_data_format(self):
        """Test price data CSV format"""
        with open('artifacts/fixtures/phase10/prices_intraday.csv', 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertGreater(len(rows), 0, "Price data should not be empty")

        required_cols = ['timestamp', 'symbol', 'bid', 'ask', 'mid', 'volume']
        for col in required_cols:
            self.assertIn(col, rows[0], f"Missing required column: {col}")

        # Check data consistency
        for row in rows:
            bid = float(row['bid'])
            ask = float(row['ask'])
            mid = float(row['mid'])

            self.assertLess(bid, ask, "Bid should be less than ask")
            self.assertGreaterEqual(mid, bid, "Mid should be >= bid")
            self.assertLessEqual(mid, ask, "Mid should be <= ask")


if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)

    # Configure test runner
    unittest.main(verbosity=2)