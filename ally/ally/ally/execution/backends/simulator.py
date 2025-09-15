#!/usr/bin/env python3
"""
Phase 10: Execution Simulator Backend
Deterministic order execution simulation with slippage, latency, and ADV constraints
"""

import json
import csv
import yaml
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Local imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slippage import (
    LinearSlippage, SquareRootSlippage, LatencyModel,
    apply_slippage_to_price, calculate_execution_cost,
    check_adv_constraint, split_order_for_adv
)
from orders import Order, OrderJournal, TradeJournal

# Receipts system
try:
    from ...ops.receipts import write_tool_receipt
except ImportError:
    import hashlib
    def write_tool_receipt(tool_name: str, params: dict, status: str, result_data: dict = None):
        data = {'tool': tool_name, 'params': params, 'status': status}
        receipt_hash = hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
        print(f"RECEIPT: {tool_name}:{receipt_hash}")
        return receipt_hash


class SimulatorBackend:
    """Deterministic execution simulator"""

    def __init__(self, seed: int = 42):
        """Initialize simulator"""
        self.seed = seed
        random.seed(seed)

        # Journals
        self.order_journal = OrderJournal()
        self.trade_journal = TradeJournal()

        # Market data cache
        self.symbols_meta = {}
        self.price_data = {}

        # Configuration
        self.config = {
            'slippage_model': 'sqrt',
            'impact_k': 6.0,
            'base_bps': 10,
            'latency_bars': 0,
            'adv_cap_pct': 0.10
        }

    def load_market_data(self, symbols_path: str, prices_path: str):
        """Load market data from CSV files"""
        # Load symbol metadata
        with open(symbols_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.symbols_meta[row['symbol']] = {
                    'tick_size': float(row['tick_size']),
                    'lot_size': int(row['lot_size']),
                    'adv_usd': float(row['adv_usd'])
                }

        # Load price data
        with open(prices_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row['symbol']
                if symbol not in self.price_data:
                    self.price_data[symbol] = []

                self.price_data[symbol].append({
                    'timestamp': row['timestamp'],
                    'bid': float(row['bid']),
                    'ask': float(row['ask']),
                    'mid': float(row['mid']),
                    'volume': int(row['volume'])
                })

    def load_cost_model(self, cost_model_path: str):
        """Load cost model configuration"""
        with open(cost_model_path, 'r') as f:
            cost_config = yaml.safe_load(f)

        # Parse model string
        model_str = cost_config.get('model', '')
        if 'bps=' in model_str:
            self.config['base_bps'] = float(model_str.split('bps=')[1].split(',')[0])
        if 'k=' in model_str:
            self.config['impact_k'] = float(model_str.split('k=')[1].split(',')[0])

        self.config['adv_cap_pct'] = cost_config.get('adv_cap_pct', 0.10)

    def set_slippage_config(self, slippage_model: str = 'sqrt', impact_k: float = 6.0,
                           latency_bars: int = 0):
        """Configure slippage and latency"""
        self.config['slippage_model'] = slippage_model
        self.config['impact_k'] = impact_k
        self.config['latency_bars'] = latency_bars

    def calculate_target_orders(self, target_weights: dict, last_positions: dict,
                              portfolio_value: float) -> List[dict]:
        """Calculate orders needed to reach target weights"""
        orders = []

        for symbol, target_weight in target_weights.items():
            target_value = target_weight * portfolio_value

            # Get current position
            current_pos = last_positions.get('positions', {}).get(symbol, {})
            current_qty = current_pos.get('quantity', 0)

            # Get current price (use mid)
            if symbol in self.price_data and self.price_data[symbol]:
                current_price = self.price_data[symbol][0]['mid']
            else:
                continue  # Skip if no price data

            # Calculate target quantity
            target_qty = target_value / current_price

            # Calculate order quantity
            order_qty = target_qty - current_qty

            if abs(order_qty) > 0.01:  # Minimum order size
                order = {
                    'symbol': symbol,
                    'side': 'BUY' if order_qty > 0 else 'SELL',
                    'quantity': abs(order_qty),
                    'notional_usd': abs(order_qty * current_price),
                    'current_qty': current_qty,
                    'target_qty': target_qty,
                    'strategy': 'REBALANCE'
                }
                orders.append(order)

        return orders

    def execute_order(self, order_dict: dict, bar_index: int = 0) -> Tuple[Order, List[dict]]:
        """Execute a single order with slippage and constraints"""

        # Create Order object
        order = Order(
            symbol=order_dict['symbol'],
            side=order_dict['side'],
            quantity=order_dict['quantity'],
            order_type='MARKET',
            strategy=order_dict.get('strategy', 'UNKNOWN')
        )

        # Add to journal
        self.order_journal.add_order(order)

        # Get market data
        symbol_meta = self.symbols_meta.get(order.symbol, {})
        prices = self.price_data.get(order.symbol, [])

        if not prices or bar_index >= len(prices):
            order.reject("No market data available")
            self.order_journal.update_order(order.order_id, "REJECT", {"reason": "No market data"})
            return order, []

        # Check ADV constraint
        can_execute, reason = check_adv_constraint(order_dict, symbol_meta, self.config['adv_cap_pct'])

        if not can_execute:
            # Try to split order
            order_slices = split_order_for_adv(order_dict, symbol_meta, self.config['adv_cap_pct'])

            if len(order_slices) > 1:
                # For now, just execute first slice and mark as partial
                order_dict = order_slices[0]
                order.quantity = order_dict['quantity']
            else:
                order.reject(reason)
                self.order_journal.update_order(order.order_id, "REJECT", {"reason": reason})
                return order, []

        # Create slippage model
        if self.config['slippage_model'] == 'linear':
            slippage_model = LinearSlippage(
                base_bps=self.config['base_bps'],
                linear_factor=50,
                seed=self.seed
            )
        else:  # sqrt
            slippage_model = SquareRootSlippage(
                base_bps=self.config['base_bps'],
                impact_k=self.config['impact_k'],
                seed=self.seed
            )

        # Create latency model
        latency_model = LatencyModel(latency_bars=self.config['latency_bars'])

        # Calculate slippage
        slippage_bps = slippage_model.calculate_slippage(order_dict, symbol_meta)

        # Get execution price with latency
        base_price, exec_timestamp = latency_model.get_execution_price(
            prices, bar_index, order.side
        )

        # Apply slippage
        fill_price = apply_slippage_to_price(base_price, slippage_bps, order.side)

        # Create fill
        fill_data = {
            'quantity': order.quantity,
            'price': fill_price,
            'timestamp': exec_timestamp,
            'venue': 'SIMULATOR'
        }

        # Update order with fill
        self.order_journal.update_order(order.order_id, "FILL", fill_data)

        # Record trade
        trade = {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'price': fill_price,
            'notional': abs(order.quantity * fill_price),
            'slippage_bps': slippage_bps,
            'latency_bars': self.config['latency_bars'],
            'timestamp': exec_timestamp
        }
        self.trade_journal.add_trade(trade)

        # Calculate execution cost
        cost_metrics = calculate_execution_cost(
            order_dict, fill_price, prices[bar_index]['mid']
        )

        return order, [trade]

    def execute_portfolio_rebalance(self, target_weights: dict, last_positions: dict,
                                   portfolio_value: float = 1_000_000) -> dict:
        """Execute full portfolio rebalance"""

        # Calculate target orders
        orders = self.calculate_target_orders(target_weights, last_positions, portfolio_value)

        # Execute orders
        executed_orders = []
        all_trades = []

        for i, order_dict in enumerate(orders):
            order, trades = self.execute_order(order_dict, bar_index=i % 3)  # Cycle through price bars
            executed_orders.append(order)
            all_trades.extend(trades)

        # Calculate summary statistics
        total_slippage = sum(t['slippage_bps'] for t in all_trades) / len(all_trades) if all_trades else 0
        total_notional = sum(t['notional'] for t in all_trades)

        result = {
            'timestamp': datetime.now().isoformat() + 'Z',
            'orders_placed': len(orders),
            'orders_filled': len([o for o in executed_orders if o.status.value == 'FILLED']),
            'total_trades': len(all_trades),
            'total_notional': total_notional,
            'avg_slippage_bps': total_slippage,
            'order_summary': self.order_journal.get_summary(),
            'trade_summary': self.trade_journal.get_summary(),
            'config': self.config
        }

        return result

    def kill_all_orders(self) -> dict:
        """Kill switch - cancel all open orders"""
        canceled = self.order_journal.cancel_all_open("Kill switch activated")

        result = {
            'timestamp': datetime.now().isoformat() + 'Z',
            'canceled_orders': canceled,
            'canceled_count': len(canceled),
            'open_order_count': len(self.order_journal.get_open_orders()),
            'status': 'KILL_SWITCH_ACTIVATED'
        }

        # Write kill receipt
        receipt_hash = write_tool_receipt(
            "execution.kill",
            {"backend": "simulator"},
            "EXECUTED",
            result
        )

        result['receipt_hash'] = receipt_hash
        return result


def simulate_execution(
    target_weights_path: str,
    last_positions_path: str,
    symbols_path: str,
    prices_path: str,
    cost_model_path: str,
    slippage_model: str = 'sqrt',
    impact_k: float = 6.0,
    latency_bars: int = 0,
    live: bool = False
) -> dict:
    """Main simulation entry point"""

    try:
        # Initialize simulator
        simulator = SimulatorBackend(seed=42)

        # Load market data
        simulator.load_market_data(symbols_path, prices_path)
        simulator.load_cost_model(cost_model_path)

        # Configure slippage
        simulator.set_slippage_config(
            slippage_model=slippage_model,
            impact_k=impact_k,
            latency_bars=latency_bars
        )

        # Load targets and positions
        with open(target_weights_path, 'r') as f:
            target_data = json.load(f)
        with open(last_positions_path, 'r') as f:
            positions_data = json.load(f)

        # Execute rebalance
        result = simulator.execute_portfolio_rebalance(
            target_weights=target_data['weights'],
            last_positions=positions_data,
            portfolio_value=target_data.get('notional_usd', 1_000_000)
        )

        # Write receipts
        params = {
            'target_weights_path': target_weights_path,
            'last_positions_path': last_positions_path,
            'slippage_model': slippage_model,
            'impact_k': impact_k,
            'latency_bars': latency_bars,
            'live': live
        }

        # Orders receipt
        orders_receipt = write_tool_receipt(
            "execution.simulator.orders",
            params,
            "OK",
            result
        )

        # Trades receipt
        trades_receipt = write_tool_receipt(
            "execution.simulator.trades",
            params,
            "OK",
            result
        )

        result['orders_receipt'] = orders_receipt
        result['trades_receipt'] = trades_receipt

        # Write result artifact
        os.makedirs("artifacts/execution", exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%dT%H%M%S")
        with open(f"artifacts/execution/simulation_{timestamp_str}.json", 'w') as f:
            json.dump(result, f, indent=2)

        return result

    except Exception as e:
        error_result = {
            'timestamp': datetime.now().isoformat() + 'Z',
            'status': 'ERROR',
            'error': str(e)
        }

        receipt_hash = write_tool_receipt(
            "execution.simulator.orders",
            {},
            "ERROR",
            error_result
        )

        error_result['receipt_hash'] = receipt_hash
        return error_result


if __name__ == "__main__":
    # Test simulator
    result = simulate_execution(
        target_weights_path="artifacts/fixtures/phase10/target_weights.json",
        last_positions_path="artifacts/fixtures/phase10/last_positions.json",
        symbols_path="artifacts/fixtures/phase10/symbols_meta.csv",
        prices_path="artifacts/fixtures/phase10/prices_intraday.csv",
        cost_model_path="artifacts/fixtures/phase10/cost_model.yaml",
        slippage_model='sqrt',
        impact_k=6.0,
        latency_bars=0,
        live=False
    )

    print(f"Simulation complete:")
    print(f"  Orders placed: {result.get('orders_placed', 0)}")
    print(f"  Orders filled: {result.get('orders_filled', 0)}")
    print(f"  Avg slippage: {result.get('avg_slippage_bps', 0):.2f} bps")
    print(f"  Orders receipt: {result.get('orders_receipt', 'N/A')}")
    print(f"  Trades receipt: {result.get('trades_receipt', 'N/A')}")