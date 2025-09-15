#!/usr/bin/env python3
"""
Phase 10: Slippage and Latency Models
Pure Python implementation for deterministic execution simulation
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple


class SlippageModel:
    """Base slippage model"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)  # Use dedicated RNG instance

    def calculate_slippage(self, order: dict, market_data: dict) -> float:
        """Calculate slippage in bps"""
        raise NotImplementedError


class LinearSlippage(SlippageModel):
    """Linear slippage model: slippage = base_bps + linear_factor * (size / adv)"""

    def __init__(self, base_bps: float = 10, linear_factor: float = 50, seed: int = 42):
        super().__init__(seed)
        self.base_bps = base_bps
        self.linear_factor = linear_factor

    def calculate_slippage(self, order: dict, market_data: dict) -> float:
        """Calculate linear slippage"""
        adv_usd = market_data.get('adv_usd', 10_000_000)
        order_size_usd = abs(order['notional_usd'])

        # Size as fraction of ADV
        size_impact = order_size_usd / adv_usd

        # Linear slippage
        slippage_bps = self.base_bps + self.linear_factor * size_impact

        # Add small random component for realism (deterministic with seed)
        noise = self._rng.gauss(0, 1)  # Deterministic due to dedicated RNG
        slippage_bps += noise

        return max(0, slippage_bps)


class SquareRootSlippage(SlippageModel):
    """Square-root impact model: slippage = base_bps + k * sqrt(size / adv)"""

    def __init__(self, base_bps: float = 10, impact_k: float = 6.0, seed: int = 42):
        super().__init__(seed)
        self.base_bps = base_bps
        self.impact_k = impact_k

    def calculate_slippage(self, order: dict, market_data: dict) -> float:
        """Calculate square-root slippage"""
        adv_usd = market_data.get('adv_usd', 10_000_000)
        order_size_usd = abs(order['notional_usd'])

        # Size as fraction of ADV
        size_impact = order_size_usd / adv_usd

        # Square-root impact
        slippage_bps = self.base_bps + self.impact_k * math.sqrt(size_impact) * 100

        # Add small random component (deterministic with seed)
        noise = self._rng.gauss(0, 0.5)
        slippage_bps += noise

        return max(0, slippage_bps)


class LatencyModel:
    """Model execution latency (delay between decision and fill)"""

    def __init__(self, latency_bars: int = 0):
        """
        Args:
            latency_bars: Number of bars of latency (0 = immediate)
        """
        self.latency_bars = latency_bars

    def get_execution_price(self, prices: List[dict], current_bar: int, side: str) -> Tuple[float, str]:
        """
        Get execution price accounting for latency

        Returns:
            (price, timestamp)
        """
        # Apply latency
        exec_bar = min(current_bar + self.latency_bars, len(prices) - 1)
        price_data = prices[exec_bar]

        # Use ask for buys, bid for sells
        if side == 'BUY':
            price = price_data['ask']
        else:
            price = price_data['bid']

        return price, price_data['timestamp']


def apply_slippage_to_price(base_price: float, slippage_bps: float, side: str) -> float:
    """Apply slippage to a base price"""
    slippage_factor = slippage_bps / 10000

    if side == 'BUY':
        # Buys execute at higher price (adverse)
        return base_price * (1 + slippage_factor)
    else:
        # Sells execute at lower price (adverse)
        return base_price * (1 - slippage_factor)


def calculate_execution_cost(order: dict, fill_price: float, base_price: float) -> dict:
    """Calculate execution cost metrics"""

    quantity = abs(order.get('quantity', 0))
    notional = quantity * fill_price

    # Calculate slippage cost
    if order['side'] == 'BUY':
        slippage_cost = (fill_price - base_price) * quantity
    else:
        slippage_cost = (base_price - fill_price) * quantity

    # Calculate cost in bps
    slippage_bps = (slippage_cost / (quantity * base_price)) * 10000 if quantity > 0 else 0

    return {
        'fill_price': fill_price,
        'base_price': base_price,
        'notional_usd': notional,
        'slippage_cost_usd': slippage_cost,
        'slippage_bps': slippage_bps,
        'quantity': quantity
    }


def check_adv_constraint(order: dict, market_data: dict, adv_cap_pct: float = 0.10) -> Tuple[bool, str]:
    """
    Check if order violates ADV constraints

    Returns:
        (can_execute, reason)
    """
    adv_usd = market_data.get('adv_usd', 10_000_000)
    order_size_usd = abs(order['notional_usd'])
    max_allowed = adv_usd * adv_cap_pct

    if order_size_usd > max_allowed:
        return False, f"Order size ${order_size_usd:.0f} exceeds {adv_cap_pct*100}% ADV (${max_allowed:.0f})"

    return True, "OK"


def split_order_for_adv(order: dict, market_data: dict, adv_cap_pct: float = 0.10) -> List[dict]:
    """Split large order to respect ADV constraints"""

    adv_usd = market_data.get('adv_usd', 10_000_000)
    max_slice = adv_usd * adv_cap_pct
    order_size_usd = abs(order['notional_usd'])

    if order_size_usd <= max_slice:
        return [order]

    # Calculate number of slices needed
    n_slices = math.ceil(order_size_usd / max_slice)
    slice_size = order_size_usd / n_slices

    slices = []
    remaining_qty = abs(order.get('quantity', 0))

    for i in range(n_slices):
        slice_qty = remaining_qty / (n_slices - i)
        slice_order = order.copy()
        slice_order['quantity'] = slice_qty if order['side'] == 'BUY' else -slice_qty
        slice_order['notional_usd'] = slice_size
        slice_order['slice_id'] = i + 1
        slice_order['total_slices'] = n_slices

        slices.append(slice_order)
        remaining_qty -= slice_qty

    return slices


if __name__ == "__main__":
    # Test slippage models
    test_order = {
        'symbol': 'AAPL',
        'side': 'BUY',
        'quantity': 1000,
        'notional_usd': 176000
    }

    test_market = {
        'adv_usd': 50_000_000,
        'bid': 175.90,
        'ask': 176.10,
        'mid': 176.00
    }

    # Test linear slippage
    linear = LinearSlippage(base_bps=10, linear_factor=50)
    linear_slip = linear.calculate_slippage(test_order, test_market)
    print(f"Linear slippage: {linear_slip:.2f} bps")

    # Test square-root slippage
    sqrt_model = SquareRootSlippage(base_bps=10, impact_k=6.0)
    sqrt_slip = sqrt_model.calculate_slippage(test_order, test_market)
    print(f"Square-root slippage: {sqrt_slip:.2f} bps")

    # Apply slippage to price
    exec_price = apply_slippage_to_price(test_market['ask'], sqrt_slip, 'BUY')
    print(f"Execution price: ${exec_price:.2f} (base: ${test_market['ask']:.2f})")

    # Calculate execution cost
    test_order['quantity'] = 1000
    cost_metrics = calculate_execution_cost(test_order, exec_price, test_market['mid'])
    print(f"Execution cost: ${cost_metrics['slippage_cost_usd']:.2f} ({cost_metrics['slippage_bps']:.2f} bps)")

    # Test ADV constraint
    can_exec, reason = check_adv_constraint(test_order, test_market, adv_cap_pct=0.01)
    print(f"ADV check: {can_exec} - {reason}")