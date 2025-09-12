"""
Paper Trading Broker Adapter
In-memory broker with deterministic fills for simulation
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
from ...schemas.exec import PlaceOrderIn, CancelOrderIn, AmendOrderIn, ExecutionReport, Fill, OrderStatus


class PaperBroker:
    """
    In-memory paper trading broker with deterministic execution
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize paper broker
        
        Args:
            seed: Random seed for deterministic behavior
        """
        self.next_order_id = 1
        self.orders: Dict[str, Dict] = {}  # order_id -> order data
        self.seed = seed
        self.reset()
    
    def reset(self):
        """Reset broker state"""
        self.next_order_id = 1
        self.orders.clear()
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        order_id = f"OID-{self.next_order_id}"
        self.next_order_id += 1
        return order_id
    
    def _current_timestamp(self) -> str:
        """Generate current timestamp in ISO-Z format"""
        return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    def place_order(self, order_params: PlaceOrderIn) -> ExecutionReport:
        """
        Place an order and return execution report
        
        Args:
            order_params: Order placement parameters
            
        Returns:
            ExecutionReport with execution details
        """
        # Generate order ID
        broker_order_id = self._generate_order_id()
        
        # Create order record
        order = {
            'broker_order_id': broker_order_id,
            'client_order_id': order_params.client_order_id,
            'symbol': order_params.symbol,
            'side': order_params.side,
            'type': order_params.type,
            'original_qty': order_params.qty,
            'remaining_qty': order_params.qty,
            'filled_qty': 0.0,
            'limit_price': order_params.limit_price,
            'tif': order_params.tif,
            'status': 'accepted',
            'fills': [],
            'avg_price': 0.0,
            'slippage_bps': order_params.slippage_bps,
            'liquidity_per_tick': order_params.liquidity_per_tick
        }
        
        self.orders[broker_order_id] = order
        
        # Try to fill immediately if price is provided
        if order_params.price is not None:
            self._try_fill_order(broker_order_id, order_params.price)
        
        return self._create_execution_report(broker_order_id)
    
    def cancel_order(self, cancel_params: CancelOrderIn) -> ExecutionReport:
        """
        Cancel an order
        
        Args:
            cancel_params: Cancellation parameters
            
        Returns:
            ExecutionReport with updated status
        """
        order_id = cancel_params.broker_order_id
        
        if order_id not in self.orders:
            # Return error report for unknown order
            return ExecutionReport(
                broker_order_id=order_id,
                client_order_id=None,
                symbol="UNKNOWN",
                side="buy",
                type="market",
                status="rejected",
                avg_price=0.0,
                filled_qty=0.0,
                remaining_qty=0.0,
                fills=[],
                meta={"error": "Order not found"}
            )
        
        order = self.orders[order_id]
        
        # Can only cancel working or partially filled orders
        if order['status'] in ['working', 'partially_filled']:
            order['status'] = 'canceled'
        
        return self._create_execution_report(order_id)
    
    def amend_order(self, amend_params: AmendOrderIn) -> ExecutionReport:
        """
        Amend an order
        
        Args:
            amend_params: Amendment parameters
            
        Returns:
            ExecutionReport with updated order
        """
        order_id = amend_params.broker_order_id
        
        if order_id not in self.orders:
            return ExecutionReport(
                broker_order_id=order_id,
                client_order_id=None,
                symbol="UNKNOWN",
                side="buy",
                type="market",
                status="rejected",
                avg_price=0.0,
                filled_qty=0.0,
                remaining_qty=0.0,
                fills=[],
                meta={"error": "Order not found"}
            )
        
        order = self.orders[order_id]
        
        # Can only amend working or partially filled orders
        if order['status'] not in ['working', 'partially_filled']:
            return self._create_execution_report(order_id)
        
        # Update order parameters
        if amend_params.limit_price is not None:
            order['limit_price'] = amend_params.limit_price
        
        if amend_params.qty is not None:
            # Can only reduce quantity
            if amend_params.qty < order['original_qty']:
                new_remaining = max(0, amend_params.qty - order['filled_qty'])
                order['remaining_qty'] = new_remaining
                order['original_qty'] = amend_params.qty
                
                # If already filled more than new qty, mark as filled
                if order['filled_qty'] >= amend_params.qty:
                    order['status'] = 'filled'
                    order['remaining_qty'] = 0.0
        
        return self._create_execution_report(order_id)
    
    def _try_fill_order(self, order_id: str, market_price: float):
        """
        Try to fill an order given current market price
        
        Args:
            order_id: Order ID to fill
            market_price: Current market price
        """
        order = self.orders[order_id]
        
        if order['status'] in ['filled', 'canceled', 'rejected']:
            return
        
        if order['remaining_qty'] <= 0:
            order['status'] = 'filled'
            return
        
        # Check if order can fill
        can_fill = False
        fill_price = market_price
        
        if order['type'] == 'market':
            # Market orders always fill with slippage
            can_fill = True
            side_multiplier = 1 if order['side'] == 'buy' else -1
            slippage_factor = order['slippage_bps'] / 10000.0
            fill_price = market_price * (1 + side_multiplier * slippage_factor)
            
        elif order['type'] == 'limit':
            # Limit orders fill if price is favorable
            if order['side'] == 'buy' and market_price <= order['limit_price']:
                can_fill = True
                fill_price = min(market_price, order['limit_price'])
            elif order['side'] == 'sell' and market_price >= order['limit_price']:
                can_fill = True
                fill_price = max(market_price, order['limit_price'])
        
        if can_fill:
            # Calculate fill quantity (limited by liquidity_per_tick)
            max_fill_qty = min(order['remaining_qty'], order['liquidity_per_tick'])
            
            # Create fill
            fill = Fill(
                price=fill_price,
                qty=max_fill_qty,
                ts=self._current_timestamp()
            )
            
            order['fills'].append(fill)
            order['filled_qty'] += max_fill_qty
            order['remaining_qty'] -= max_fill_qty
            
            # Update average price
            total_notional = sum(f.price * f.qty for f in order['fills'])
            order['avg_price'] = total_notional / order['filled_qty']
            
            # Update status
            if order['remaining_qty'] <= 0:
                order['status'] = 'filled'
            elif order['filled_qty'] > 0:
                order['status'] = 'partially_filled'
            else:
                order['status'] = 'working'
    
    def _create_execution_report(self, order_id: str) -> ExecutionReport:
        """
        Create execution report for an order
        
        Args:
            order_id: Order ID
            
        Returns:
            ExecutionReport
        """
        order = self.orders[order_id]
        
        return ExecutionReport(
            broker_order_id=order['broker_order_id'],
            client_order_id=order['client_order_id'],
            symbol=order['symbol'],
            side=order['side'],
            type=order['type'],
            status=order['status'],
            avg_price=order['avg_price'],
            filled_qty=order['filled_qty'],
            remaining_qty=order['remaining_qty'],
            fills=order['fills'],
            meta={
                'original_qty': order['original_qty'],
                'limit_price': order['limit_price'],
                'tif': order['tif']
            }
        )
    
    def mark_to_market(self, prices: Dict[str, float]):
        """
        Process market prices and attempt fills
        
        Args:
            prices: Symbol to price mapping
        """
        for order_id, order in self.orders.items():
            symbol = order['symbol']
            if symbol in prices and order['status'] in ['working', 'partially_filled']:
                self._try_fill_order(order_id, prices[symbol])


# Global paper broker instance for tools to use
_paper_broker = PaperBroker(seed=42)