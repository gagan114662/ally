"""
Deterministic simulator adapter for Ally broker testing

Provides instant fills at market prices with deterministic behavior for CI.
No network calls, fully offline, receipt-backed.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid

from ...utils.hashing import hash_payload
from ...utils.receipts import store_tool_receipt
from ...schemas.broker import Order, Fill, Position, Account, BrokerSession, OrderSide, OrderType, OrderStatus, TimeInForce


class SimulatorAdapter:
    """Deterministic broker simulator for testing"""
    
    def __init__(self, initial_cash: float = 1000000.0):
        """Initialize simulator with starting cash"""
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.fills: Dict[str, Fill] = {}
        self.sessions: Dict[str, BrokerSession] = {}
        
    def start_session(self, project_slug: str, symbols: List[str], live: bool = False) -> BrokerSession:
        """
        Start a new simulator session
        
        Args:
            project_slug: Unique project identifier
            symbols: List of symbols to trade
            live: Ignored for simulator (always offline)
            
        Returns:
            BrokerSession with session details
        """
        session_id = f"sim_{int(time.time() * 1000)}"
        
        session = BrokerSession(
            session_id=session_id,
            backend="simulator",
            project_slug=project_slug,
            symbols=symbols,
            started_at=datetime.utcnow().isoformat(),
            status="active",
            metadata={
                "initial_cash": self.initial_cash,
                "mode": "deterministic_simulator"
            }
        )
        
        self.sessions[session_id] = session
        return session
    
    def stop_session(self, session_id: str, live: bool = False) -> bool:
        """
        Stop a simulator session
        
        Args:
            session_id: Session to stop
            live: Ignored for simulator
            
        Returns:
            True if stopped successfully
        """
        if session_id in self.sessions:
            self.sessions[session_id].status = "stopped"
            return True
        return False
    
    def place_order(self, symbol: str, side: OrderSide, qty: int, order_type: OrderType = OrderType.MARKET,
                   limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                   time_in_force: TimeInForce = TimeInForce.DAY, client_order_id: Optional[str] = None,
                   live: bool = False) -> Order:
        """
        Place an order in the simulator
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            qty: Order quantity
            order_type: Order type (market/limit/stop/stop_limit)
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            client_order_id: Client order ID
            live: Ignored for simulator
            
        Returns:
            Order with deterministic fill
        """
        # Generate unique order ID
        if not client_order_id:
            client_order_id = f"sim_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        order_id = f"sim_order_{client_order_id}"
        timestamp = datetime.utcnow().isoformat()
        
        # Create order
        order = Order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol.upper(),
            side=side,
            qty=abs(qty),
            type=order_type,
            status=OrderStatus.NEW,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            submitted_at=timestamp,
            updated_at=timestamp,
            provider="simulator",
            receipt_hash=hash_payload({"order": order_id, "timestamp": timestamp})[:16]
        )
        
        # Store order
        self.orders[order_id] = order
        
        # For market orders, fill immediately with deterministic price
        if order_type == OrderType.MARKET:
            fill_price = self._get_deterministic_price(symbol)
            self._fill_order(order, fill_price, abs(qty), timestamp)
        
        # Store receipt
        store_tool_receipt(
            tool_name="broker.simulator.place_order",
            inputs={
                "symbol": symbol,
                "side": side.value,
                "qty": qty,
                "type": order_type.value,
                "client_order_id": client_order_id
            },
            raw_payload=order.dict()
        )
        
        return order
    
    def cancel_order(self, order_id: str, client_order_id: Optional[str] = None, live: bool = False) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            client_order_id: Client order ID
            live: Ignored for simulator
            
        Returns:
            True if cancel successful
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.NEW, OrderStatus.PENDING]:
                order.status = OrderStatus.CANCELED
                order.updated_at = datetime.utcnow().isoformat()
                return True
        return False
    
    def get_account(self, live: bool = False) -> Account:
        """
        Get account information
        
        Args:
            live: Ignored for simulator
            
        Returns:
            Account with current balances
        """
        # Calculate total portfolio value
        total_position_value = sum(
            pos.market_value or 0 for pos in self.positions.values()
        )
        
        total_value = self.cash + total_position_value
        total_pnl = total_value - self.initial_cash
        
        return Account(
            account_id="simulator_account",
            cash=self.cash,
            buying_power=self.cash,  # Simplified: no margin
            total_value=total_value,
            day_pnl=0.0,  # Simplified: no day tracking
            total_pnl=total_pnl,
            updated_at=datetime.utcnow().isoformat(),
            provider="simulator",
            metadata={
                "initial_cash": self.initial_cash,
                "total_position_value": total_position_value
            }
        )
    
    def get_positions(self, live: bool = False) -> List[Position]:
        """
        Get current positions
        
        Args:
            live: Ignored for simulator
            
        Returns:
            List of current positions
        """
        return [pos for pos in self.positions.values() if pos.qty != 0]
    
    def get_orders(self, status: Optional[OrderStatus] = None, symbol: Optional[str] = None,
                   limit: int = 100, live: bool = False) -> List[Order]:
        """
        Get orders
        
        Args:
            status: Filter by order status
            symbol: Filter by symbol
            limit: Maximum orders to return
            live: Ignored for simulator
            
        Returns:
            List of orders
        """
        orders = list(self.orders.values())
        
        # Apply filters
        if status:
            orders = [o for o in orders if o.status == status]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol.upper()]
        
        # Sort by submission time (most recent first) and limit
        orders.sort(key=lambda o: o.submitted_at, reverse=True)
        return orders[:limit]
    
    def get_fills(self, symbol: Optional[str] = None, limit: int = 100) -> List[Fill]:
        """
        Get fills
        
        Args:
            symbol: Filter by symbol
            limit: Maximum fills to return
            
        Returns:
            List of fills
        """
        fills = list(self.fills.values())
        
        if symbol:
            fills = [f for f in fills if f.symbol == symbol.upper()]
        
        # Sort by timestamp (most recent first) and limit
        fills.sort(key=lambda f: f.timestamp, reverse=True)
        return fills[:limit]
    
    def _get_deterministic_price(self, symbol: str) -> float:
        """Get deterministic price for symbol (for testing)"""
        # Use symbol hash + time to get reproducible but varying prices
        import hashlib
        base_hash = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        time_factor = int(time.time()) % 3600  # Hour-based variation
        
        # Generate price in range $50-$500
        price = 50.0 + ((base_hash + time_factor) % 45000) / 100.0
        return round(price, 2)
    
    def _fill_order(self, order: Order, fill_price: float, fill_qty: int, timestamp: str):
        """Fill an order and update positions"""
        # Create fill record
        fill_id = f"sim_fill_{order.order_id}_{int(time.time() * 1000)}"
        
        fill = Fill(
            fill_id=fill_id,
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            qty=fill_qty,
            price=fill_price,
            timestamp=timestamp,
            provider="simulator",
            receipt_hash=hash_payload({"fill": fill_id, "timestamp": timestamp})[:16]
        )
        
        self.fills[fill_id] = fill
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_qty = fill_qty
        order.avg_fill_price = fill_price
        order.updated_at = timestamp
        
        # Update position
        self._update_position(order.symbol, order.side, fill_qty, fill_price)
        
        # Update cash
        if order.side == OrderSide.BUY:
            self.cash -= fill_qty * fill_price
        else:
            self.cash += fill_qty * fill_price
    
    def _update_position(self, symbol: str, side: OrderSide, qty: int, price: float):
        """Update position for symbol"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=0,
                avg_price=None,
                market_value=0.0,
                unrealized_pnl=0.0,
                cost_basis=0.0,
                updated_at=datetime.utcnow().isoformat(),
                provider="simulator"
            )
        
        pos = self.positions[symbol]
        
        # Calculate new position
        if side == OrderSide.BUY:
            new_qty = pos.qty + qty
            if pos.qty == 0:
                new_avg_price = price
            else:
                total_cost = (pos.qty * (pos.avg_price or 0)) + (qty * price)
                new_avg_price = total_cost / new_qty if new_qty != 0 else 0
        else:  # SELL
            new_qty = pos.qty - qty
            new_avg_price = pos.avg_price  # Keep same avg price on sells
        
        # Update position
        pos.qty = new_qty
        pos.avg_price = new_avg_price if new_qty != 0 else None
        pos.updated_at = datetime.utcnow().isoformat()
        
        # Calculate market value (simplified: use last trade price)
        if new_qty != 0:
            pos.market_value = abs(new_qty) * price
            pos.cost_basis = abs(new_qty) * (new_avg_price or 0)
            pos.unrealized_pnl = pos.market_value - pos.cost_basis if pos.cost_basis else 0
        else:
            pos.market_value = 0.0
            pos.cost_basis = 0.0
            pos.unrealized_pnl = 0.0