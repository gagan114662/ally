#!/usr/bin/env python3
"""
Phase 10: Order Lifecycle and Journaling
Manages order states, transitions, and deterministic journaling
"""

import json
import hashlib
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from pathlib import Path


class OrderStatus(Enum):
    """Order status enumeration"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class Order:
    """Represents a single order with lifecycle management"""

    def __init__(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET",
                 limit_price: Optional[float] = None, strategy: Optional[str] = None):
        """Initialize order"""
        self.order_id = self._generate_order_id()
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.limit_price = limit_price
        self.strategy = strategy or "UNKNOWN"

        # Lifecycle fields
        self.status = OrderStatus.NEW
        self.created_at = datetime.now().isoformat() + "Z"
        self.updated_at = self.created_at
        self.filled_quantity = 0.0
        self.avg_fill_price = 0.0
        self.fills = []
        self.status_history = [(self.created_at, OrderStatus.NEW.value, "Order created")]

    def _generate_order_id(self) -> str:
        """Generate deterministic order ID"""
        timestamp = datetime.now().isoformat()
        data = f"{timestamp}_{id(self)}"
        return "ORD_" + hashlib.sha1(data.encode()).hexdigest()[:12]

    def add_fill(self, quantity: float, price: float, timestamp: str, venue: str = "SIMULATOR") -> dict:
        """Add a fill to the order"""
        fill = {
            "fill_id": f"FILL_{len(self.fills)+1:04d}",
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
            "venue": venue
        }

        self.fills.append(fill)
        self.filled_quantity += quantity

        # Update average fill price
        if self.filled_quantity > 0:
            total_value = sum(f['quantity'] * f['price'] for f in self.fills)
            self.avg_fill_price = total_value / self.filled_quantity

        # Update status
        if abs(self.filled_quantity) >= abs(self.quantity) * 0.999:  # Allow small rounding
            self._update_status(OrderStatus.FILLED, f"Fully filled: {self.filled_quantity:.2f}")
        elif self.filled_quantity > 0:
            self._update_status(OrderStatus.PARTIALLY_FILLED, f"Partial fill: {self.filled_quantity:.2f}/{self.quantity:.2f}")

        self.updated_at = timestamp
        return fill

    def cancel(self, reason: str = "User requested") -> bool:
        """Cancel the order"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
            return False

        self._update_status(OrderStatus.CANCELED, reason)
        return True

    def reject(self, reason: str) -> bool:
        """Reject the order"""
        if self.status != OrderStatus.NEW:
            return False

        self._update_status(OrderStatus.REJECTED, reason)
        return True

    def _update_status(self, new_status: OrderStatus, reason: str = ""):
        """Update order status with history"""
        self.status = new_status
        self.updated_at = datetime.now().isoformat() + "Z"
        self.status_history.append((self.updated_at, new_status.value, reason))

    def to_dict(self) -> dict:
        """Convert order to dictionary"""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "limit_price": self.limit_price,
            "strategy": self.strategy,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "filled_quantity": self.filled_quantity,
            "avg_fill_price": self.avg_fill_price,
            "fills": self.fills,
            "status_history": self.status_history
        }


class OrderJournal:
    """Manages order journaling to JSONL files"""

    def __init__(self, journal_path: str = "artifacts/execution/orders.jsonl"):
        """Initialize journal"""
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.orders = {}

    def add_order(self, order: Order) -> str:
        """Add order to journal"""
        self.orders[order.order_id] = order
        self._write_event({
            "event": "ORDER_NEW",
            "timestamp": order.created_at,
            "order": order.to_dict()
        })
        return order.order_id

    def update_order(self, order_id: str, update_type: str, data: dict) -> bool:
        """Update order in journal"""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        event = {
            "event": f"ORDER_{update_type}",
            "timestamp": datetime.now().isoformat() + "Z",
            "order_id": order_id,
            "data": data
        }

        if update_type == "FILL":
            fill = order.add_fill(
                quantity=data['quantity'],
                price=data['price'],
                timestamp=data.get('timestamp', datetime.now().isoformat() + "Z"),
                venue=data.get('venue', 'SIMULATOR')
            )
            event['fill'] = fill
        elif update_type == "CANCEL":
            order.cancel(data.get('reason', 'User requested'))
        elif update_type == "REJECT":
            order.reject(data.get('reason', 'Unknown'))

        event['order_status'] = order.status.value
        self._write_event(event)
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        """Get all open orders"""
        return [
            order for order in self.orders.values()
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
        ]

    def cancel_all_open(self, reason: str = "Kill switch activated") -> List[str]:
        """Cancel all open orders"""
        canceled = []
        for order in self.get_open_orders():
            if order.cancel(reason):
                self.update_order(order.order_id, "CANCEL", {"reason": reason})
                canceled.append(order.order_id)
        return canceled

    def _write_event(self, event: dict):
        """Write event to journal file"""
        with open(self.journal_path, 'a') as f:
            f.write(json.dumps(event, default=str) + '\n')

    def get_summary(self) -> dict:
        """Get journal summary statistics"""
        total_orders = len(self.orders)
        status_counts = {}

        for order in self.orders.values():
            status = order.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_orders": total_orders,
            "status_breakdown": status_counts,
            "open_orders": len(self.get_open_orders()),
            "filled_orders": status_counts.get(OrderStatus.FILLED.value, 0),
            "canceled_orders": status_counts.get(OrderStatus.CANCELED.value, 0),
            "rejected_orders": status_counts.get(OrderStatus.REJECTED.value, 0)
        }


class TradeJournal:
    """Manages trade/fill journaling"""

    def __init__(self, journal_path: str = "artifacts/execution/trades.jsonl"):
        """Initialize trade journal"""
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.trades = []

    def add_trade(self, trade: dict) -> str:
        """Add trade to journal"""
        trade_id = f"TRD_{len(self.trades)+1:06d}"
        trade['trade_id'] = trade_id
        trade['timestamp'] = trade.get('timestamp', datetime.now().isoformat() + "Z")

        self.trades.append(trade)
        self._write_trade(trade)
        return trade_id

    def _write_trade(self, trade: dict):
        """Write trade to journal file"""
        with open(self.journal_path, 'a') as f:
            f.write(json.dumps(trade, default=str) + '\n')

    def get_trades_by_order(self, order_id: str) -> List[dict]:
        """Get all trades for an order"""
        return [t for t in self.trades if t.get('order_id') == order_id]

    def get_summary(self) -> dict:
        """Get trade summary statistics"""
        if not self.trades:
            return {
                "total_trades": 0,
                "total_volume": 0.0,
                "total_notional": 0.0
            }

        total_volume = sum(abs(t.get('quantity', 0)) for t in self.trades)
        total_notional = sum(abs(t.get('quantity', 0) * t.get('price', 0)) for t in self.trades)

        return {
            "total_trades": len(self.trades),
            "total_volume": total_volume,
            "total_notional": total_notional,
            "avg_trade_size": total_notional / len(self.trades) if self.trades else 0
        }


if __name__ == "__main__":
    # Test order lifecycle
    order = Order(
        symbol="AAPL",
        side="BUY",
        quantity=1000,
        order_type="MARKET",
        strategy="TEST_STRATEGY"
    )

    print(f"Created order: {order.order_id}")
    print(f"Status: {order.status.value}")

    # Add partial fill
    order.add_fill(
        quantity=600,
        price=176.05,
        timestamp=datetime.now().isoformat() + "Z"
    )
    print(f"After partial fill: {order.status.value}, filled: {order.filled_quantity}")

    # Complete fill
    order.add_fill(
        quantity=400,
        price=176.10,
        timestamp=datetime.now().isoformat() + "Z"
    )
    print(f"After complete fill: {order.status.value}, avg price: {order.avg_fill_price:.2f}")

    # Test journal
    journal = OrderJournal()
    journal.add_order(order)

    print(f"\nJournal summary: {journal.get_summary()}")