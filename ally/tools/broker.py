"""
Broker tools for Ally - unified facade for paper trading and simulation

Supports multiple backends:
- qc_paper: QuantConnect Paper Brokerage (live, gated)
- simulator: Deterministic fill engine (offline, CI-safe)
"""

from ..tools import register
from ..schemas.base import ToolResult
from ..schemas.broker import (
    PlaceOrderIn, CancelOrderIn, GetAccountIn, GetPositionsIn, GetOrdersIn,
    Order, Fill, Position, Account, BrokerSession
)


@register("broker.start_session")
def broker_start_session(**kwargs) -> ToolResult:
    """
    Start a broker trading session
    
    Args:
        project_slug: Unique project identifier
        symbols: List of symbols to trade
        backend: Broker backend (qc_paper, simulator)
        live: Enable live trading (requires ALLY_LIVE=1)
    
    Returns:
        ToolResult with BrokerSession
    """
    try:
        project_slug = kwargs.get('project_slug', 'default')
        symbols = kwargs.get('symbols', ['AAPL'])
        backend = kwargs.get('backend', 'simulator')
        live = kwargs.get('live', False)
        
        if backend == "qc_paper":
            from ..adapters.broker.qc_paper_adapter import QCPaperAdapter
            adapter = QCPaperAdapter()
            session = adapter.start_session(project_slug, symbols, live)
        elif backend == "simulator":
            from ..adapters.broker.simulator_adapter import SimulatorAdapter
            adapter = SimulatorAdapter()
            session = adapter.start_session(project_slug, symbols, live)
        else:
            return ToolResult.error([f"Unknown backend: {backend}"])
        
        return ToolResult.success(
            data={
                "session": session.dict(),
                "message": f"Started {backend} session: {session.session_id}"
            }
        )
        
    except Exception as e:
        return ToolResult.error([f"Failed to start session: {e}"])


@register("broker.stop_session")
def broker_stop_session(**kwargs) -> ToolResult:
    """
    Stop a broker trading session
    
    Args:
        session_id: Session ID to stop
        backend: Broker backend (qc_paper, simulator)
        live: Whether this is a live session
    
    Returns:
        ToolResult with stop confirmation
    """
    try:
        session_id = kwargs.get('session_id')
        backend = kwargs.get('backend', 'simulator')
        live = kwargs.get('live', False)
        
        if not session_id:
            return ToolResult.error(["session_id is required"])
        
        if backend == "qc_paper":
            from ..adapters.broker.qc_paper_adapter import QCPaperAdapter
            adapter = QCPaperAdapter()
            success = adapter.stop_session(session_id, live)
        elif backend == "simulator":
            from ..adapters.broker.simulator_adapter import SimulatorAdapter
            adapter = SimulatorAdapter()
            success = adapter.stop_session(session_id, live)
        else:
            return ToolResult.error([f"Unknown backend: {backend}"])
        
        if success:
            return ToolResult.success(
                data={"message": f"Stopped session: {session_id}"}
            )
        else:
            return ToolResult.error([f"Failed to stop session: {session_id}"])
        
    except Exception as e:
        return ToolResult.error([f"Failed to stop session: {e}"])


@register("broker.place_order")
def broker_place_order(**kwargs) -> ToolResult:
    """
    Place a trading order
    
    Supports market, limit, stop, and stop-limit orders
    """
    try:
        inputs = PlaceOrderIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        if inputs.backend == "qc_paper":
            from ..adapters.broker.qc_paper_adapter import QCPaperAdapter
            adapter = QCPaperAdapter()
            order = adapter.place_order(
                symbol=inputs.symbol,
                side=inputs.side,
                qty=inputs.qty,
                order_type=inputs.type,
                limit_price=inputs.limit_price,
                stop_price=inputs.stop_price,
                time_in_force=inputs.time_in_force,
                client_order_id=inputs.client_order_id,
                live=inputs.live
            )
        elif inputs.backend == "simulator":
            from ..adapters.broker.simulator_adapter import SimulatorAdapter
            adapter = SimulatorAdapter()
            order = adapter.place_order(
                symbol=inputs.symbol,
                side=inputs.side,
                qty=inputs.qty,
                order_type=inputs.type,
                limit_price=inputs.limit_price,
                stop_price=inputs.stop_price,
                time_in_force=inputs.time_in_force,
                client_order_id=inputs.client_order_id,
                live=inputs.live
            )
        else:
            return ToolResult.error([f"Unknown backend: {inputs.backend}"])
        
        return ToolResult.success(
            data={
                "order": order.dict(),
                "message": f"Order placed: {order.order_id}"
            }
        )
        
    except Exception as e:
        return ToolResult.error([f"Failed to place order: {e}"])


@register("broker.cancel_order")
def broker_cancel_order(**kwargs) -> ToolResult:
    """
    Cancel a trading order
    """
    try:
        inputs = CancelOrderIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        if inputs.backend == "qc_paper":
            from ..adapters.broker.qc_paper_adapter import QCPaperAdapter
            adapter = QCPaperAdapter()
            success = adapter.cancel_order(
                order_id=inputs.order_id,
                client_order_id=inputs.client_order_id,
                live=inputs.live
            )
        elif inputs.backend == "simulator":
            from ..adapters.broker.simulator_adapter import SimulatorAdapter
            adapter = SimulatorAdapter()
            success = adapter.cancel_order(
                order_id=inputs.order_id,
                client_order_id=inputs.client_order_id,
                live=inputs.live
            )
        else:
            return ToolResult.error([f"Unknown backend: {inputs.backend}"])
        
        if success:
            return ToolResult.success(
                data={"message": f"Cancel request sent for order: {inputs.order_id}"}
            )
        else:
            return ToolResult.error([f"Failed to cancel order: {inputs.order_id}"])
        
    except Exception as e:
        return ToolResult.error([f"Failed to cancel order: {e}"])


@register("broker.get_account")
def broker_get_account(**kwargs) -> ToolResult:
    """
    Get account information
    """
    try:
        inputs = GetAccountIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        if inputs.backend == "qc_paper":
            from ..adapters.broker.qc_paper_adapter import QCPaperAdapter
            adapter = QCPaperAdapter()
            account = adapter.get_account(live=inputs.live)
        elif inputs.backend == "simulator":
            from ..adapters.broker.simulator_adapter import SimulatorAdapter
            adapter = SimulatorAdapter()
            account = adapter.get_account(live=inputs.live)
        else:
            return ToolResult.error([f"Unknown backend: {inputs.backend}"])
        
        return ToolResult.success(
            data={"account": account.dict()}
        )
        
    except Exception as e:
        return ToolResult.error([f"Failed to get account: {e}"])


@register("broker.get_positions")
def broker_get_positions(**kwargs) -> ToolResult:
    """
    Get current positions
    """
    try:
        inputs = GetPositionsIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        if inputs.backend == "qc_paper":
            from ..adapters.broker.qc_paper_adapter import QCPaperAdapter
            adapter = QCPaperAdapter()
            positions = adapter.get_positions(live=inputs.live)
        elif inputs.backend == "simulator":
            from ..adapters.broker.simulator_adapter import SimulatorAdapter
            adapter = SimulatorAdapter()
            positions = adapter.get_positions(live=inputs.live)
        else:
            return ToolResult.error([f"Unknown backend: {inputs.backend}"])
        
        return ToolResult.success(
            data={
                "positions": [pos.dict() for pos in positions],
                "count": len(positions)
            }
        )
        
    except Exception as e:
        return ToolResult.error([f"Failed to get positions: {e}"])


@register("broker.get_orders")
def broker_get_orders(**kwargs) -> ToolResult:
    """
    Get orders
    """
    try:
        inputs = GetOrdersIn(**kwargs)
    except Exception as e:
        return ToolResult.error([f"Invalid inputs: {e}"])
    
    try:
        if inputs.backend == "qc_paper":
            from ..adapters.broker.qc_paper_adapter import QCPaperAdapter
            adapter = QCPaperAdapter()
            orders = adapter.get_orders(
                status=inputs.status,
                symbol=inputs.symbol,
                limit=inputs.limit,
                live=inputs.live
            )
        elif inputs.backend == "simulator":
            from ..adapters.broker.simulator_adapter import SimulatorAdapter
            adapter = SimulatorAdapter()
            orders = adapter.get_orders(
                status=inputs.status,
                symbol=inputs.symbol,
                limit=inputs.limit,
                live=inputs.live
            )
        else:
            return ToolResult.error([f"Unknown backend: {inputs.backend}"])
        
        return ToolResult.success(
            data={
                "orders": [order.dict() for order in orders],
                "count": len(orders)
            }
        )
        
    except Exception as e:
        return ToolResult.error([f"Failed to get orders: {e}"])


# Convenience functions for direct usage
def place_market_order(symbol: str, side: str, qty: int, backend: str = "simulator", live: bool = False) -> Order:
    """Place a market order directly"""
    from ..adapters.broker.simulator_adapter import SimulatorAdapter
    from ..adapters.broker.qc_paper_adapter import QCPaperAdapter
    from ..schemas.broker import OrderSide, OrderType
    
    if backend == "qc_paper":
        adapter = QCPaperAdapter()
    else:
        adapter = SimulatorAdapter()
    
    return adapter.place_order(
        symbol=symbol,
        side=OrderSide(side.lower()),
        qty=qty,
        order_type=OrderType.MARKET,
        live=live
    )


if __name__ == "__main__":
    # Test broker functionality
    print("🧪 Testing Broker Tools...")
    
    # Test simulator
    result = broker_place_order(
        symbol="AAPL",
        side="buy",
        qty=100,
        type="market",
        backend="simulator",
        live=False
    )
    
    print(f"Simulator order result: {result.ok}")
    if result.ok:
        print(f"Order ID: {result.data['order']['order_id']}")
    else:
        print(f"Errors: {result.errors}")