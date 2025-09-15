"""
QuantConnect Paper Trading adapter for Ally

Integrates with LEAN CLI Paper Brokerage via:
- Project bootstrapping (config + algorithm)
- Order intent file communication (orders_inbox.jsonl)
- Receipt generation from QC results/logs
- Live session management with Docker/lean CLI

Gating: Requires live=True AND ALLY_LIVE=1 for network access
"""

import os
import json
import time
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ...utils.gating import check_live_mode_allowed
from ...utils.receipts import store_tool_receipt
from ...utils.hashing import hash_inputs, hash_payload
from ...utils.db import DatabaseManager
from ...integrations.quantconnect.qc_project_bootstrap import bootstrap_qc_project
from ...schemas.broker import Order, Fill, Position, Account, BrokerSession, OrderSide, OrderType, OrderStatus, TimeInForce


class QCPaperAdapter:
    """QuantConnect Paper Brokerage adapter with gating and receipts"""
    
    def __init__(self, qc_user_id: Optional[str] = None, qc_api_token: Optional[str] = None):
        """Initialize with QuantConnect credentials"""
        self.qc_user_id = qc_user_id or os.getenv("QC_USER_ID")
        self.qc_api_token = qc_api_token or os.getenv("QC_API_TOKEN")
        self.qc_org_id = os.getenv("QC_ORG_ID")
        self.qc_data_dir = os.getenv("QC_DATA_DIR", "./qc-data")
        self.qc_results_dir = os.getenv("QC_RESULTS_DIR", "./qc-results")
        self.lean_docker = os.getenv("LEAN_DOCKER", "1") == "1"
        
        # Active sessions tracking
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._db_path = "data/ally_memory.duckdb"
        
    def start_session(self, project_slug: str, symbols: List[str], live: bool = False) -> BrokerSession:
        """
        Start a new QC paper trading session
        
        Args:
            project_slug: Unique project identifier
            symbols: List of symbols to trade
            live: Whether to run live session (requires gating)
            
        Returns:
            BrokerSession with session details
        """
        # Check gating requirements
        check_live_mode_allowed(
            live=live,
            api_key=self.qc_api_token,
            service_name="QuantConnect"
        )
        
        if not live:
            # Return mock session for offline mode
            return self._create_mock_session(project_slug, symbols)
        
        # Validate QC credentials
        if not self.qc_user_id or not self.qc_api_token:
            raise ValueError("QC_USER_ID and QC_API_TOKEN required for live sessions")
        
        if self.qc_user_id in ["your_qc_user_id_here", "", "demo"]:
            raise ValueError("Invalid or placeholder QC_USER_ID")
        
        if self.qc_api_token in ["your_qc_api_token_here", "", "demo"]:
            raise ValueError("Invalid or placeholder QC_API_TOKEN")
        
        # Check if lean CLI is available
        if not self._check_lean_cli():
            raise RuntimeError("LEAN CLI not found or Docker not running")
        
        # Bootstrap QC project
        bootstrap_result = bootstrap_qc_project(
            project_slug=project_slug,
            symbols=symbols,
            out_root=".ally_qc",
            templates_root="ally/integrations/quantconnect/templates",
            results_dir=self.qc_results_dir,
            data_dir=self.qc_data_dir,
            params={"ALLY_INBOX": "orders_inbox.jsonl"},
            db_path=self._db_path,
            deterministic=True
        )
        
        # Start LEAN live session
        session_id = self._start_lean_session(bootstrap_result)
        
        # Create session record
        session = BrokerSession(
            session_id=session_id,
            backend="qc_paper",
            project_slug=project_slug,
            symbols=symbols,
            started_at=datetime.utcnow().isoformat(),
            status="active",
            metadata={
                "project_dir": bootstrap_result.project_dir,
                "config_path": bootstrap_result.config_path,
                "algorithm_path": bootstrap_result.algorithm_path,
                "inbox_path": bootstrap_result.inbox_path,
                "params_hash": bootstrap_result.params_hash,
                "receipts": bootstrap_result.receipts
            }
        )
        
        self._sessions[session_id] = {
            "session": session,
            "bootstrap": bootstrap_result,
            "process": None  # Will be set by _start_lean_session
        }
        
        return session
    
    def stop_session(self, session_id: str, live: bool = False) -> bool:
        """
        Stop a QC paper trading session
        
        Args:
            session_id: Session to stop
            live: Whether this is a live session
            
        Returns:
            True if stopped successfully
        """
        if not live:
            # Mock stop for offline mode
            return True
        
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_info = self._sessions[session_id]
        
        # Stop LEAN process
        if session_info.get("process"):
            try:
                if self.lean_docker:
                    # Stop Docker container
                    subprocess.run(["lean", "stop", session_info["bootstrap"].project_dir], 
                                 check=True, capture_output=True)
                else:
                    # Terminate process
                    session_info["process"].terminate()
                    session_info["process"].wait(timeout=30)
            except Exception as e:
                print(f"Warning: Error stopping LEAN session: {e}")
        
        # Update session status
        session_info["session"].status = "stopped"
        
        # Store final receipts
        self._store_session_receipts(session_id, "stop")
        
        return True
    
    def place_order(self, symbol: str, side: OrderSide, qty: int, order_type: OrderType = OrderType.MARKET,
                   limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                   time_in_force: TimeInForce = TimeInForce.DAY, client_order_id: Optional[str] = None,
                   live: bool = False) -> Order:
        """
        Place an order through QC Paper Brokerage
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            qty: Order quantity
            order_type: Order type (market/limit/stop/stop_limit)
            limit_price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            client_order_id: Client order ID
            live: Whether to place live order
            
        Returns:
            Order with initial status
        """
        if not live:
            # Return mock order for offline mode
            return self._create_mock_order(symbol, side, qty, order_type, limit_price, stop_price, 
                                         time_in_force, client_order_id)
        
        # Generate unique client order ID if not provided
        if not client_order_id:
            client_order_id = f"ally_{int(time.time() * 1000)}"
        
        # Create order intent
        intent = {
            "symbol": symbol.upper(),
            "side": side.value,
            "qty": abs(qty) if side == OrderSide.BUY else -abs(qty),
            "type": order_type.value,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "tif": time_in_force.value,
            "client_order_id": client_order_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Hash intent for receipt
        intent_hash = hash_payload(intent)
        
        # Write intent to active session inboxes
        active_sessions = [s for s in self._sessions.values() if s["session"].status == "active"]
        if not active_sessions:
            raise RuntimeError("No active QC paper sessions found")
        
        for session_info in active_sessions:
            inbox_path = session_info["bootstrap"].inbox_path
            with open(inbox_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(intent) + "\n")
        
        # Store receipt
        store_tool_receipt(
            tool_name="broker.qc_paper.place_order",
            inputs=intent,
            raw_payload=intent
        )
        
        # Create initial order record
        order = Order(
            order_id=f"qc_pending_{client_order_id}",
            client_order_id=client_order_id,
            symbol=symbol.upper(),
            side=side,
            qty=abs(qty),
            type=order_type,
            status=OrderStatus.NEW,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            submitted_at=intent["timestamp"],
            updated_at=intent["timestamp"],
            provider="qc_paper",
            receipt_hash=intent_hash[:16],
            metadata={"intent_hash": intent_hash}
        )
        
        # Store in database
        self._store_order(order)
        
        return order
    
    def cancel_order(self, order_id: str, client_order_id: Optional[str] = None, live: bool = False) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            client_order_id: Client order ID
            live: Whether to cancel live order
            
        Returns:
            True if cancel request sent successfully
        """
        if not live:
            # Mock cancel for offline mode
            return True
        
        # Create cancel intent
        cancel_intent = {
            "action": "cancel",
            "order_id": order_id,
            "client_order_id": client_order_id,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Write to active session inboxes
        active_sessions = [s for s in self._sessions.values() if s["session"].status == "active"]
        for session_info in active_sessions:
            inbox_path = session_info["bootstrap"].inbox_path
            with open(inbox_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(cancel_intent) + "\n")
        
        # Store receipt
        store_tool_receipt(
            tool_name="broker.qc_paper.cancel_order",
            inputs=cancel_intent,
            raw_payload=cancel_intent
        )
        
        return True
    
    def get_account(self, live: bool = False) -> Account:
        """
        Get account information
        
        Args:
            live: Whether to get live account info
            
        Returns:
            Account with current balances
        """
        if not live:
            # Return mock account for offline mode
            return Account(
                account_id="qc_paper_mock",
                cash=1000000.0,
                buying_power=1000000.0,
                total_value=1000000.0,
                day_pnl=0.0,
                total_pnl=0.0,
                updated_at=datetime.utcnow().isoformat(),
                provider="qc_paper",
                metadata={"mode": "offline_mock"}
            )
        
        # TODO: Parse QC live results for account info
        # For now, return placeholder
        return Account(
            account_id="qc_paper_live",
            cash=1000000.0,
            buying_power=1000000.0,
            total_value=1000000.0,
            updated_at=datetime.utcnow().isoformat(),
            provider="qc_paper"
        )
    
    def get_positions(self, live: bool = False) -> List[Position]:
        """
        Get current positions
        
        Args:
            live: Whether to get live positions
            
        Returns:
            List of current positions
        """
        if not live:
            # Return empty positions for offline mode
            return []
        
        # TODO: Parse QC live results for positions
        # For now, return empty
        return []
    
    def get_orders(self, status: Optional[OrderStatus] = None, symbol: Optional[str] = None,
                   limit: int = 100, live: bool = False) -> List[Order]:
        """
        Get orders
        
        Args:
            status: Filter by order status
            symbol: Filter by symbol
            limit: Maximum orders to return
            live: Whether to get live orders
            
        Returns:
            List of orders
        """
        # Query from database
        db = DatabaseManager(self._db_path)
        
        # TODO: Implement proper order querying from database
        # For now, return empty
        orders = []
        
        db.close()
        return orders
    
    def _check_lean_cli(self) -> bool:
        """Check if LEAN CLI is available"""
        try:
            result = subprocess.run(["lean", "--version"], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _start_lean_session(self, bootstrap_result) -> str:
        """Start LEAN live session and return session ID"""
        session_id = f"qc_paper_{int(time.time())}"
        
        # Set environment variables for LEAN
        env = os.environ.copy()
        env.update({
            "QC_USER_ID": self.qc_user_id,
            "QC_API_TOKEN": self.qc_api_token,
            "QC_DATA_DIR": self.qc_data_dir,
            "QC_RESULTS_DIR": self.qc_results_dir
        })
        if self.qc_org_id:
            env["QC_ORG_ID"] = self.qc_org_id
        
        # Start LEAN process
        cmd = ["lean", "live", bootstrap_result.project_dir, "-c", "config.live.paper.json"]
        if self.lean_docker:
            cmd.append("--detach")
        
        try:
            process = subprocess.Popen(cmd, env=env, cwd=bootstrap_result.project_dir)
            if session_id in self._sessions:
                self._sessions[session_id]["process"] = process
            print(f"Started LEAN session {session_id} (PID: {process.pid})")
            return session_id
        except Exception as e:
            raise RuntimeError(f"Failed to start LEAN session: {e}")
    
    def _create_mock_session(self, project_slug: str, symbols: List[str]) -> BrokerSession:
        """Create mock session for offline mode"""
        session_id = f"mock_paper_{int(time.time())}"
        
        return BrokerSession(
            session_id=session_id,
            backend="qc_paper",
            project_slug=project_slug,
            symbols=symbols,
            started_at=datetime.utcnow().isoformat(),
            status="active",
            metadata={"mode": "offline_mock"}
        )
    
    def _create_mock_order(self, symbol: str, side: OrderSide, qty: int, order_type: OrderType,
                          limit_price: Optional[float], stop_price: Optional[float],
                          time_in_force: TimeInForce, client_order_id: Optional[str]) -> Order:
        """Create mock order for offline mode"""
        if not client_order_id:
            client_order_id = f"mock_{int(time.time() * 1000)}"
        
        timestamp = datetime.utcnow().isoformat()
        
        return Order(
            order_id=f"mock_qc_{client_order_id}",
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
            provider="qc_paper",
            metadata={"mode": "offline_mock"}
        )
    
    def _store_order(self, order: Order):
        """Store order in database"""
        # TODO: Implement proper order storage
        pass
    
    def _store_session_receipts(self, session_id: str, action: str):
        """Store session receipts"""
        # TODO: Implement session receipt storage
        pass