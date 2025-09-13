import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    import sqlite3
    DUCKDB_AVAILABLE = False


class DatabaseManager:
    def __init__(self, db_path: str = "data/ally_memory.duckdb"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database with required tables."""
        if DUCKDB_AVAILABLE:
            self.conn = duckdb.connect(self.db_path)
        else:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row

        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                inputs_hash TEXT NOT NULL,
                task TEXT NOT NULL,
                notes TEXT
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT,
                key TEXT,
                value DOUBLE,
                PRIMARY KEY (run_id, key)
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT,
                type TEXT,
                payload TEXT
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                run_id TEXT,
                symbol TEXT,
                side TEXT,
                qty DOUBLE,
                price DOUBLE,
                ts TEXT
            )
        """)

        # M-RealData Gate: Receipt tracking table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_receipts (
                content_sha1 TEXT PRIMARY KEY,
                vendor TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                params_json TEXT NOT NULL,
                ts_iso TEXT NOT NULL,
                bytes INTEGER NOT NULL,
                cost_cents INTEGER,
                session_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        if not DUCKDB_AVAILABLE:
            self.conn.commit()

    def log_run(self, run_id: str, task: str, code_hash: str, inputs_hash: str, 
                ts: str, metrics: Dict[str, float], events: List[Dict[str, Any]], 
                trades: List[Dict[str, Any]], notes: Optional[str] = None) -> bool:
        """Log a run to the database. Idempotent on same (run_id, code_hash, inputs_hash)."""
        try:
            # Check if run already exists with same hashes
            existing = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM runs WHERE run_id = ? AND code_hash = ? AND inputs_hash = ?",
                (run_id, code_hash, inputs_hash)
            ).fetchone()
            
            count = existing[0] if DUCKDB_AVAILABLE else existing["cnt"]
            if count > 0:
                return True  # Already logged, idempotent

            # Insert run
            self.conn.execute(
                "INSERT INTO runs (run_id, ts, code_hash, inputs_hash, task, notes) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, ts, code_hash, inputs_hash, task, notes)
            )

            # Insert metrics
            for key, value in metrics.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO metrics (run_id, key, value) VALUES (?, ?, ?)",
                    (run_id, key, value)
                )

            # Insert events
            for event in events:
                self.conn.execute(
                    "INSERT INTO events (run_id, type, payload) VALUES (?, ?, ?)",
                    (run_id, event.get("type", ""), json.dumps(event, sort_keys=True))
                )

            # Insert trades
            for trade in trades:
                self.conn.execute(
                    "INSERT INTO trades (run_id, symbol, side, qty, price, ts) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, trade.get("symbol", ""), trade.get("side", ""), 
                     trade.get("qty", 0), trade.get("price", 0), trade.get("ts", ""))
                )

            if not DUCKDB_AVAILABLE:
                self.conn.commit()

            return True

        except Exception as e:
            print(f"Error logging run: {e}")
            return False

    def query(self, table: str, where: Optional[str] = None, limit: int = 100) -> Dict[str, Any]:
        """Query database with optional WHERE clause."""
        try:
            query = f"SELECT * FROM {table}"
            params = []
            
            if where:
                query += f" WHERE {where}"
            
            query += f" LIMIT {limit}"

            if DUCKDB_AVAILABLE:
                rows = self.conn.execute(query, params).fetchall()
                columns = [desc[0] for desc in self.conn.description]
                result_rows = [dict(zip(columns, row)) for row in rows]
            else:
                cursor = self.conn.execute(query, params)
                result_rows = [dict(row) for row in cursor.fetchall()]

            # Get total count
            count_query = f"SELECT COUNT(*) as cnt FROM {table}"
            if where:
                count_query += f" WHERE {where}"
            
            count_result = self.conn.execute(count_query, params).fetchone()
            total_count = count_result[0] if DUCKDB_AVAILABLE else count_result["cnt"]

            return {"rows": result_rows, "count": total_count}

        except Exception as e:
            print(f"Error querying database: {e}")
            return {"rows": [], "count": 0}

    def insert_receipt(self, receipt) -> bool:
        """
        Insert a receipt into the data_receipts table
        
        Args:
            receipt: Receipt object with vendor, endpoint, etc.
        
        Returns:
            bool: Success status
        """
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO data_receipts 
                (content_sha1, vendor, endpoint, params_json, ts_iso, bytes, cost_cents, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                receipt.content_sha1,
                receipt.vendor,
                receipt.endpoint,
                json.dumps(receipt.params, sort_keys=True),
                receipt.ts_iso,
                receipt.bytes,
                receipt.cost_cents,
                getattr(receipt, 'session_id', None)
            ))
            
            if not DUCKDB_AVAILABLE:
                self.conn.commit()
            
            return True
            
        except Exception as e:
            print(f"Error inserting receipt: {e}")
            return False

    def get_receipts_by_vendor(self, vendor: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get receipts for a specific vendor"""
        try:
            query = "SELECT * FROM data_receipts WHERE vendor = ? ORDER BY created_at DESC LIMIT ?"
            
            if DUCKDB_AVAILABLE:
                rows = self.conn.execute(query, (vendor, limit)).fetchall()
                columns = [desc[0] for desc in self.conn.description]
                return [dict(zip(columns, row)) for row in rows]
            else:
                cursor = self.conn.execute(query, (vendor, limit))
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            print(f"Error querying receipts: {e}")
            return []

    def get_receipt_by_sha1(self, content_sha1: str) -> Optional[Dict[str, Any]]:
        """Get receipt by content SHA1"""
        try:
            query = "SELECT * FROM data_receipts WHERE content_sha1 = ?"
            
            if DUCKDB_AVAILABLE:
                result = self.conn.execute(query, (content_sha1,)).fetchone()
                if result:
                    columns = [desc[0] for desc in self.conn.description]
                    return dict(zip(columns, result))
            else:
                cursor = self.conn.execute(query, (content_sha1,))
                result = cursor.fetchone()
                if result:
                    return dict(result)
            
            return None
            
        except Exception as e:
            print(f"Error querying receipt: {e}")
            return None

    def get_receipt_stats(self) -> Dict[str, Any]:
        """Get receipt statistics for M-RealData proofs"""
        try:
            stats = {}
            
            # Total receipts
            count_result = self.conn.execute("SELECT COUNT(*) as cnt FROM data_receipts").fetchone()
            stats["total_receipts"] = count_result[0] if DUCKDB_AVAILABLE else count_result["cnt"]
            
            # Receipts by vendor
            vendor_query = "SELECT vendor, COUNT(*) as cnt FROM data_receipts GROUP BY vendor"
            if DUCKDB_AVAILABLE:
                vendor_rows = self.conn.execute(vendor_query).fetchall()
                stats["by_vendor"] = {row[0]: row[1] for row in vendor_rows}
            else:
                cursor = self.conn.execute(vendor_query)
                stats["by_vendor"] = {row["vendor"]: row["cnt"] for row in cursor.fetchall()}
            
            # Total cost
            cost_result = self.conn.execute("SELECT SUM(cost_cents) as total FROM data_receipts WHERE cost_cents IS NOT NULL").fetchone()
            stats["total_cost_cents"] = cost_result[0] if (cost_result and cost_result[0]) else 0
            
            return stats
            
        except Exception as e:
            print(f"Error getting receipt stats: {e}")
            return {"total_receipts": 0, "by_vendor": {}, "total_cost_cents": 0}

    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


# Global instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get or create global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager