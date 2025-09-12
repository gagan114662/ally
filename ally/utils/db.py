"""
Database utilities for Ally - Memory persistence with DuckDB
"""

import os
import json
import re
import duckdb
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path


# Allowlisted tables for secure query operations
ALLOWED_TABLES = {"runs", "metrics", "events", "trades"}


class DatabaseManager:
    """Manages DuckDB database connections and operations for Ally memory system"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database manager
        
        Args:
            db_path: Path to DuckDB database file. If None, uses default location.
        """
        if db_path is None:
            # Default to data/ally_memory.duckdb
            project_root = Path(__file__).parent.parent.parent
            db_path = project_root / "data" / "ally_memory.duckdb"
        
        self.db_path = str(db_path)
        self._ensure_data_dir()
        self._conn = None
        self._initialize_db()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get database connection, creating if needed"""
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn
    
    def _initialize_db(self):
        """Initialize database schema if tables don't exist"""
        conn = self._get_connection()
        
        # Create runs table with basic schema
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                task VARCHAR NOT NULL,
                code_hash VARCHAR NOT NULL,
                inputs_hash VARCHAR NOT NULL,
                ts VARCHAR NOT NULL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Try to add missing columns for enhanced functionality
        columns_to_add = [
            ("metrics", "JSON"),
            ("events", "JSON"),
            ("trades", "JSON")
        ]
        
        for col_name, col_type in columns_to_add:
            try:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")
            except Exception:
                # Column already exists or other issue, continue
                pass
        
        # Create indexes for common queries
        try:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at)")
        except Exception:
            # Index creation might fail, continue
            pass
        
        conn.commit()
    
    def safe_query(self, table: str, filters: dict = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Execute safe parameterized query on allowlisted table
        
        Args:
            table: Table name (must be in ALLOWED_TABLES)
            filters: Dictionary of column=value filters
            limit: Maximum rows to return
            
        Returns:
            List of rows as dictionaries
        """
        if table not in ALLOWED_TABLES:
            raise ValueError(f"Invalid table: {table}. Allowed: {', '.join(ALLOWED_TABLES)}")
        
        filters = filters or {}
        conn = self._get_connection()
        
        # Build parameterized query
        keys = list(filters.keys())
        clause = " AND ".join([f"{k} = ?" for k in keys])
        sql = f"SELECT * FROM {table}"
        if clause:
            sql += f" WHERE {clause}"
        sql += " LIMIT ?"
        
        # Parameters in same order as placeholders
        params = [filters[k] for k in keys] + [int(limit)]
        
        result = conn.execute(sql, params).fetchall()
        columns = [desc[0] for desc in conn.description] if conn.description else []
        
        # Convert to list of dicts
        rows = []
        for row in result:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                # Convert JSON strings back to objects
                if col in ['metrics', 'events', 'trades'] and isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except:
                        pass
                # Convert timestamps to ISO strings
                elif isinstance(value, datetime):
                    value = value.isoformat() + 'Z'
                row_dict[col] = value
            rows.append(row_dict)
        
        return rows
    
    def log_run(
        self,
        run_id: str,
        task: str,
        code_hash: str,
        inputs_hash: str,
        ts: str,
        metrics: Dict[str, Union[float, int]] = None,
        events: List[Dict[str, Any]] = None,
        trades: List[Dict[str, Any]] = None,
        notes: str = None
    ) -> bool:
        """
        Log a run to the database
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            
            # Keep timestamp as string to match existing schema
            conn.execute("""
                INSERT OR REPLACE INTO runs 
                (run_id, task, code_hash, inputs_hash, ts, metrics, events, trades, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id,
                task,
                code_hash,
                inputs_hash,
                ts,  # Keep as string
                json.dumps(metrics or {}),
                json.dumps(events or []),
                json.dumps(trades or []),
                notes
            ])
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error logging run: {e}")
            return False
    
    def query_runs(
        self,
        query: str = None,
        table: str = None,
        where: str = None,
        params: Dict[str, Any] = None,
        limit: int = None
    ) -> Dict[str, Any]:
        """
        Query runs from database with security improvements
        
        Args:
            query: DEPRECATED - Use table + where instead
            table: Table name for SELECT queries
            where: DEPRECATED - Use safe filters instead
            params: Query parameters (not used)
            limit: Maximum rows to return
            
        Returns:
            Dict with rows, count, columns, and execution_time_ms
        """
        try:
            import time
            start_time = time.time()
            
            # Default table if none specified
            if not table and not query:
                table = "runs"
            
            # Handle legacy where clause for backwards compatibility
            filters = {}
            if where:
                # Legacy support: parse simple where="run_id='<id>'" patterns
                match = re.fullmatch(r"run_id\s*=\s*'([^']+)'", where)
                if match:
                    filters = {"run_id": match.group(1)}
                else:
                    # For safety, ignore complex where clauses
                    print(f"Warning: Ignoring unsafe where clause: {where}")
            
            # Use safe query method
            if table and not query:
                rows = self.safe_query(table=table, filters=filters, limit=limit or 100)
                columns = list(rows[0].keys()) if rows else []
            else:
                # Fall back to original method for custom queries (DEPRECATED)
                print("Warning: Custom SQL queries are deprecated for security")
                conn = self._get_connection()
                result = conn.execute(query).fetchall()
                columns = [desc[0] for desc in conn.description] if conn.description else []
                
                # Convert to list of dicts (simplified for deprecated path)
                rows = [dict(zip(columns, row)) for row in result]
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return {
                "rows": rows,
                "count": len(rows),
                "columns": columns,
                "execution_time_ms": execution_time_ms
            }
            
        except Exception as e:
            print(f"Error querying runs: {e}")
            return {
                "rows": [],
                "count": 0,
                "columns": [],
                "execution_time_ms": 0
            }
    
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific run by ID (now secure)
        
        Args:
            run_id: Run identifier
            
        Returns:
            Run data as dict or None if not found
        """
        try:
            rows = self.safe_query(table="runs", filters={"run_id": run_id}, limit=1)
            return rows[0] if rows else None
        except Exception as e:
            print(f"Error getting run {run_id}: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Global database manager instance
_db_manager = None


def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager