"""DuckDB receipts system for ops tools."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


class ReceiptsDB:
    """Pure Python DuckDB-like receipts system for CI compatibility."""

    def __init__(self, db_path="artifacts/receipts.jsonl"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def write_receipt(self, tool: str, params: dict, result: dict, extra: dict = None):
        """Write a receipt row to the database."""
        # Create deterministic hashes
        params_hash = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]

        receipt_data = {
            'tool': tool,
            'params': params,
            'result': result,
            'extra': extra or {}
        }
        receipt_hash = hashlib.sha1(json.dumps(receipt_data, sort_keys=True).encode()).hexdigest()[:16]

        # Create receipt row
        receipt_row = {
            'tool': tool,
            'params_hash': params_hash,
            'receipt_hash': receipt_hash,
            'ts': datetime.utcnow().isoformat() + 'Z',
            'extra': extra or {}
        }

        # Append to JSONL file
        with open(self.db_path, 'a') as f:
            f.write(json.dumps(receipt_row) + '\n')

        return receipt_hash

    def query_latest(self, tool: str):
        """Get latest receipt for a tool."""
        if not self.db_path.exists():
            return None

        latest = None
        with open(self.db_path, 'r') as f:
            for line in f:
                row = json.loads(line.strip())
                if row['tool'] == tool:
                    if latest is None or row['ts'] > latest['ts']:
                        latest = row
        return latest


def get_receipts_db():
    """Get receipts database instance."""
    return ReceiptsDB()


def write_tool_receipt(tool_name: str, params: dict, status: str, result_data: dict = None):
    """Convenience function to write a tool receipt."""
    db = get_receipts_db()

    result = result_data or {}
    extra = {'status': status}

    receipt_hash = db.write_receipt(tool_name, params, result, extra)

    print(f"RECEIPT: {tool_name}:{receipt_hash}")
    return receipt_hash