# ally/utils/file_receipts.py
from __future__ import annotations
import os, datetime
from .hashing import hash_payload
from .db import DatabaseManager

def file_sha1(path: str) -> str:
    with open(path, "rb") as f:
        return hash_payload(f.read())

def record_file_receipt(db_path: str, tool: str, label: str, path: str, params_hash8: str) -> str:
    sha1_full = file_sha1(path)
    receipt16 = sha1_full[:16]
    
    # Store receipt in database
    db = DatabaseManager(db_path)
    db.store_receipt(
        tool_name=tool,
        args_hash=params_hash8,
        receipt_hash=receipt16,
        payload_raw=f'{{"label": "{label}", "path": "{os.path.abspath(path)}", "sha1": "{sha1_full}"}}',
        timestamp=datetime.datetime.utcnow().isoformat()
    )
    db.close()
    
    return receipt16