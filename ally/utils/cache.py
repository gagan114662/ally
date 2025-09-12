import json, sqlite3, time, hashlib
from datetime import timedelta
from pathlib import Path
from typing import Optional, Any

class SqliteCache:
    def __init__(self, path: str = "data/cache/connectors.sqlite"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, exp REAL)")
        self.conn.commit()

    @staticmethod
    def key(namespace: str, payload: dict) -> str:
        s = json.dumps({"ns": namespace, **payload}, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        cur = self.conn.execute("SELECT v, exp FROM kv WHERE k=?", (key,))
        row = cur.fetchone()
        if not row: return None
        v, exp = row
        if exp is not None and time.time() > exp:
            self.conn.execute("DELETE FROM kv WHERE k=?", (key,))
            self.conn.commit()
            return None
        return json.loads(v)

    def set(self, key: str, value: Any, ttl: timedelta) -> None:
        exp = time.time() + ttl.total_seconds() if ttl else None
        self.conn.execute("INSERT OR REPLACE INTO kv(k,v,exp) VALUES(?,?,?)", (key, json.dumps(value), exp))
        self.conn.commit()