# ally/tools/jules.py
# Live helper with receipts + double gate
import os, json, hashlib, pathlib, time
from datetime import datetime, timezone
from ally.schemas.base import ToolResult

def _write_receipt(kind, payload):
    sha = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    p = pathlib.Path("runs/receipts"); p.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "sha1": sha, "ts": datetime.now(timezone.utc).isoformat(), "payload": payload}
    (p/f"jules_{sha}.json").write_text(json.dumps(rec, indent=2))
    print(f"PROOF:JULES_RECEIPT:{sha}")
    return sha

def jules_help(request) -> ToolResult:
    live = request.get("live", False)
    allowed = live and os.getenv("ALLY_LIVE")=="1"
    # dry path: write stub + receipt
    sha = _write_receipt("jules_request", {"allowed": allowed, "request": request})
    return ToolResult(ok=True, data={"allowed": allowed, "receipt": sha})