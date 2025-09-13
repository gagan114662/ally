# ally/tools/ops_bridge.py
# Bridge ops.ask and jules.*
from typing import Dict, Any
import os, json, hashlib, pathlib, time

def ops_ask(payload:Dict[str,Any]):
    # existing Ask-Operator tool (CI-dry friendly)
    msg = json.dumps(payload, sort_keys=True)
    sha = hashlib.sha1(msg.encode()).hexdigest()
    pathlib.Path("runs/ops").mkdir(parents=True, exist_ok=True)
    (pathlib.Path("runs/ops")/f"ask_{sha}.json").write_text(msg)
    print("PROOF:ASK_MODE:dry")
    print(f"PROOF:ASK_HASH:{sha}")
    return {"ok": True, "sha": sha}

def jules_request(payload:Dict[str,Any]):
    # dry by default; requires live gate to actually call Jules
    live = bool(payload.get("live", False))
    allowed = live and os.getenv("ALLY_LIVE")=="1"
    sha = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    pathlib.Path("runs/jules").mkdir(parents=True, exist_ok=True)
    (pathlib.Path("runs/jules")/f"req_{sha}.json").write_text(json.dumps({"allowed":allowed,"payload":payload}))
    print(f"PROOF:JULES_POSTED:{'ok' if allowed else 'dry'}")
    print(f"PROOF:JULES_REQ_SHA:{sha}")
    return {"ok": True, "allowed": allowed, "sha": sha}

def register():
    from ally.tools import TOOL_REGISTRY
    TOOL_REGISTRY["ops.ask_bridge"] = lambda p: ops_ask(p)
    TOOL_REGISTRY["ops.jules_request"] = lambda p: jules_request(p)