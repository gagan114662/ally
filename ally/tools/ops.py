# ally/tools/ops.py
from __future__ import annotations
import os, json, hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from ally.tools.registry import TOOL_REGISTRY, ToolResult

ASK_QUEUE = Path("runs/supervisor/ask_queue.json")

def _sha1(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def enqueue(tool:str, inputs:dict)->ToolResult:
    qf = Path("runs/supervisor/queue.json"); qf.parent.mkdir(parents=True, exist_ok=True)
    queue = json.loads(qf.read_text()) if qf.exists() else []
    queue.append({"tool":tool,"inputs":inputs})
    qf.write_text(json.dumps(queue, indent=2, sort_keys=True))
    return ToolResult(ok=True, data={"queued":len(queue)})

def ask(question: str,
        context: Optional[Dict[str, Any]] = None,
        live: bool = False,
        router_hint: str = "nlp") -> ToolResult:
    """
    Ask the human/assistant. Deterministic, CI-safe.
    - CI/dry: append to queue file, return mode='dry'
    - Live: requires live=True AND ALLY_LIVE=1; calls runtime.generate deterministically
    """
    payload = {"q": question, "ctx": context or {}, "router_hint": router_hint}
    h = _sha1(payload)

    # LIVE path (guarded)
    if live and os.getenv("ALLY_LIVE") == "1":
        # Use runtime+router deterministically
        if "runtime.generate" not in TOOL_REGISTRY:
            return ToolResult(ok=False, errors=["runtime.generate not available"], data={"hash": h, "mode": "live"})
        out = TOOL_REGISTRY["runtime.generate"](
            task="ops.ask",
            prompt=question,
            context=context or {},
            router_hint=router_hint,
            temperature=0,
            seed=1337
        )
        # Pass through result text if available
        answer = None
        if out and out.ok:
            # expect .data to be dict with 'text' or similar
            if isinstance(out.data, dict):
                answer = out.data.get("text") or out.data.get("output") or out.data.get("answer")
        return ToolResult(ok=True, data={"hash": h, "mode": "live", "answer": answer})

    # DRY path (CI safe): persist ask to queue
    ASK_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    queue = []
    if ASK_QUEUE.exists():
        try:
            queue = json.loads(ASK_QUEUE.read_text())
        except Exception:
            queue = []
    record = {"hash": h, "payload": payload}
    queue.append(record)
    ASK_QUEUE.write_text(json.dumps(queue, indent=2, sort_keys=True))
    return ToolResult(ok=True, data={"hash": h, "mode": "dry", "queue_len": len(queue), "live_denied": live and os.getenv("ALLY_LIVE") != "1"})

def register():
    TOOL_REGISTRY["ops.ask"] = ask
    TOOL_REGISTRY["ops.enqueue"] = enqueue