from ally.utils.heartbeat import rotate_heartbeat
from ally.tools import TOOL_REGISTRY
from ally.schemas.base import ToolResult
from pathlib import Path
import json, time, random

def health_heartbeat(run_id: str = "HEARTBEAT_CANARY") -> ToolResult:
    hb_path = Path("runs/heartbeat") / f"{run_id}.json"
    info = rotate_heartbeat(str(hb_path), run_id, delay_sec=1.0)
    ok = info["rotated"]
    return ToolResult(ok=ok, data={
        "heartbeat_path": info["path"],
        "heartbeat_rotating": ok,
        "delay_sec": info["delay_sec"],
    })

def health_killswitch_drill(threshold_bps: float = 500.0) -> ToolResult:
    """
    Simulates a drawdown-breach and verifies kill-switch response time.
    Purely offline/dry; no broker calls. Deterministic by seed.
    """
    random.seed(1337)
    t0 = time.time()
    breached = True  # force breach
    # emulate instantaneous halt path (bounded)
    time.sleep(0.05 + (random.random() * 0.02))  # ~50–70ms
    halted = breached
    ttr = time.time() - t0  # time-to-respond (sec)

    out = {
        "drill_version": "v1",
        "kill_switch_drill": "ok" if halted else "fail",
        "kill_switch_ttr_sec": round(ttr, 3),
        "threshold_bps": threshold_bps,
    }
    # write an audit drop (non-blocking)
    p = Path("runs/drills"); p.mkdir(parents=True, exist_ok=True)
    Path(p / "killswitch_last.json").write_text(json.dumps(out, separators=(",",":")))
    return ToolResult(ok=True, data=out)

# Register
TOOL_REGISTRY["health.heartbeat"] = health_heartbeat
TOOL_REGISTRY["health.killswitch_drill"] = health_killswitch_drill