# ally/autopilot/loop.py
from dataclasses import dataclass
from datetime import timedelta, datetime, timezone
from typing import Callable, List, Optional, Dict, Any
import json, os, time, hashlib, pathlib
from ally.tools import TOOL_REGISTRY

@dataclass
class LoopPolicy:
    max_rounds:int=6
    max_wall_secs:int=900
    max_same_output_rounds:int=2
    backoff_base:float=1.0
    escalate_round:int=3

@dataclass
class LoopEvent:
    ts:str
    kind:str
    detail:Dict[str,Any]

class NeverStuckLoop:
    """
    Try → Verify → Repair → (bounded retries) → Escalate (Ask-Operator → Jules) → Exit-with-proof.
    Guarantees progress or exit with artifacts + receipts (no infinite loops).
    """
    def __init__(self, task_id:str, policy:LoopPolicy=LoopPolicy()):
        self.task_id = task_id
        self.policy = policy
        self.events:List[LoopEvent]=[]
        self.out_hashes:List[str]=[]
        self.started = datetime.now(timezone.utc)
        self.run_dir = pathlib.Path(f"runs/NS_{self.started:%Y%m%dT%H%M%SZ}_{task_id}")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _stamp(self, kind, **detail):
        ev = LoopEvent(datetime.now(timezone.utc).isoformat(), kind, detail)
        self.events.append(ev)

    def _hash_out(self, out:Any)->str:
        b = json.dumps(out, sort_keys=True, default=str).encode()
        return hashlib.sha1(b).hexdigest()

    def _progress_ok(self, out_hash:str)->bool:
        # Prevent "busy loop": require output novelty.
        n_same = sum(1 for h in self.out_hashes[-self.policy.max_same_output_rounds:] if h==out_hash)
        return n_same < self.policy.max_same_output_rounds

    def run(self, plan:Callable[[],Any], verify:Callable[[Any],Dict[str,Any]], repair:Callable[[Any,Dict[str,Any]],Any]):
        t0 = time.time()
        rounds=0
        while True:
            if time.time()-t0 > self.policy.max_wall_secs or rounds>=self.policy.max_rounds:
                self._stamp("exit", reason="limits")
                break

            try:
                out = plan()
                out_hash = self._hash_out(out)
                self.out_hashes.append(out_hash)
                if not self._progress_ok(out_hash):
                    self._stamp("stalled", hash=out_hash)
                    # escalate
                    self._escalate(out, why="no_progress")
                    break

                verdict = verify(out)
                self._stamp("verify", **verdict)
                if verdict.get("ok"):
                    self._stamp("success", hash=out_hash)
                    break

                out = repair(out, verdict)
                self._stamp("repair", status="attempted")
            except Exception as e:
                # capture repro + trace
                if "debug.capture_trace" in TOOL_REGISTRY:
                    TOOL_REGISTRY["debug.capture_trace"](f"neverstuck_{self.task_id}", str(e))
                if "debug.make_repro" in TOOL_REGISTRY:
                    TOOL_REGISTRY["debug.make_repro"]("neverstuck.plan", {"task_id": self.task_id}, seed=1337)
                self._stamp("exception", msg=str(e))
                # bounded backoff then escalate if needed
                rounds += 1
                if rounds >= self.policy.escalate_round:
                    self._escalate({"last_error": str(e)}, why="exception")
                    break
                time.sleep(self.policy.backoff_base * (2**(rounds-1)))
                continue
            rounds += 1

        # write proof bundle
        proof = {
            "task_id": self.task_id,
            "rounds": rounds,
            "events": [e.__dict__ for e in self.events],
            "exit_reason": self.events[-1].detail.get("reason","ok") if self.events else "ok"
        }
        (self.run_dir/"neverstuck_proofs.json").write_text(json.dumps(proof, indent=2))
        print(f"PROOF:NEVERSTUCK_ROUNDS:{rounds}")
        print(f"PROOF:LOOP_EXIT:{proof['exit_reason']}")
        print(f"PROOF:ESCALATIONS:{sum(1 for e in self.events if e.kind=='escalate')}")
        return proof

    def _escalate(self, context:Any, why:str):
        # 1) Ask-Operator (internal)
        if "ops.ask" in TOOL_REGISTRY:
            TOOL_REGISTRY["ops.ask"](question=f"[NEVERSTUCK] Escalation ({why})", context=context, live=False)
        # 2) Jules bridge (dry/live guarded)
        if "ops.jules_request" in TOOL_REGISTRY:
            TOOL_REGISTRY["ops.jules_request"]({"title": f"Ally stuck: {self.task_id}", "payload": context, "live": os.getenv("ALLY_LIVE")=="1"})
        self._stamp("escalate", target=["ops.ask","ops.jules_request"], why=why)