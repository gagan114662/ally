#!/usr/bin/env python3
import json, os
from pathlib import Path
from ally.tools.ops import ask, ASK_QUEUE

def main():
    # CI-safe dry proof
    res = ask("Ping operator?", {"from":"CI"}, live=False)
    proofs = {
        "ASK_MODE": res.data.get("mode"),
        "ASK_HASH": res.data.get("hash"),
        "ASK_QUEUE_N": res.data.get("queue_len", 1),
    }
    # stdout PROOF lines
    print(f"PROOF:ASK_MODE: {proofs['ASK_MODE']}")
    print(f"PROOF:ASK_HASH: {proofs['ASK_HASH']}")
    print(f"PROOF:ASK_QUEUE_N: {proofs['ASK_QUEUE_N']}")
    # artifacts
    outdir = Path("ask-proof-bundle")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "mask_proofs.json").write_text(json.dumps(proofs, indent=2, sort_keys=True))
    # include queue for audit
    if ASK_QUEUE.exists():
        (outdir / "ask_queue_snapshot.json").write_text(ASK_QUEUE.read_text())

if __name__ == "__main__":
    main()