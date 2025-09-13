import json
from pathlib import Path
from ally.tools import TOOL_REGISTRY

def main():
    proofs = {}
    hb = TOOL_REGISTRY["health.heartbeat"](run_id="HEARTBEAT_CANARY")
    proofs["PROOF:HEARTBEAT_ROTATING"] = "ok" if hb.data["heartbeat_rotating"] else "fail"
    proofs["PROOF:HEARTBEAT_PATH"] = hb.data["heartbeat_path"]

    ks = TOOL_REGISTRY["health.killswitch_drill"](threshold_bps=500.0)
    proofs["PROOF:KILL_SWITCH_DRILL"] = ks.data["kill_switch_drill"]
    proofs["PROOF:KILL_SWITCH_TTR_SEC"] = ks.data["kill_switch_ttr_sec"]
    proofs["PROOF:DRILL_VERSION"] = ks.data["drill_version"]

    outdir = Path("mhealth-proof-bundle"); outdir.mkdir(parents=True, exist_ok=True)
    Path(outdir / "mhealth_proofs.json").write_text(json.dumps(proofs, indent=2))
    for k, v in proofs.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()