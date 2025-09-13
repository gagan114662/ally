from __future__ import annotations
import json, hashlib, os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml

# Ally tools
from ally.tools.qc_templates import qc_generate_python
from ally.tools.qc_lean import qc_smoke_run  # the same harness used in M-QC Gate

CFG = yaml.safe_load(Path("ally/qc/canary_config.yaml").read_text())
TZ = timezone.utc

def _iso(d: datetime) -> str: return d.astimezone(TZ).strftime("%Y-%m-%d")

def main():
    # 1) Date window (last N calendar days in UTC)
    end = datetime.now(TZ).date()
    start = end - timedelta(days=int(CFG.get("days", 5)))
    start_s, end_s = _iso(datetime.combine(start, datetime.min.time(), TZ)), _iso(datetime.combine(end, datetime.min.time(), TZ))

    # 2) Generate a minimal QC algorithm from template
    symbols = [s["ticker"] for s in CFG["symbols"]]
    r = qc_generate_python(
        class_name="AllyCanary",
        symbols=symbols,
        start=start_s, end=end_s,
        warmup_bars=10,
        trade_logic="if not self.Portfolio.Invested:\n    self.SetHoldings(self.symbols[0], 0.5)\n"
    )
    algo_path = r.data["algo_path"]

    # 3) Run Lean smoke backtest
    run = qc_smoke_run(algo_path, max_minutes=5)
    ok = run.ok

    # 4) Compute a deterministic hash over key outputs
    # Prefer result hash if provided by smoke harness; otherwise hash engine/backtest logs if exposed.
    result_hash = run.data.get("result_hash")
    if not result_hash:
        payload = json.dumps(run.data, sort_keys=True).encode() if run.data else b"{}"
        result_hash = hashlib.sha1(payload).hexdigest()

    proofs = {
        "canary_ok": bool(ok),
        "canary_hash": result_hash,
        "symbols": symbols,
        "window": {"start": start_s, "end": end_s},
    }

    # 5) Compare with repo-tracked baseline (if present)
    baseline = CFG.get("baseline_hash")
    if baseline:
        proofs["diff"] = (baseline != result_hash)
    else:
        proofs["diff"] = None  # no baseline yet

    # 6) Emit PROOF lines + artifact
    out_dir = Path("canary-proof-bundle"); out_dir.mkdir(parents=True, exist_ok=True)
    Path(out_dir/"canary_proofs.json").write_text(json.dumps(proofs, indent=2))
    print("PROOF:QC_CANARY:", "ok" if ok else "fail")
    print("PROOF:QC_CANARY_HASH:", result_hash)
    if baseline:
        print("PROOF:QC_CANARY_DIFF:", 1 if baseline != result_hash else 0)

    # Non-zero exit on fail so nightly alerts
    if not ok:
        raise SystemExit(1)

if __name__ == "__main__":
    main()