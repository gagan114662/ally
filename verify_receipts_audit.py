#!/usr/bin/env python3
"""
verify_receipts_audit.py — Verifies an ALLY_PROOF_BLOCK_V1 against files in the repo.

What it checks:
  1) Block header/footer and required metadata keys (repo, branch, commit, asof_utc).
  2) FILES list: each path exists; file SHA-256 (hex) truncated to 16 equals the claimed hash.
  3) VERIFICATION_REPORT_SHA256 equals the actual sha256 of verification_report.json.
  4) MASTER_PROOF equals the first 32 hex chars of VERIFICATION_REPORT_SHA256.
Outputs:
  - JSON report (see --out) and exit code 0 on success; 1 on any failure.
  - Prints a compact summary and a PROOF:run line you can paste into PRs.

Usage:
  python verify_receipts_audit.py --block-file CHATGPT_AUDIT_READY.md --out artifacts/audit_check.json
  # or
  python verify_receipts_audit.py --stdin --out artifacts/audit_check.json < proof_block.txt
"""

import argparse, hashlib, json, os, re, sys
from datetime import datetime
from typing import Dict, List, Tuple

BLOCK_START = "<<<ALLY_PROOF_BLOCK_V1"
BLOCK_END = ">>>"

REQ_META = ["repo", "branch", "commit", "asof_utc"]

FILE_LINE_RE = re.compile(r"^([^\s:]+):([0-9a-fA-F]{8,64})\s*$")
SHA256_TRUNC = 16  # hex chars
SHA256_LEN = 64
MASTER_LEN = 32

def sha256_hex(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_block(text: str) -> str:
    start = text.find(BLOCK_START)
    end = text.find(BLOCK_END, start + len(BLOCK_START))
    if start == -1 or end == -1:
        raise ValueError("ALLY_PROOF_BLOCK_V1 markers not found")
    return text[start:end+len(BLOCK_END)]

def parse_block(block: str) -> Dict:
    lines = [ln.rstrip() for ln in block.splitlines()]
    meta: Dict[str,str] = {}
    files: List[Tuple[str,str]] = []
    verification_sha256 = None
    master_proof = None

    section = "meta"
    for ln in lines:
        if ln.strip() == BLOCK_START or ln.strip() == BLOCK_END:
            continue
        s = ln.strip()
        if not s:
            continue

        if s == "MASTER_PROOF":
            section = "master"
            continue
        if s.startswith("FILES"):
            section = "files"
            continue
        if s == "VERIFICATION_REPORT_SHA256":
            section = "vrsha"
            continue

        if section == "meta":
            if "=" in s:
                k, v = s.split("=", 1)
                meta[k.strip()] = v.strip()
        elif section == "master":
            master_proof = s
            section = "meta"  # only one line payload
        elif section == "files":
            m = FILE_LINE_RE.match(s)
            if m:
                files.append((m.group(1), m.group(2).lower()))
        elif section == "vrsha":
            verification_sha256 = s.lower()
            section = "meta"

    return {
        "meta": meta,
        "files": files,
        "master_proof": (master_proof or "").lower(),
        "verification_sha256": (verification_sha256 or "").lower(),
    }

def validate_meta(meta: Dict[str,str]) -> List[str]:
    errs = []
    for k in REQ_META:
        if k not in meta or not meta[k]:
            errs.append(f"Missing meta key: {k}")
    # commit looks like 40-hex
    if "commit" in meta and not re.fullmatch(r"[0-9a-fA-F]{40}", meta["commit"]):
        errs.append("commit is not a 40-char hex sha")
    # ISO8601-ish
    if "asof_utc" in meta:
        try:
            datetime.fromisoformat(meta["asof_utc"].replace("Z","+00:00"))
        except Exception:
            errs.append("asof_utc is not ISO8601")
    return errs

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--block-file", help="File containing ALLY_PROOF_BLOCK_V1 (e.g., CHATGPT_AUDIT_READY.md)")
    src.add_argument("--stdin", action="store_true", help="Read proof block from stdin")
    ap.add_argument("--out", default="artifacts/audit_check.json", help="Where to write JSON audit report")
    ap.add_argument("--repo-root", default=".", help="Repo root (default: .)")
    ap.add_argument("--verification-report", default="verification_report.json",
                    help="Path to verification_report.json (default: repo root)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.stdin:
        text = sys.stdin.read()
    else:
        text = read_text(args.block_file)

    try:
        block = extract_block(text)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    try:
        parsed = parse_block(block)
    except Exception as e:
        print(f"[ERROR] Failed to parse block: {e}", file=sys.stderr)
        sys.exit(1)

    meta = parsed["meta"]
    files = parsed["files"]
    master = parsed["master_proof"]
    vr_sha256 = parsed["verification_sha256"]

    errors = []
    warnings = []

    # 1) meta checks
    errors += validate_meta(meta)

    # 2) verification_report.json sha256
    vr_path = os.path.join(args.repo_root, args.verification_report)
    if not os.path.exists(vr_path):
        errors.append(f"Missing {args.verification_report}")
        vr_calc = None
    else:
        vr_calc = sha256_hex(vr_path)
        if not re.fullmatch(r"[0-9a-fA-F]{64}", vr_sha256 or ""):
            errors.append("VERIFICATION_REPORT_SHA256 is not 64-hex")
        elif (vr_sha256 or "") != vr_calc:
            errors.append(f"VERIFICATION_REPORT_SHA256 mismatch: claimed {vr_sha256}, actual {vr_calc}")

    # 3) master proof = first 32 hex chars of verification SHA256
    if master:
        if len(master) != MASTER_LEN or not re.fullmatch(r"[0-9a-fA-F]{32}", master):
            errors.append("MASTER_PROOF must be 32 hex chars")
        elif vr_calc and master != vr_calc[:MASTER_LEN]:
            errors.append(f"MASTER_PROOF mismatch: {master} != {vr_calc[:MASTER_LEN]}")
    else:
        errors.append("MASTER_PROOF missing")

    # 4) files check - using SHA256 truncated to 16 chars
    missing = []
    bad_hash = []
    for path, short_hash in files:
        full = os.path.join(args.repo_root, path)
        if not os.path.exists(full):
            missing.append(path)
            continue
        actual = sha256_hex(full)[:SHA256_TRUNC]
        if actual.lower() != short_hash.lower():
            bad_hash.append({"path": path, "claimed": short_hash, "actual": actual})

    if missing:
        errors.append(f"Missing files: {len(missing)}")
    if bad_hash:
        errors.append(f"Hash mismatches: {len(bad_hash)}")

    report = {
        "meta": meta,
        "counts": {
            "files_claimed": len(files),
            "files_missing": len(missing),
            "files_hash_mismatch": len(bad_hash),
        },
        "missing_files": missing,
        "hash_mismatches": bad_hash,
        "verification_report": {
            "path": args.verification_report,
            "claimed_sha256": vr_sha256,
            "actual_sha256": vr_calc,
            "master_proof": master,
            "master_from_actual": (vr_calc[:MASTER_LEN] if vr_calc else None)
        },
        "errors": errors,
        "warnings": warnings,
        "ok": len(errors) == 0
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    # Human summary
    status = "OK" if report["ok"] else "FAIL"
    print(f"[ALLY VERIFY] {status} — files_claimed={len(files)} missing={len(missing)} mismatches={len(bad_hash)}")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
    if warnings:
        for w in warnings:
            print(f"[WARN] {w}")

    # Emit a deterministic PROOF:run line
    # params_hash := first 8 hex of sha256 over meta+filecount
    ph = hashlib.sha256(json.dumps({"meta": meta, "n": len(files)}, sort_keys=True).encode()).hexdigest()[:8]
    rh = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16]
    print(f"PROOF:run:chatgpt.verify@{ph}:{rh}")

    sys.exit(0 if report["ok"] else 1)

if __name__ == "__main__":
    main()