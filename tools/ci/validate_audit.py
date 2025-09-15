#!/usr/bin/env python3
import json, sys, pathlib

root = pathlib.Path(".")
audit = root / "artifacts" / "audit_check_ci.json"
transcript = root / "artifacts" / "chat" / "transcript_ci.jsonl"

errors = []

if not audit.exists() or audit.stat().st_size == 0:
    errors.append("Missing or empty artifacts/audit_check_ci.json")

if not transcript.exists() or transcript.stat().st_size == 0:
    errors.append("Missing or empty artifacts/chat/transcript_ci.jsonl")

if audit.exists():
    try:
        data = json.loads(audit.read_text())
        for k in ("ok","missing","mismatches","total_files"):
            if k not in data:
                errors.append(f"audit_check_ci.json missing key: {k}")
        if data.get("ok") is False:
            errors.append("audit_check_ci.json reports ok=false")
    except Exception as e:
        errors.append(f"audit_check_ci.json invalid JSON: {e}")

if errors:
    print("CI VALIDATION ERRORS:")
    for e in errors:
        print(" -", e)
    sys.exit(1)

print("CI validation passed ✅")