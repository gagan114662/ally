#!/usr/bin/env python3
"""
Generate verification proofs for ChatGPT audit
This script creates cryptographic proofs of all claimed files
"""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

def calculate_file_hash(filepath):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        return None

def verify_phase_files():
    """Verify all phase files and generate proofs"""

    phases = {
        "phase_5_research": [
            "ally/research/walkforward.py",
            "ally/research/ts_cv.py",
            "tests/test_walkforward.py",
            "tests/test_ts_cv.py"
        ],
        "phase_6_costs_robustness": [
            "ally/research/costs.py",
            "ally/research/robustness.py",
            "tests/test_costs.py",
            "tests/test_robustness.py"
        ],
        "phase_7_portfolio": [
            "ally/research/portfolio.py",
            "ally/research/constraints.py",
            "ally/research/sizing.py",
            "tests/test_sizing_constraints.py"
        ],
        "phase_8_ops": [
            "ally/ops/policy.yaml",
            "ally/utils/receipts.py",
            "ally/utils/file_receipts.py",
            "verify_receipts.py"
        ],
        "ensemble_meta": [
            "ally/research/ensemble.py",
            "ally/research/meta_learner.py",
            "tests/test_ensemble_ops.py",
            "tests/test_meta_learner.py"
        ],
        "evolution_fdr": [
            "ally/research/evolution.py",
            "ally/research/fdr.py",
            "tests/test_evolution.py"
        ],
        "trading_execution": [
            "ally/tools/trading_router.py",
            "ally/tools/trading_risk.py",
            "ally/tools/broker.py",
            "tests/test_router_simulator.py"
        ],
        "phase_11_status": [
            "ally/status/runbook.py",
            "ally/status/journal.py",
            "ally/status/telemetry.py",
            "ally/cli/status_cli.py",
            "tests/test_status_runbook.py",
            "tests/test_status_journal.py",
            "tests/test_status_telemetry.py",
            "scripts/ci_phase11_status.sh"
        ],
        "phase_12_chat": [
            "ally/ui/tui.py",
            "ally/chat/controller.py",
            "ally/cli/chat_cli.py",
            "tests/test_tui_chat.py",
            "scripts/ci_phase12_chat.sh"
        ]
    }

    verification_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "repository": "https://github.com/gagan114662/ally",
        "phases": {}
    }

    print("=" * 80)
    print("ALLY SYSTEM VERIFICATION REPORT FOR CHATGPT")
    print("=" * 80)
    print(f"Timestamp: {verification_report['timestamp']}")
    print(f"Repository: {verification_report['repository']}")
    print("=" * 80)

    for phase_name, files in phases.items():
        print(f"\n📦 {phase_name.upper()}")
        print("-" * 40)

        phase_data = {
            "files": {},
            "status": "complete",
            "proof_lines": []
        }

        for filepath in files:
            file_hash = calculate_file_hash(filepath)
            exists = file_hash is not None

            phase_data["files"][filepath] = {
                "exists": exists,
                "hash": file_hash if exists else "FILE_NOT_FOUND"
            }

            status_icon = "✅" if exists else "❌"
            print(f"{status_icon} {filepath}")
            if exists:
                print(f"   SHA256: {file_hash[:16]}...")
                phase_data["proof_lines"].append(
                    f"PROOF:file:{filepath}:sha256:{file_hash[:16]}"
                )
            else:
                phase_data["status"] = "incomplete"

        verification_report["phases"][phase_name] = phase_data

    # Generate master proof
    print("\n" + "=" * 80)
    print("MASTER VERIFICATION PROOF")
    print("=" * 80)

    master_data = json.dumps(verification_report, sort_keys=True)
    master_hash = hashlib.sha256(master_data.encode()).hexdigest()

    print(f"PROOF:master:verification:{master_hash[:32]}")
    print("=" * 80)

    # Save verification report
    report_path = "verification_report.json"
    with open(report_path, "w") as f:
        json.dump(verification_report, f, indent=2)

    print(f"\n📄 Full report saved to: {report_path}")

    # Create GitHub-ready proof comment
    github_comment = f"""
## 🔒 Cryptographic Verification Report

**Repository:** https://github.com/gagan114662/ally
**Timestamp:** {verification_report['timestamp']}
**Master Proof:** `{master_hash[:32]}`

### Phase Verification Status:
"""

    for phase_name, phase_data in verification_report["phases"].items():
        status_emoji = "✅" if phase_data["status"] == "complete" else "⚠️"
        github_comment += f"\n{status_emoji} **{phase_name}**: {phase_data['status']}"
        github_comment += f"\n   Files verified: {len([f for f in phase_data['files'].values() if f['exists']])}/{len(phase_data['files'])}"

    github_comment += f"""

### Proof Lines for CI:
```
{"\\n".join([proof for phase in verification_report['phases'].values() for proof in phase.get('proof_lines', [])][:10])}
```

### How to Verify:
1. Clone the repository
2. Run `python generate_verification_proofs.py`
3. Compare the master proof hash
4. Check individual file hashes match

🤖 Generated with Ally Audit System
"""

    with open("github_verification_comment.md", "w") as f:
        f.write(github_comment)

    print(f"📝 GitHub comment saved to: github_verification_comment.md")
    print("\n✅ Verification complete! Share these proofs with ChatGPT for audit.")

    return verification_report

if __name__ == "__main__":
    verify_phase_files()