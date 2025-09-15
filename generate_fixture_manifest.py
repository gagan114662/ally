#!/usr/bin/env python3
"""
Generate fixture manifest for research orchestrator

Creates artifacts/fixtures/manifest.json with hash and size
for each required fixture file for verification.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

def calculate_file_hash(filepath: Path) -> dict:
    """Calculate hash and size for a file"""
    try:
        content = filepath.read_bytes()
        return {
            "exists": True,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest()[:16],
            "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
        }
    except FileNotFoundError:
        return {
            "exists": False,
            "size": 0,
            "sha256": None,
            "modified": None
        }

def main():
    """Generate fixture manifest"""
    # Required fixtures from orchestrator_fixes.py
    required_fixtures = [
        "data/fixtures/factor_panels.json",
        "data/fixtures/synthetic_ohlcv.json",
        "data/fixtures/walk_forward_results.json",
        "artifacts/scoring/offline_benchmarks.json"
    ]

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "generator": "generate_fixture_manifest.py",
        "fixtures": {}
    }

    print("📄 Generating Fixture Manifest")

    for fixture_path in required_fixtures:
        filepath = Path(fixture_path)
        file_info = calculate_file_hash(filepath)
        manifest["fixtures"][fixture_path] = file_info

        status = "✅" if file_info["exists"] else "❌"
        size = f"{file_info['size']} bytes" if file_info["exists"] else "missing"
        print(f"   {status} {fixture_path} ({size})")

    # Save manifest
    os.makedirs("artifacts/fixtures", exist_ok=True)
    manifest_path = "artifacts/fixtures/manifest.json"

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"📋 Manifest saved to: {manifest_path}")

    # Summary
    total = len(required_fixtures)
    present = sum(1 for f in manifest["fixtures"].values() if f["exists"])
    print(f"📊 Fixtures: {present}/{total} present")

    return present == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)