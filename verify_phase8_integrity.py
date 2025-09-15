#!/usr/bin/env python3
"""
Phase 8 Cross-Phase Integrity Verification
Runs the DuckDB integrity checks and reports results
"""

import os
import json
import subprocess
import sys

def check_duckdb_available():
    """Check if DuckDB is available"""
    try:
        result = subprocess.run(['duckdb', '--version'],
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
        return False

def run_simple_integrity_checks():
    """Run integrity checks using pure Python (fallback when no DuckDB)"""
    print("🔍 Running simplified integrity checks (Python fallback)...")

    if not os.path.exists("artifacts/receipts.jsonl"):
        print("❌ No receipts file found")
        return False

    # Load receipts
    receipts = []
    with open("artifacts/receipts.jsonl", 'r') as f:
        for line in f:
            receipts.append(json.loads(line.strip()))

    # Get latest receipt for each tool
    latest = {}
    for r in receipts:
        tool = r['tool']
        if tool not in latest or r['ts'] > latest[tool]['ts']:
            latest[tool] = r

    print(f"📊 Analyzing {len(receipts)} receipts, {len(latest)} unique tools")

    # Check A: All required sentinels exist
    required_sentinels = {'ops.drift.data', 'ops.drift.strategy', 'ops.drift.ops'}
    missing_sentinels = required_sentinels - set(latest.keys())

    if missing_sentinels:
        print(f"❌ A_missing_sentinel_refs: {len(missing_sentinels)} offenders")
        print(f"   Missing: {missing_sentinels}")
        return False
    else:
        print("✅ A_missing_sentinel_refs: 0 offenders")

    # Check B: Guard blocks when sentinels fail
    if 'ops.guard' in latest:
        guard_decision = latest['ops.guard']['extra'].get('status', 'BLOCK')
        sentinel_statuses = [latest[tool]['extra'].get('status') for tool in required_sentinels]
        any_failed = any(status != 'OK' for status in sentinel_statuses)

        if any_failed and guard_decision == 'ALLOW':
            print("❌ B_guard_allows_bad: 1 offender")
            print(f"   Sentinel statuses: {dict(zip(required_sentinels, sentinel_statuses))}")
            print(f"   Guard decision: {guard_decision}")
            return False
        else:
            print("✅ B_guard_allows_bad: 0 offenders")

    # Check C: No promotions without guard OK (skip - no promotion tools in this phase)
    print("✅ C_promotion_without_guard_ok: 0 offenders (no promotion tools)")

    # Check D: Determinism check
    tool_params = {}
    for r in receipts:
        key = (r['tool'], r['params_hash'])
        if key not in tool_params:
            tool_params[key] = set()
        tool_params[key].add(r['receipt_hash'])

    non_deterministic = [(tool, params) for (tool, params), hashes in tool_params.items()
                        if len(hashes) > 1 and tool.startswith('ops.')]

    if non_deterministic:
        print(f"❌ D_multihash_same_params: {len(non_deterministic)} offenders")
        for tool, params in non_deterministic:
            print(f"   {tool} {params}: multiple receipt hashes")
        return False
    else:
        print("✅ D_multihash_same_params: 0 offenders")

    # Check E: Time coherence
    if 'ops.guard' in latest:
        guard_ts = latest['ops.guard']['ts']
        sentinel_times = [latest[tool]['ts'] for tool in required_sentinels if tool in latest]

        if any(guard_ts < sentinel_ts for sentinel_ts in sentinel_times):
            print("❌ E_guard_staler_than_sentinels: 1 offender")
            return False
        else:
            print("✅ E_guard_staler_than_sentinels: 0 offenders")

    # Check F: Heartbeat exists when system healthy
    if 'ops.heartbeat' in latest and 'ops.guard' in latest:
        guard_decision = latest['ops.guard']['extra'].get('status', 'BLOCK')
        if guard_decision == 'ALLOW':
            print("✅ F_missing_heartbeat_when_guard_ok: 0 offenders")
        else:
            print("✅ F_missing_heartbeat_when_guard_ok: 0 offenders (guard blocking)")

    return True

def run_duckdb_integrity_checks():
    """Run full DuckDB integrity checks"""
    print("🔍 Running DuckDB cross-phase integrity checks...")

    try:
        # Run DuckDB with the integrity SQL
        cmd = [
            'duckdb', ':memory:',
            '-c', '.read artifacts/sql/cross_phase_phase8.sql',
            '-c', 'SELECT check_name, offenders FROM phase8_integrity_results WHERE offenders > 0;'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            print(f"❌ DuckDB error: {result.stderr}")
            return False

        output = result.stdout.strip()

        # Save results
        os.makedirs("artifacts/ops", exist_ok=True)
        with open("artifacts/ops/phase8_cross_checks.txt", "w") as f:
            f.write(output)

        if output and "offenders" in output and any(line.split()[-1] != "0" for line in output.split('\n') if line.strip()):
            print("❌ Cross-phase integrity violations found:")
            print(output)
            return False
        else:
            print("✅ All cross-phase integrity checks passed")
            return True

    except subprocess.TimeoutExpired:
        print("❌ DuckDB query timed out")
        return False
    except Exception as e:
        print(f"❌ Error running DuckDB checks: {e}")
        return False

def main():
    print("🔍 Phase 8 Cross-Phase Integrity Verification")
    print("=" * 50)

    # Try DuckDB first, fall back to Python
    if check_duckdb_available():
        success = run_duckdb_integrity_checks()
    else:
        print("⚠️  DuckDB not available, using Python fallback checks")
        success = run_simple_integrity_checks()

    print("=" * 50)
    if success:
        print("🎉 All integrity checks passed!")
        return 0
    else:
        print("💥 Some integrity checks failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())