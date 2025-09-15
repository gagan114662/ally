#!/usr/bin/env python3
"""
Phase Contract Test Runner
Runs explicit contract tests for Phases 0-12 to verify each phase works as intended
"""
import sys
import subprocess
import glob
import os

PHASE_SUITES = [
    # Phase 0: Gating, receipts & ally proofs aggregation
    "tests/test_gating.py",
    "tests/test_receipts.py",
    "tests/test_simple.py",

    # Phase 1: Data adapters respect double-gate; offline deterministic mocks
    "tests/test_alpha_vantage_*.py",
    "tests/test_polygon_*.py",
    "tests/test_finnhub_*.py",
    "tests/test_data_features.py",

    # Phase 3: Router & risk caps (pre-trade checks, kill-switch drills)
    "tests/test_router_simulator.py",
    "tests/test_broker_risk.py",
    "tests/test_router_mr.py",
    "tests/test_risk_exec.py",

    # Phase 4: Strategy zoo specs validate
    "tests/test_research_pipeline.py",
    "tests/test_detect.py",

    # Phase 5.x: Walk-forward / TS-CV / costs / robustness gates
    "tests/test_walkforward.py",
    "tests/test_ts_cv.py",
    "tests/test_costs.py",
    "tests/test_robustness.py",

    # Phase 6: Evolution & meta-learner fences (only survivors that pass 5.x)
    "tests/test_evolution.py",
    "tests/test_meta_learner.py",

    # Phase 7: Covariance → portfolio → sizing → constraints chain
    "tests/test_ensemble_governance.py",
    "tests/test_sizing_constraints.py",

    # Phase 8: Drift sentinels + guard snapshot determinism
    "tests/test_phase8_guard.py",
    "tests/test_ops_guard.py",

    # Phase 9: Ensemble governance (weight caps, correlation caps, turnover)
    "tests/test_ensemble_ops.py",

    # Phase 10: Execution simulator (slippage/latency) + journaling
    "tests/test_execution_simulator.py",

    # Phase 11: Status/telemetry/journal
    "tests/test_status_runbook.py",
    "tests/test_status_journal.py",
    "tests/test_status_telemetry.py",

    # Phase 12: Chat/TUI transcript determinism
    "tests/test_tui_chat.py",

    # Orchestrator fixes
    "tests/test_orchestrator_fixes.py",

    # Additional tests found in repository
    "tests/test_cv_detect.py",
    "tests/test_mcache_runtime.py",
    "tests/test_mcp.py",
    "tests/test_memory_reporting.py",
    "tests/test_nlp_events.py",
    "tests/test_timestamp_serialization.py",
    "tests/test_web_tools.py",
]

def expand_test_patterns(patterns):
    """Expand glob patterns and return existing files only"""
    existing_files = []
    for pattern in patterns:
        if '*' in pattern:
            # Expand glob pattern
            matches = glob.glob(pattern)
            existing_files.extend(matches)
        elif os.path.exists(pattern):
            # Direct file exists
            existing_files.append(pattern)
        else:
            print(f"⚠️  Test file not found: {pattern}")

    return existing_files

def main():
    print("🧪 Running Phase Contract Test Suite")
    print("=" * 60)

    # Expand patterns to actual files
    test_files = expand_test_patterns(PHASE_SUITES)

    print(f"📊 Found {len(test_files)} test files to run")
    for i, test_file in enumerate(test_files[:10], 1):  # Show first 10
        print(f"  {i}. {test_file}")
    if len(test_files) > 10:
        print(f"  ... and {len(test_files) - 10} more")

    if not test_files:
        print("❌ No test files found - cannot validate phases")
        return 1

    print("\n🚀 Starting pytest execution...")

    # Build pytest command
    args = [
        "pytest",
        "-q",
        "--maxfail=5",
        "--disable-warnings",
        "--junitxml=artifacts/ci/junit.xml",
        "--cov=ally",
        "--cov-report=xml:artifacts/ci/coverage.xml",
        "--cov-report=term-missing"
    ] + test_files

    print(f"Command: {' '.join(args[:6])} ... ({len(test_files)} test files)")

    # Run pytest
    try:
        result = subprocess.call(args)
        print(f"\n📋 Pytest completed with exit code: {result}")
        return result
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())