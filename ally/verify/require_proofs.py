import json
import sys
import pathlib

REQUIRED = {
    "m8": ["TOOL_REGISTRY", "MEMORY_LOG_OK", "REPORT_OK", "REPORT_SUMMARY_HASH"],
    "m9": ["ORCH_SUMMARY_HASH", "ORCH_RUN_ID", "REPORT_PATH", "MEMORY_ROWS"],
    "m10": ["WFO_SPLITS", "WFO_RESULTS_HASH", "DEFLATED_SHARPE", "SPA_PVALUE", "KPI_STABILITY"],
    "mdata": ["DATA_SOURCES", "ADAPTER_SMOKE", "CACHE_HIT_RATE", "RATE_LIMITER"],
    "mresearch": ["RESEARCH_SOURCES", "EVIDENCE_COUNTS", "POSTERIOR_NEUTRAL", "DEDUPE_RATE", "SUMMARY_HASH"],
    "m11": ["TCOST_CONFIG", "TCOST_IMPACT_BPS", "FILLS_FINGERPRINT"],
    "mreliability": ["RELIABILITY"],  # this PR
}


def main(kind: str, path: str):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"Missing proof file: {path}", file=sys.stderr)
        sys.exit(2)
    data = json.loads(p.read_text())
    missing = [k for k in REQUIRED[kind] if k not in data]
    if missing:
        print(f"Missing proof keys: {missing}", file=sys.stderr)
        sys.exit(3)
    print("PROOF OK")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])