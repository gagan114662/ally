from __future__ import annotations
import os
import json
import hashlib
from typing import Dict, List, Optional
from ally.schemas.base import ToolResult, Meta
from ally.providers.github_client import GH

REQUIRED_KEYS = {
    "M11 (T-Costs)": ["TCOST_CONFIG", "TCOST_IMPACT_BPS", "FILLS_FINGERPRINT"],
    "M10 (WFO)": ["WFO_SPLITS", "WFO_RESULTS_HASH", "DEFLATED_SHARPE", "SPA_PVALUE", "KPI_STABILITY"],
    "M-Data Connectors": ["DATA_SOURCES", "ADAPTER_SMOKE", "CACHE_HIT_RATE", "RATE_LIMITER"],
    "M-Research": ["RESEARCH_SOURCES", "EVIDENCE_COUNTS", "POSTERIOR_NEUTRAL", "DEDUPE_RATE", "SUMMARY_HASH"],
    "M9 (Orchestrator)": ["ORCH_SUMMARY_HASH", "ORCH_RUN_ID", "REPORT_PATH", "MEMORY_ROWS"],
    "M8 (Memory & Reporting)": ["TOOL_REGISTRY", "MEMORY_LOG_OK", "REPORT_OK", "REPORT_SUMMARY_HASH"],
    "M-Reliability": ["RELIABILITY"],
    "M-Assistant (GPT Copilot)": ["ASSISTANT_PROVIDER", "ASSISTANT_SUMMARY_HASH"],
    "M11 (T-Costs) optional": []  # allow extensibility
}


def _flatten(d: Dict) -> List[str]:
    lines = []
    
    def rec(k, v):
        if isinstance(v, dict):
            for kk, vv in v.items():
                rec(f"{k}.{kk}", vv)
        else:
            lines.append(f"{k}={v}")
    
    for k, v in d.items():
        rec(k, v)
    return lines


def review_pr(pr_number: int, owner: Optional[str] = None, repo: Optional[str] = None) -> ToolResult:
    owner = owner or os.getenv("GITHUB_OWNER")
    repo = repo or os.getenv("GITHUB_REPO")
    gh = GH(owner, repo)
    
    try:
        pr = gh.pr_by_number(pr_number)
        sha = pr["head"]["sha"]
        branch = pr["head"]["ref"]
        runs = gh.list_workflow_runs(branch)
        
        # Pick latest run for this sha
        run = next((r for r in runs if r["head_sha"] == sha), None)
        artifacts = gh.list_run_artifacts(run["id"]) if run else []
        proofs: Dict[str, Dict] = {}
        missing: Dict[str, List[str]] = {}
        total_artifacts = 0

        for a in artifacts:
            name = a["name"]
            if name.endswith("proof-bundle"):
                try:
                    data = gh.download_artifact_json(a["id"])
                    total_artifacts += 1
                    proofs[name] = data
                except Exception as e:
                    # Skip artifacts that can't be downloaded/parsed
                    pass

        # Summarize: which required proof groups are satisfied?
        verdict_ok = True
        groups_ok = {}
        for job, keys in REQUIRED_KEYS.items():
            # find any artifact that contains those keys
            ok = False
            for art, data in proofs.items():
                if all(k in data for k in keys):
                    ok = True
                    break
            if keys:  # only check non-empty groups strictly
                groups_ok[job] = ok
                if not ok:
                    verdict_ok = False

        digest = hashlib.sha1(json.dumps({"sha": sha, "groups_ok": groups_ok}, sort_keys=True).encode()).hexdigest()
        summary = {
            "pr": pr_number,
            "sha": sha,
            "artifact_count": total_artifacts,
            "groups_ok": groups_ok,
            "verdict": "approve" if verdict_ok else "request_changes",
            "digest": digest
        }

        body = [
            "### 🤖 Ally Reviewer — Automated Proof Check",
            f"- PR: #{pr_number}",
            f"- SHA: `{sha[:7]}`",
            f"- Artifacts found: **{total_artifacts}**",
            f"- Verdict: **{summary['verdict']}**",
            "",
            "#### Groups",
            *[f"- {k}: {'✅' if v else '❌'}" for k, v in groups_ok.items()],
            "",
            "#### Digest",
            f"`{digest}`",
        ]
        
        gh.comment_pr(pr_number, "\n".join(body))
        gh.set_check(sha, "Ally Reviewer", "completed", "success" if verdict_ok else "failure", f"Verdict: {summary['verdict']}")

        meta = Meta(ts=None, duration_ms=0, provenance={"tool_name": "reviewer.check_pr"})
        return ToolResult(ok=True, data=summary, errors=[], meta=meta)
    
    except Exception as e:
        meta = Meta(ts=None, duration_ms=0, provenance={"tool_name": "reviewer.check_pr"})
        return ToolResult(
            ok=False,
            data={"error": str(e), "pr": pr_number},
            errors=[str(e)],
            meta=meta
        )