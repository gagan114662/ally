"""
QuantConnect (LEAN) project bootstrapper for Ally.

Creates a minimal QC project directory with:
- config.live.paper.json (PaperBrokerage, params injected)
- algorithm.py (rendered from Jinja2 template or simple format)
- orders_inbox.jsonl (empty; used for Ally -> QC intents)

Every created file is receipt-hashed (SHA-1) and recorded in DuckDB.
No network calls. Safe for CI.

Usage (offline/CI):
    from ally.integrations.quantconnect.qc_project_bootstrap import bootstrap_qc_project
    meta = bootstrap_qc_project(
        project_slug="ally-paper-aapl",
        symbols=["AAPL","MSFT"],
        out_root=".ally_qc",
        templates_root="ally/integrations/quantconnect/templates",
        results_dir="./qc-results",
        data_dir="./qc-data",
        params={"ALLY_INBOX": "orders_inbox.jsonl"},
        db_path="artifacts/proof_receipts.duckdb",
        deterministic=True,
    )
"""

from __future__ import annotations
import os, json, pathlib, datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from ...utils.hashing import hash_payload  # your Phase 0 helper (SHA-1 hex)
from ...utils.file_receipts import record_file_receipt


@dataclass(frozen=True)
class BootstrapResult:
    project_dir: str
    config_path: str
    algorithm_path: str
    inbox_path: str
    params_hash: str      # 8 chars
    receipts: Dict[str, str]  # {"config": <sha1_16>, "algorithm": <sha1_16>, "inbox": <sha1_16>}
    utc_timestamp: str


def _first8(hexstr: str) -> str:
    return hexstr[:8]


def _first16(hexstr: str) -> str:
    return hexstr[:16]


def _write_text(path: str, text: str) -> bytes:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return text.encode("utf-8")


def bootstrap_qc_project(
    project_slug: str,
    symbols: List[str],
    out_root: str = ".ally_qc",
    templates_root: str = "ally/integrations/quantconnect/templates",
    results_dir: str = "./qc-results",
    data_dir: str = "./qc-data",
    params: Optional[Dict[str, str]] = None,
    db_path: str = "artifacts/proof_receipts.duckdb",
    deterministic: bool = True,
) -> BootstrapResult:
    """
    Create a deterministic QC project skeleton for PaperBrokerage runs.

    OFFLINE ONLY: no network, safe for CI.

    - Writes config.live.paper.json with injected parameters
    - Renders algorithm.py from template (simple format fallback if Jinja2 is absent)
    - Creates empty orders_inbox.jsonl for Ally -> QC intents
    - Receipts for every file are saved to DuckDB

    Returns:
        BootstrapResult with paths and 16-char receipt hashes.

    Raises:
        AssertionError if inputs invalid or symbols empty.
        OSError on file IO issues.
    """

    assert project_slug.strip(), "project_slug is required"
    assert symbols and all(isinstance(s, str) and s.strip() for s in symbols), "symbols must be non-empty strings"

    # Normalize & compute params hash (8 chars) for reproducibility labels
    norm = {
        "project_slug": project_slug,
        "symbols": [s.strip().upper() for s in symbols],
        "results_dir": str(results_dir),
        "data_dir": str(data_dir),
        "params": dict(sorted((params or {}).items())),
        "deterministic": bool(deterministic),
    }
    params_hash8 = _first8(hash_payload(json.dumps(norm, sort_keys=True).encode("utf-8")))

    # Paths
    project_dir = os.path.join(out_root, project_slug)
    config_path = os.path.join(project_dir, "config.live.paper.json")
    algorithm_path = os.path.join(project_dir, "algorithm.py")
    inbox_path = os.path.join(project_dir, (params or {}).get("ALLY_INBOX", "orders_inbox.jsonl"))

    # Load templates (simple approach; if Jinja2 exists you can swap in)
    config_template_path = os.path.join(templates_root, "config.live.paper.json")
    algo_template_path = os.path.join(templates_root, "algorithm.py.j2")
    if not os.path.exists(config_template_path):
        raise FileNotFoundError(f"Missing template: {config_template_path}")
    if not os.path.exists(algo_template_path):
        raise FileNotFoundError(f"Missing template: {algo_template_path}")

    # Render config (plain JSON substitution, keep deterministic key order)
    with open(config_template_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Inject paths and symbols; keep placeholders for QC creds (never write secrets)
    cfg["data-folder"] = str(data_dir)
    cfg["result-destination-folder"] = str(results_dir)
    # Parameters flow into algorithm via GetParameter
    cfg.setdefault("parameters", {})
    cfg["parameters"]["ALLY_INBOX"] = (params or {}).get("ALLY_INBOX", "orders_inbox.jsonl")
    cfg["parameters"]["ALLY_SYMBOLS"] = ",".join(norm["symbols"])

    # Stable JSON dump
    cfg_bytes = json.dumps(cfg, indent=2, sort_keys=True).encode("utf-8")
    _write_text(config_path, cfg_bytes.decode("utf-8"))

    # Render algorithm from template (very small, deterministic replacement)
    with open(algo_template_path, "r", encoding="utf-8") as f:
        algo_tpl = f.read()
    algo_src = (
        algo_tpl
        .replace("{{ALLY_INBOX}}", cfg["parameters"]["ALLY_INBOX"])
        .replace("{{ALLY_SYMBOLS}}", cfg["parameters"]["ALLY_SYMBOLS"])
    )
    _write_text(algorithm_path, algo_src)

    # Create empty inbox file
    _write_text(inbox_path, "")

    # Receipts
    utc_ts = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    receipts = {
        "config": record_file_receipt(db_path, "qc.bootstrap", "config.live.paper.json", config_path, params_hash8),
        "algorithm": record_file_receipt(db_path, "qc.bootstrap", "algorithm.py", algorithm_path, params_hash8),
        "inbox": record_file_receipt(db_path, "qc.bootstrap", "orders_inbox.jsonl", inbox_path, params_hash8),
    }

    return BootstrapResult(
        project_dir=project_dir,
        config_path=config_path,
        algorithm_path=algorithm_path,
        inbox_path=inbox_path,
        params_hash=params_hash8,
        receipts=receipts,
        utc_timestamp=utc_ts,
    )