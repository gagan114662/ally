from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json, hashlib
from ..schemas.base import ToolResult, Meta
from ..schemas.fdr import FDRConfig, Candidate, FDRResult
from ..utils.stats import p_from_t_two_sided, benjamini_hochberg
from ..utils.audit import AuditLogger
from . import register

def _sha1(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def _evaluate_group(cands: List[Candidate], cfg: FDRConfig) -> FDRResult:
    # filter by OOS obs
    cands = [c for c in cands if c.oos_obs >= cfg.min_oos_obs]
    if cfg.require_positive_alpha:
        cands = [c for c in cands if c.alpha_oos > 0]

    if not cands:
        det = _sha1({"empty": True, "cfg": cfg.model_dump()})
        return FDRResult(
            n_tested=0, n_promoted=0, promoted_ids=[], q_values={},
            method=cfg.method, alpha=cfg.alpha, groups=[],
            mean_t_promoted=0.0, pos_alpha_enforced=cfg.require_positive_alpha,
            det_hash=det
        )

    ids = [c.id for c in cands]
    tvals = [c.t_oos for c in cands]
    pvals = [p_from_t_two_sided(t) for t in tvals]
    qvals = benjamini_hochberg(pvals)

    promoted = [ids[i] for i,q in enumerate(qvals) if q <= cfg.alpha]
    mean_t = sum(t for i,t in enumerate(tvals) if qvals[i] <= cfg.alpha) / max(1, len(promoted))

    q_map = {ids[i]: q for i,q in enumerate(qvals)}
    det = _sha1({"ids": promoted, "alpha": cfg.alpha, "method": cfg.method})

    return FDRResult(
        n_tested=len(ids),
        n_promoted=len(promoted),
        promoted_ids=promoted,
        q_values=q_map,
        method=cfg.method,
        alpha=cfg.alpha,
        groups=["default"],
        mean_t_promoted=mean_t,
        pos_alpha_enforced=cfg.require_positive_alpha,
        det_hash=det
    )

@register("fdr.evaluate")
def fdr_evaluate(candidates: List[Dict[str, Any]], alpha: float = 0.05,
                 require_positive_alpha: bool = True, min_oos_obs: int = 60) -> ToolResult:
    """
    Evaluate candidates using False Discovery Rate (BH) correction

    Args:
        candidates: List of candidate dicts with id, t_oos, oos_obs, alpha_oos
        alpha: FDR control level (default 0.05)
        require_positive_alpha: Filter to positive OOS alpha only
        min_oos_obs: Minimum OOS observations required

    Returns:
        ToolResult with FDR analysis and promoted candidates
    """
    try:
        cfg = FDRConfig(alpha=alpha, require_positive_alpha=require_positive_alpha, min_oos_obs=min_oos_obs)
        cands = [Candidate(**c) for c in candidates]
        res = _evaluate_group(cands, cfg)

        # Create proofs for CI verification
        proofs = {
            "FDR_ALPHA": cfg.alpha,
            "FDR_METHOD": res.method,
            "N_TESTED": res.n_tested,
            "N_PROMOTED": res.n_promoted,
            "MEAN_T_OOS": round(res.mean_t_promoted, 3),
            "POS_ALPHA_ENFORCED": res.pos_alpha_enforced,
            "FDR_HASH": res.det_hash
        }

        # Add proofs to result data for proof script access
        result_data = res.model_dump()
        result_data["proofs"] = proofs

        return ToolResult.success(result_data)

    except Exception as e:
        return ToolResult.error([f"FDR evaluation error: {str(e)}"])


@register("fdr.mock_candidates")
def generate_mock_candidates(n_candidates: int = 12, seed: int = 42) -> ToolResult:
    """
    Generate mock candidates for testing FDR gate

    Args:
        n_candidates: Number of candidates to generate
        seed: Random seed for reproducibility

    Returns:
        ToolResult with mock candidate list
    """
    import numpy as np
    np.random.seed(seed)

    # Generate realistic t-stats and alpha values
    candidates = []
    for i in range(n_candidates):
        # Create some strong, medium, and weak candidates
        if i < 3:  # Strong candidates
            t_oos = np.random.uniform(2.5, 3.5)
            alpha_oos = np.random.uniform(0.2, 0.5)
        elif i < 6:  # Medium candidates
            t_oos = np.random.uniform(1.5, 2.5)
            alpha_oos = np.random.uniform(0.01, 0.2)
        else:  # Weak/negative candidates
            t_oos = np.random.uniform(-2.5, 1.5)
            alpha_oos = np.random.uniform(-0.05, 0.02)

        candidates.append({
            "id": chr(65 + i),  # A, B, C, ...
            "t_oos": round(float(t_oos), 1),
            "oos_obs": 252,  # Full year of OOS data
            "alpha_oos": round(float(alpha_oos), 3),
            "meta": {"strategy_type": "test"}
        })

    return ToolResult.success({"candidates": candidates})