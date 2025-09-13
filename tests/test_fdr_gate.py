"""
Tests for M-FDR Gate with Benjamini-Hochberg correction
"""

import json
import pytest
from ally.tools import TOOL_REGISTRY

pytestmark = pytest.mark.mfdr

def test_fdr_bh_promotions_default():
    """Test FDR BH promotion logic with deterministic fixtures"""
    cands = [
      {"id":"A","t_oos":3.2,"oos_obs":252,"alpha_oos":0.45},
      {"id":"B","t_oos":2.9,"oos_obs":252,"alpha_oos":0.30},
      {"id":"C","t_oos":2.5,"oos_obs":252,"alpha_oos":0.20},
      {"id":"D","t_oos":2.1,"oos_obs":252,"alpha_oos":0.05},
      {"id":"E","t_oos":1.8,"oos_obs":252,"alpha_oos":0.02},
      {"id":"F","t_oos":1.5,"oos_obs":252,"alpha_oos":0.01},
      {"id":"G","t_oos":1.2,"oos_obs":252,"alpha_oos":0.01},
      {"id":"H","t_oos":0.8,"oos_obs":252,"alpha_oos":0.00},
      {"id":"I","t_oos":0.0,"oos_obs":252,"alpha_oos":0.00},
      {"id":"J","t_oos":-0.5,"oos_obs":252,"alpha_oos":-0.01},
      {"id":"K","t_oos":-1.2,"oos_obs":252,"alpha_oos":-0.02},
      {"id":"L","t_oos":-2.2,"oos_obs":252,"alpha_oos":-0.03}
    ]

    res = TOOL_REGISTRY["fdr.evaluate"](candidates=cands, alpha=0.05, require_positive_alpha=True, min_oos_obs=60)

    assert res.ok
    data = res.data

    # Expected: 7 tested (positive alpha only, zeros filtered), 3 promoted (A,B,C)
    assert data["n_tested"] == 7   # negatives and zeros removed by positive-alpha policy
    assert data["n_promoted"] == 3
    assert set(data["promoted_ids"]) == {"A","B","C"}
    assert data["method"] == "BH"
    assert data["alpha"] == 0.05
    assert data["pos_alpha_enforced"] is True
    assert abs(data["mean_t_promoted"] - 2.867) < 0.01  # A=3.2, B=2.9, C=2.5 -> mean=2.867


def test_fdr_no_positive_alpha_filter():
    """Test FDR without positive alpha filtering"""
    cands = [
      {"id":"A","t_oos":3.2,"oos_obs":252,"alpha_oos":0.45},
      {"id":"B","t_oos":-2.8,"oos_obs":252,"alpha_oos":-0.30},  # Strong negative
      {"id":"C","t_oos":1.5,"oos_obs":252,"alpha_oos":0.10}
    ]

    res = TOOL_REGISTRY["fdr.evaluate"](candidates=cands, alpha=0.05, require_positive_alpha=False, min_oos_obs=60)

    assert res.ok
    data = res.data

    assert data["n_tested"] == 3   # All candidates included
    assert data["pos_alpha_enforced"] is False
    # Should promote A and B (both have strong t-stats)
    assert data["n_promoted"] >= 2


def test_fdr_insufficient_oos_obs():
    """Test FDR filtering by minimum OOS observations"""
    cands = [
      {"id":"A","t_oos":3.2,"oos_obs":50,"alpha_oos":0.45},   # Below min_oos_obs
      {"id":"B","t_oos":2.9,"oos_obs":252,"alpha_oos":0.30},  # Above min_oos_obs
      {"id":"C","t_oos":2.5,"oos_obs":30,"alpha_oos":0.20}   # Below min_oos_obs
    ]

    res = TOOL_REGISTRY["fdr.evaluate"](candidates=cands, alpha=0.05, require_positive_alpha=True, min_oos_obs=60)

    assert res.ok
    data = res.data

    assert data["n_tested"] == 1   # Only B has sufficient OOS obs
    assert data["n_promoted"] == 1
    assert data["promoted_ids"] == ["B"]


def test_fdr_empty_candidate_set():
    """Test FDR with empty candidate set after filtering"""
    cands = [
      {"id":"A","t_oos":1.0,"oos_obs":252,"alpha_oos":-0.10},  # Negative alpha
      {"id":"B","t_oos":0.5,"oos_obs":252,"alpha_oos":-0.05}   # Negative alpha
    ]

    res = TOOL_REGISTRY["fdr.evaluate"](candidates=cands, alpha=0.05, require_positive_alpha=True, min_oos_obs=60)

    assert res.ok
    data = res.data

    assert data["n_tested"] == 0
    assert data["n_promoted"] == 0
    assert data["promoted_ids"] == []
    assert data["det_hash"] is not None  # Still generates deterministic hash


def test_fdr_deterministic_hash():
    """Test FDR produces deterministic hashes for same inputs"""
    cands = [
      {"id":"A","t_oos":3.2,"oos_obs":252,"alpha_oos":0.45},
      {"id":"B","t_oos":2.9,"oos_obs":252,"alpha_oos":0.30}
    ]

    res1 = TOOL_REGISTRY["fdr.evaluate"](candidates=cands, alpha=0.05, require_positive_alpha=True, min_oos_obs=60)
    res2 = TOOL_REGISTRY["fdr.evaluate"](candidates=cands, alpha=0.05, require_positive_alpha=True, min_oos_obs=60)
    res3 = TOOL_REGISTRY["fdr.evaluate"](candidates=cands, alpha=0.10, require_positive_alpha=True, min_oos_obs=60)

    assert res1.ok and res2.ok and res3.ok

    # Same params should give same hash
    assert res1.data["det_hash"] == res2.data["det_hash"]

    # Different alpha should give different hash
    assert res1.data["det_hash"] != res3.data["det_hash"]


def test_fdr_q_values_computed():
    """Test that q-values are properly computed and stored"""
    cands = [
      {"id":"A","t_oos":3.2,"oos_obs":252,"alpha_oos":0.45},
      {"id":"B","t_oos":2.9,"oos_obs":252,"alpha_oos":0.30},
      {"id":"C","t_oos":1.5,"oos_obs":252,"alpha_oos":0.10}
    ]

    res = TOOL_REGISTRY["fdr.evaluate"](candidates=cands, alpha=0.05, require_positive_alpha=True, min_oos_obs=60)

    assert res.ok
    data = res.data

    # Should have q-values for all tested candidates
    assert len(data["q_values"]) == data["n_tested"]
    assert "A" in data["q_values"]
    assert "B" in data["q_values"]
    assert "C" in data["q_values"]

    # Q-values should be between 0 and 1
    for q_val in data["q_values"].values():
        assert 0 <= q_val <= 1


def test_mock_candidate_generation():
    """Test mock candidate generator for consistency"""
    res = TOOL_REGISTRY["fdr.mock_candidates"](n_candidates=12, seed=42)

    assert res.ok
    data = res.data

    candidates = data["candidates"]
    assert len(candidates) == 12

    # Check structure of generated candidates
    for cand in candidates:
        assert "id" in cand
        assert "t_oos" in cand
        assert "oos_obs" in cand
        assert "alpha_oos" in cand
        assert cand["oos_obs"] == 252  # Should be full year

    # Should be deterministic with same seed
    res2 = TOOL_REGISTRY["fdr.mock_candidates"](n_candidates=12, seed=42)
    assert res2.data["candidates"] == candidates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])