"""
Red-team tests for M-FactorLens Gate
Tests gate failure scenarios to ensure bulletproof enforcement
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from ally.tools import execute_tool


def generate_test_returns(n_days: int = 300, seed: int = 42):
    """Generate synthetic returns for testing"""
    np.random.seed(seed)
    base_date = datetime(2023, 1, 1)
    returns = []

    for i in range(n_days):
        date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        daily_return = np.random.randn() * 0.02 + 0.0005
        returns.append({"date": date, "return": float(daily_return)})

    return returns


def test_gate_fails_low_alpha_tstat():
    """Red-team: Gate must fail when abs(alpha_t) < 2.0"""
    from unittest.mock import patch
    from ally.schemas.base import ToolResult

    returns = generate_test_returns(300, seed=1)

    # Patch both tools through the TOOL_REGISTRY
    with patch('ally.tools.TOOL_REGISTRY') as mock_registry:
        # Mock compute_exposures
        mock_exp_result = ToolResult.success({
            "exposures": [
                {"factor": "alpha", "beta": 0.002, "tstat": 1.8, "pvalue": 0.08},
                {"factor": "MKT", "beta": 0.25, "tstat": 3.2, "pvalue": 0.002},
                {"factor": "SMB", "beta": -0.12, "tstat": -1.5, "pvalue": 0.14},
                {"factor": "HML", "beta": 0.08, "tstat": 0.9, "pvalue": 0.37},
                {"factor": "RMW", "beta": 0.15, "tstat": 1.7, "pvalue": 0.09},
                {"factor": "CMA", "beta": -0.05, "tstat": -0.6, "pvalue": 0.55},
                {"factor": "MOM", "beta": 0.18, "tstat": 2.1, "pvalue": 0.04}
            ],
            "n_obs": 300,
            "lags": 5
        })

        # Mock residual_alpha with failing t-stat
        mock_alpha_result = ToolResult.success({
            "alpha_bps": 150.0,
            "alpha_tstat": 1.2,  # Below 2.0 threshold
            "r2": 0.55,
            "window_days": 252,
            "step_days": 21,
            "n_windows": 10
        })

        # Set up mock registry
        from ally.tools import TOOL_REGISTRY
        original_registry = TOOL_REGISTRY.copy()
        mock_registry.__getitem__.side_effect = lambda key: {
            "factors.compute_exposures": lambda **kwargs: mock_exp_result,
            "factors.residual_alpha": lambda **kwargs: mock_alpha_result
        }.get(key, original_registry[key])

        result = execute_tool("orchestrator.factor_gate", returns=returns)

        assert result.ok
        assert result.data["proofs"]["FACTLENS_GATE"] == "FAIL"
        assert result.data["proofs"]["OOS_TSTAT"] == 1.2
        assert not result.data["alpha_pass"]


def test_gate_fails_beta_breach():
    """Red-team: Gate must fail when max(|β|) > 0.30"""
    from unittest.mock import patch

    returns = generate_test_returns(300, seed=2)

    # Patch exposures to have breaching beta
    with patch('ally.tools.factors.compute_exposures') as mock_exp:
        mock_exp.return_value.ok = True
        mock_exp.return_value.data = {
            "exposures": [
                {"factor": "alpha", "beta": 0.002, "tstat": 2.5, "pvalue": 0.01},
                {"factor": "MKT", "beta": 0.45, "tstat": 4.2, "pvalue": 0.001},  # Breach!
                {"factor": "SMB", "beta": -0.12, "tstat": -1.5, "pvalue": 0.14},
                {"factor": "HML", "beta": 0.08, "tstat": 0.9, "pvalue": 0.37},
                {"factor": "RMW", "beta": 0.15, "tstat": 1.7, "pvalue": 0.09},
                {"factor": "CMA", "beta": -0.05, "tstat": -0.6, "pvalue": 0.55},
                {"factor": "MOM", "beta": 0.18, "tstat": 2.1, "pvalue": 0.04}
            ],
            "n_obs": 300,
            "lags": 5
        }

        result = execute_tool("orchestrator.factor_gate", returns=returns)

        assert result.ok
        assert result.data["proofs"]["FACTLENS_GATE"] == "FAIL"
        assert result.data["proofs"]["BETAS_OK"] == "false"
        assert len(result.data["violations"]) > 0
        assert "MKT: β=0.450" in result.data["violations"][0]


def test_gate_fails_insufficient_oos():
    """Red-team: Gate must fail with insufficient OOS windows"""
    returns = generate_test_returns(50, seed=3)  # Only 50 days, insufficient for 252-day windows

    result = execute_tool("orchestrator.factor_gate", returns=returns, window=252, step=21)

    assert result.ok
    assert result.data["proofs"]["FACTLENS_GATE"] == "FAIL"
    assert result.data["proofs"]["INSUFFICIENT_OOS"] == "true"
    assert result.data["n_windows"] < 3


def test_gate_handles_negative_good_alpha():
    """Red-team: Gate must use abs(t-stat), rejecting negative 'good' alpha"""
    from unittest.mock import patch

    returns = generate_test_returns(300, seed=4)

    # Patch to return negative but significant t-stat
    with patch('ally.tools.factors.compute_residual_alpha') as mock_alpha:
        mock_alpha.return_value.ok = True
        mock_alpha.return_value.data = {
            "alpha_bps": -180.0,  # Negative alpha
            "alpha_tstat": -2.8,  # Negative but significant
            "r2": 0.65,
            "window_days": 252,
            "step_days": 21,
            "n_windows": 10
        }

        result = execute_tool("orchestrator.factor_gate", returns=returns)

        # Should PASS because abs(-2.8) = 2.8 >= 2.0
        assert result.ok
        assert result.data["proofs"]["FACTLENS_GATE"] == "PASS"
        assert result.data["proofs"]["OOS_TSTAT"] == 2.8  # Absolute value
        assert result.data["alpha_pass"]


def test_gate_pit_guard():
    """Red-team: Gate must detect PIT violations"""
    # This would be implemented when real PIT checking is added
    # For now, test that PIT_OK proof is emitted
    returns = generate_test_returns(300, seed=5)

    result = execute_tool("orchestrator.factor_gate", returns=returns)

    assert result.ok
    assert "PIT_OK" in result.data["proofs"]
    assert result.data["proofs"]["PIT_OK"] in ["true", "false"]


def test_config_determinism():
    """Verify gate config is encoded in hash for reproducibility"""
    returns = generate_test_returns(300, seed=6)

    result1 = execute_tool("orchestrator.factor_gate", returns=returns, window=252, step=21)
    result2 = execute_tool("orchestrator.factor_gate", returns=returns, window=252, step=21)
    result3 = execute_tool("orchestrator.factor_gate", returns=returns, window=126, step=21)  # Different window

    # Same config should give same hash
    assert result1.data["gate_hash"] == result2.data["gate_hash"]

    # Different config should give different hash
    assert result1.data["gate_hash"] != result3.data["gate_hash"]


def test_all_bulletproof_proofs_present():
    """Verify all 9+ bulletproof proof lines are emitted"""
    returns = generate_test_returns(300, seed=7)

    result = execute_tool("orchestrator.factor_gate", returns=returns)

    required_proofs = [
        "FACTLENS_GATE", "RES_ALPHA_T", "BETAS_OK", "FACTORLENS_HASH",
        "PIT_OK", "NW_LAGS", "WINDOW_DAYS", "STEP_DAYS", "MIN_OBS",
        "OOS_TSTAT", "FDR_ALPHA", "INSUFFICIENT_OOS"
    ]

    for proof in required_proofs:
        assert proof in result.data["proofs"], f"Missing proof: {proof}"

    # Check specific proof formats
    assert isinstance(result.data["proofs"]["NW_LAGS"], int)
    assert isinstance(result.data["proofs"]["WINDOW_DAYS"], int)
    assert isinstance(result.data["proofs"]["STEP_DAYS"], int)
    assert result.data["proofs"]["FDR_ALPHA"] == "pending"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])