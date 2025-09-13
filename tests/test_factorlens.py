import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from ally.utils.factorlens import exposures, rolling_alpha, det_hash, pit_align
from ally.tools.factors import load_ff, compute_exposures, compute_residual_alpha
from ally.schemas.factors import FactorLensSummary
import jsonschema
import json

pytestmark = pytest.mark.mfl

@pytest.fixture
def synthetic_data():
    # Generate synthetic data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')

    # Create synthetic factors
    factors = pd.DataFrame({
        'MKT': np.random.normal(0.0005, 0.015, 300),
        'SMB': np.random.normal(0.0, 0.01, 300),
        'HML': np.random.normal(0.0, 0.01, 300),
        'RMW': np.random.normal(0.0, 0.008, 300),
        'CMA': np.random.normal(0.0, 0.008, 300),
        'MOM': np.random.normal(0.0, 0.012, 300)
    }, index=dates)

    # Create synthetic returns with known factor loadings
    true_alpha = 0.0002  # 20 bps annualized
    true_betas = {'MKT': 0.8, 'SMB': 0.3, 'HML': -0.2, 'RMW': 0.1, 'CMA': 0.05, 'MOM': 0.15}

    returns = pd.Series(index=dates, dtype=float)
    for i in range(300):
        factor_contrib = sum(true_betas[f] * factors.iloc[i][f] for f in factors.columns)
        noise = np.random.normal(0, 0.005)
        returns.iloc[i] = true_alpha + factor_contrib + noise

    return factors, pd.DataFrame({'ret': returns}), true_betas, true_alpha

def test_exposures_synthetic_recovery(synthetic_data):
    """Test that we can recover known factor exposures from synthetic data"""
    factors, returns, true_betas, true_alpha = synthetic_data

    res = exposures(returns, factors, lags=5)

    # Check alpha (intercept) recovery
    recovered_alpha = res["beta"][0]
    assert abs(recovered_alpha - true_alpha) < 0.0005, f"Alpha recovery failed: {recovered_alpha} vs {true_alpha}"

    # Check beta recovery
    factor_names = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    for i, factor in enumerate(factor_names):
        recovered_beta = res["beta"][i+1]
        true_beta = true_betas[factor]
        assert abs(recovered_beta - true_beta) < 0.1, f"Beta recovery failed for {factor}: {recovered_beta} vs {true_beta}"

    # Check that R-squared is reasonable
    assert 0.3 < res["r2"] < 0.95, f"R-squared unreasonable: {res['r2']}"

def test_rolling_alpha_detects_alpha(synthetic_data):
    """Test that rolling alpha calculation detects positive alpha"""
    factors, returns, _, true_alpha = synthetic_data

    # Add extra alpha to make detection easier
    enhanced_returns = returns.copy()
    enhanced_returns['ret'] += 0.0008  # Add 80 bps annual alpha
    expected_alpha_bps = (true_alpha + 0.0008) * 252 * 10000  # Convert to annualized bps

    result = rolling_alpha(enhanced_returns, factors, window=100, step=20, lags=5)

    # Check alpha detection
    assert result["alpha_bps"] > 150, f"Alpha too low: {result['alpha_bps']} bps"
    assert result["alpha_t"] > 2.0, f"Alpha t-stat too low: {result['alpha_t']}"
    assert 0.2 < result["r2"] < 0.95, f"R-squared unreasonable: {result['r2']}"

def test_alignment_no_lookahead():
    """Test that alignment prevents look-ahead bias"""
    dates1 = pd.date_range('2023-01-01', periods=10, freq='D')
    dates2 = pd.date_range('2023-01-02', periods=10, freq='D')  # Shifted by 1 day

    returns = pd.DataFrame({'ret': np.random.randn(10)}, index=dates1)
    factors = pd.DataFrame({'MKT': np.random.randn(10)}, index=dates2)

    r_aligned, f_aligned = pit_align(returns, factors)

    # Should only have overlapping dates (9 days)
    assert len(r_aligned) == 9
    assert len(f_aligned) == 9

    # Check that dates match exactly
    assert all(r_aligned.index == f_aligned.index)

def test_determinism_hash():
    """Test that hash function produces deterministic results"""
    test_obj = {"alpha_bps": 23.45, "r2": 0.65, "factors": ["MKT", "SMB"]}

    hash1 = det_hash(test_obj)
    hash2 = det_hash(test_obj)

    assert hash1 == hash2
    assert len(hash1) == 40  # SHA1 hex digest length

def test_schema_valid():
    """Test that FactorLensSummary validates against JSON schema"""
    # Load the JSON schema
    with open('ally/verify/jsonschema/factorlens_summary.schema.json') as f:
        schema = json.load(f)

    # Create a valid summary object
    from ally.schemas.factors import FactorSetMeta, ExposureRow, ExposuresOut, ResidualAlphaOut

    summary = FactorLensSummary(
        meta=FactorSetMeta(name="FF5+Mom", frequency="D", columns=["MKT","SMB","HML","RMW","CMA","MOM"]),
        exposures=ExposuresOut(
            r2=0.65,
            exposures=[
                ExposureRow(factor="MKT", beta=0.8, tstat=3.2),
                ExposureRow(factor="SMB", beta=0.3, tstat=1.8)
            ],
            method="OLS-NeweyWest",
            lags=5
        ),
        residual=ResidualAlphaOut(
            alpha_bps=25.5,
            alpha_tstat=2.1,
            r2=0.65,
            window_days=252,
            residual_series_path=""
        ),
        det_hash="abc123",
        ts_utc=datetime.now(timezone.utc)
    )

    # Convert to dict and validate
    summary_dict = summary.model_dump()
    summary_dict['ts_utc'] = summary_dict['ts_utc'].isoformat()

    # Should not raise exception
    jsonschema.validate(summary_dict, schema)

def test_tools_integration():
    """Test that the registered tools work correctly"""
    # Test load_ff
    result = load_ff()
    assert result.ok
    assert "meta" in result.data
    assert "frame" in result.data

    # Create some test returns data
    test_returns = [
        {"date": "2023-01-03", "ret": 0.0025},
        {"date": "2023-01-04", "ret": -0.0015},
        {"date": "2023-01-05", "ret": 0.0032},
        {"date": "2023-01-06", "ret": -0.0008},
        {"date": "2023-01-09", "ret": 0.0018}
    ]

    # Test exposures
    exp_result = compute_exposures(test_returns, lags=3)
    assert exp_result.ok
    assert "r2" in exp_result.data
    assert "exposures" in exp_result.data

    # Test residual alpha (need more data for meaningful results)
    longer_returns = test_returns * 60  # Repeat to get more data points
    for i, ret in enumerate(longer_returns):
        ret["date"] = pd.to_datetime("2023-01-01") + pd.Timedelta(days=i)
        ret["date"] = ret["date"].strftime("%Y-%m-%d")

    alpha_result = compute_residual_alpha(longer_returns, window=100, step=20, lags=3)
    assert alpha_result.ok
    assert "residual" in alpha_result.data
    assert "det_hash" in alpha_result.data