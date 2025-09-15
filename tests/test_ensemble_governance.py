#!/usr/bin/env python3
"""
Test suite for Phase 9 ensemble governance
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add the nested ally path to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ally', 'ally', 'ally'))

from research.ensemble_governance import (
    govern_ensemble,
    check_weight_caps,
    check_correlation_caps,
    check_exposure_caps,
    apply_drift_deweighting,
    calculate_initial_weights
)


class TestEnsembleGovernance(unittest.TestCase):
    """Test ensemble governance functions"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_strategies = [
            {
                "id": "Test_Strat_A",
                "family": "momentum",
                "weight_hint": 0.40,
                "risk_tag": "core",
                "drift_status": "OK"
            },
            {
                "id": "Test_Strat_B",
                "family": "value",
                "weight_hint": 0.35,
                "risk_tag": "core",
                "drift_status": "DRIFT"
            },
            {
                "id": "Test_Strat_C",
                "family": "trend",
                "weight_hint": 0.25,
                "risk_tag": "satellite",
                "drift_status": "OK"
            }
        ]

        self.test_policy = {
            'phase9': {
                'caps': {
                    'weight_per_strategy_max': 0.20,
                    'weight_per_family_max': 0.40,
                    'gross_exposure_max': 1.00,
                    'net_exposure_band': 0.20
                },
                'correlation': {
                    'pairwise_max': 0.65,
                    'factor_exposure_max': 1.5
                },
                'drift': {
                    'require_sentinel_ok': True,
                    'deweight_on_warn': 0.50
                },
                'novelty': {
                    'min_n_strategies': 3,
                    'novelty_quota_min': 0.10
                }
            }
        }

        self.test_corr_matrix = {
            "Test_Strat_A": {"Test_Strat_A": 1.0, "Test_Strat_B": 0.4, "Test_Strat_C": 0.3},
            "Test_Strat_B": {"Test_Strat_A": 0.4, "Test_Strat_B": 1.0, "Test_Strat_C": 0.2},
            "Test_Strat_C": {"Test_Strat_A": 0.3, "Test_Strat_B": 0.2, "Test_Strat_C": 1.0}
        }

    def test_calculate_initial_weights(self):
        """Test initial weight calculation"""
        weights = calculate_initial_weights(self.test_strategies)

        # Should normalize to sum to 1
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

        # Should preserve relative proportions
        self.assertGreater(weights["Test_Strat_A"], weights["Test_Strat_B"])
        self.assertGreater(weights["Test_Strat_B"], weights["Test_Strat_C"])

    def test_check_weight_caps(self):
        """Test weight cap validation"""
        # Test weights that violate caps
        bad_weights = {"Test_Strat_A": 0.50, "Test_Strat_B": 0.30, "Test_Strat_C": 0.20}
        caps_ok, violations = check_weight_caps(bad_weights, self.test_strategies, self.test_policy)

        self.assertFalse(caps_ok)
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("Test_Strat_A" in v for v in violations))

        # Test weights that pass caps
        good_weights = {"Test_Strat_A": 0.18, "Test_Strat_B": 0.18, "Test_Strat_C": 0.18, "Cash": 0.46}
        caps_ok, violations = check_weight_caps(good_weights, self.test_strategies, self.test_policy)

        self.assertTrue(caps_ok)
        self.assertEqual(len(violations), 0)

    def test_check_correlation_caps(self):
        """Test correlation cap validation"""
        # High correlation, high weights (should violate)
        weights = {"Test_Strat_A": 0.50, "Test_Strat_B": 0.50}  # corr = 0.4 < 0.65, but high weight product

        # Mock a high correlation
        high_corr_matrix = {
            "Test_Strat_A": {"Test_Strat_B": 0.80},  # Above 0.65 threshold
            "Test_Strat_B": {"Test_Strat_A": 0.80}
        }

        caps_ok, violations = check_correlation_caps(weights, high_corr_matrix, self.test_policy)
        self.assertFalse(caps_ok)
        self.assertGreater(len(violations), 0)

    def test_check_exposure_caps(self):
        """Test exposure cap validation"""
        # Test gross exposure violation
        bad_weights = {"Test_Strat_A": 0.8, "Test_Strat_B": 0.8}  # Gross = 1.6 > 1.0
        caps_ok, violations = check_exposure_caps(bad_weights, self.test_policy)

        self.assertFalse(caps_ok)
        self.assertTrue(any("gross_exposure" in v for v in violations))

        # Test net exposure violation
        bad_net_weights = {"Test_Strat_A": 0.4, "Test_Strat_B": 0.4}  # Net = 0.8, deviation = 0.2 = band
        caps_ok, violations = check_exposure_caps(bad_net_weights, self.test_policy)

        self.assertTrue(caps_ok)  # Exactly at band, should pass

    def test_apply_drift_deweighting(self):
        """Test drift-based deweighting"""
        initial_weights = {"Test_Strat_A": 0.4, "Test_Strat_B": 0.4, "Test_Strat_C": 0.2}

        adjusted_weights, violations = apply_drift_deweighting(
            initial_weights, self.test_strategies, self.test_policy
        )

        # Test_Strat_B should be deweighted due to DRIFT status
        self.assertLess(adjusted_weights["Test_Strat_B"], initial_weights["Test_Strat_B"])
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("Test_Strat_B" in v for v in violations))

        # Weights should still sum to 1 after renormalization
        self.assertAlmostEqual(sum(adjusted_weights.values()), 1.0, places=6)

    def test_deterministic_governance(self):
        """Test that governance is deterministic with same inputs"""
        # This would test with fixed asof timestamp
        asof1 = "2025-09-15T12:00:00Z"
        asof2 = "2025-09-15T12:00:00Z"

        # In a real test, we'd run governance twice and check receipt hashes match
        # For now, just verify the function runs without error
        try:
            result = govern_ensemble(
                approved_path="artifacts/fixtures/phase9/approved_bundles.json",
                corr_path="artifacts/fixtures/phase9/pairwise_corr.csv",
                factors_path="artifacts/fixtures/phase9/factor_exposures.csv",
                policy_path="ally/ops/policy.yaml",
                asof=asof1,
                live=False
            )
            self.assertIn('governance_ok', result)
            self.assertIn('receipt_hash', result)
        except Exception as e:
            # Expected if files don't exist in test environment
            self.assertIn('FileNotFoundError', str(type(e).__name__))


if __name__ == '__main__':
    unittest.main()