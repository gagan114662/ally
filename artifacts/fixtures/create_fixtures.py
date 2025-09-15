#!/usr/bin/env python3
"""Create minimal offline fixtures for ops tools testing."""
import json
import pickle
import hashlib
from datetime import datetime, timedelta


def create_features_small():
    """Create tiny parquet-like data as JSON (since no pandas)."""
    # Simulate feature data for 30 days with 3 features
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]

    data = {
        'date': dates,
        'feature1': [0.5 + 0.1 * (i % 5) for i in range(30)],  # Stable feature
        'feature2': [1.0 + 0.2 * (i % 3) for i in range(30)],  # Slightly varying
        'feature3': [2.0 + 0.05 * i for i in range(30)]        # Trending feature
    }

    with open('artifacts/fixtures/features_small.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("Created features_small.json")


def create_determinism_fixture():
    """Create determinism test fixture."""
    # Simple data that can be processed deterministically
    data = {
        'matrix': [
            [1.0, 0.3, 0.1],
            [0.3, 1.0, 0.2],
            [0.1, 0.2, 1.0]
        ],
        'returns': [0.01, 0.02, -0.01, 0.015, -0.005] * 10,
        'seed': 42
    }

    with open('artifacts/fixtures/determinism.pkl', 'wb') as f:
        pickle.dump(data, f)

    print("Created determinism.pkl")


def create_strategy_fixture():
    """Create strategy comparison data."""
    # Synthetic live vs reference returns
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(21)]

    data = {
        'strategy': 'TEST_STRAT_XS_MOMENTUM',
        'dates': dates,
        'live_returns': [0.001 * (1 + 0.1 * (i % 7)) for i in range(21)],
        'ref_returns': [0.001 * (1 + 0.05 * (i % 7)) for i in range(21)],
        'live_sharpe': 0.82,
        'ref_sharpe': 0.85
    }

    with open('artifacts/fixtures/strategy_test.json', 'w') as f:
        json.dump(data, f, indent=2)

    print("Created strategy_test.json")


if __name__ == '__main__':
    create_features_small()
    create_determinism_fixture()
    create_strategy_fixture()
    print("All fixtures created successfully")