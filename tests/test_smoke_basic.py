"""
Basic smoke tests for CI green-first verification
These tests must always pass to ensure build success
"""
import pytest
import sys
import os
from pathlib import Path


@pytest.mark.smoke
def test_python_version():
    """Verify Python version is compatible"""
    assert sys.version_info >= (3, 8), f"Python {sys.version} is too old"


@pytest.mark.smoke
def test_ally_package_structure():
    """Verify basic package structure exists"""
    ally_dir = Path("ally")
    assert ally_dir.exists(), "ally/ package directory not found"
    assert ally_dir.is_dir(), "ally/ is not a directory"


@pytest.mark.smoke
def test_required_dependencies():
    """Verify core dependencies can be imported"""
    try:
        import pandas as pd
        import numpy as np
        import pydantic
        # Basic functionality check
        assert len(pd.__version__) > 0
        assert len(np.__version__) > 0
        assert hasattr(pydantic, 'BaseModel')
    except ImportError as e:
        pytest.fail(f"Required dependency missing: {e}")


@pytest.mark.smoke
def test_artifacts_directory():
    """Ensure artifacts directory can be created"""
    artifacts_dir = Path("artifacts/ci")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    assert artifacts_dir.exists(), "Could not create artifacts/ci directory"


@pytest.mark.smoke
def test_environment_variables():
    """Verify CI environment is properly configured"""
    ally_live = os.getenv("ALLY_LIVE", "0")
    assert ally_live == "0", f"ALLY_LIVE should be 0 in CI, got {ally_live}"

    tz = os.getenv("TZ", "")
    if tz:  # Only check if TZ is set
        assert tz == "UTC", f"TZ should be UTC in CI, got {tz}"