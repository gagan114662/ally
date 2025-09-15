"""
Test configuration: auto-skip live tests in CI environments
"""
import os
import pytest

# Auto-skip live tests when ALLY_LIVE=0 (default in CI)
ALLY_LIVE = os.getenv("ALLY_LIVE", "0")

if ALLY_LIVE != "1":
    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(config, items):
        skip_live = pytest.mark.skip(reason="ALLY_LIVE=0; skipping @live tests in CI")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)