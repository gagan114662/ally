import pytest

pytestmark = pytest.mark.mreview


def test_placeholder_offline_smoke():
    """Minimal: ensure module imports and REQUIRED_KEYS defined"""
    from ally.tools.reviewer import REQUIRED_KEYS
    assert "M-Reliability" in REQUIRED_KEYS
    assert "M11 (T-Costs)" in REQUIRED_KEYS
    assert "M-Research" in REQUIRED_KEYS
    
    # Verify all groups have proper key lists
    for group, keys in REQUIRED_KEYS.items():
        assert isinstance(keys, list)
        if group != "M11 (T-Costs) optional":
            assert len(keys) > 0  # Most groups should have required keys
    
    # Test that reviewer tool is registered
    from ally.tools import TOOL_REGISTRY
    assert "reviewer.check_pr" in TOOL_REGISTRY


def test_github_client_import():
    """Test that GitHub client can be imported"""
    from ally.providers.github_client import GH
    
    # Test client can be instantiated (without making API calls)
    client = GH("test_owner", "test_repo", token=None)
    assert client.owner == "test_owner"
    assert client.repo == "test_repo"
    assert client.token is None  # No token in test environment


def test_flatten_function():
    """Test the _flatten utility function"""
    from ally.tools.reviewer import _flatten
    
    test_dict = {
        "a": 1,
        "b": {
            "c": 2,
            "d": {
                "e": 3
            }
        },
        "f": "test"
    }
    
    result = _flatten(test_dict)
    assert "a=1" in result
    assert "b.c=2" in result
    assert "b.d.e=3" in result
    assert "f=test" in result