import pytest
pytestmark = pytest.mark.mautopilot

def test_autopilot_runs():
    from ally.tools.autopilot import run
    r = run("implement add")
    assert r.ok
    assert r.data["rounds"] in (0,1,2)
    assert len(r.data["det_hash"])==40