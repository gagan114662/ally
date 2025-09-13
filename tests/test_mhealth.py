import pytest
from ally.tools import TOOL_REGISTRY

pytestmark = pytest.mark.mhealth

def test_heartbeat_rotates():
    res = TOOL_REGISTRY["health.heartbeat"](run_id="TEST_HEARTBEAT")
    assert res.ok and res.data["heartbeat_rotating"] is True

def test_killswitch_drill_ok():
    res = TOOL_REGISTRY["health.killswitch_drill"](threshold_bps=500.0)
    assert res.ok and res.data["kill_switch_drill"] == "ok"
    assert 0 <= res.data["kill_switch_ttr_sec"] < 1.0  # bounded & quick