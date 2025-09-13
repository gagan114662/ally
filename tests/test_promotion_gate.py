import pytest
from ally.tools import TOOL_REGISTRY

pytestmark = pytest.mark.mpromotion

def test_holdout_gate_and_bundle(tmp_path):
    res = TOOL_REGISTRY["promotion.holdout_gate"](
        strategy_id="demo_strategy",
        selection_sha1="deadbeef"*5,
        symbols=["SPY","QQQ","TLT","GLD"],
        seed=1337,
    )
    assert "decision" in res.data
    # deterministic shape
    assert res.data["holdout_days"] == 21

    bun = TOOL_REGISTRY["promotion.bundle"](
        strategy_id="demo_strategy", selection_sha1="deadbeef"*5
    )
    assert bun.ok
    assert "bundle_sha1" in bun.data