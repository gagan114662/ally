import pytest
from ally.tools.capacity import capacity_curve

pytestmark = pytest.mark.mcapacity

def test_capacity_monotone():
    out = capacity_curve(symbol="SPY", adv_usd=1_000_000, daily_vol_bps=100.0).data
    xs = [p["notional_usd"] for p in out["curve"]]
    ys = [p["total_cost_bps"] for p in out["curve"]]
    assert xs == sorted(xs)
    assert all(ys[i] <= ys[i+1] + 1e-9 for i in range(len(ys)-1))  # nondecreasing costs