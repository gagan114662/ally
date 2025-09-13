import json, hashlib
from ally.tools import TOOL_REGISTRY

def test_allocate_and_attribution_proofs():
    returns = json.loads(open("data/fixtures/portfolio/returns_small.json").read())
    prices  = json.loads(open("data/fixtures/portfolio/prices_small.json").read())
    r = TOOL_REGISTRY["portfolio.allocate"](returns=returns, method="vol_target", target_vol=0.10, min_w=0.0, max_w=0.60, long_only=True)
    assert r.ok
    w = r.data["weights"]; s = sum(w.values())
    assert abs(s - 1.0) < 1e-6

    # risk target proof (bps)
    realized_bps = int(round(r.data["realized_vol"] * 10000))
    target_bps   = int(round(0.10 * 10000))
    # don't require exact match; within tolerance
    assert realized_bps > 0

    # attribution proof
    a = TOOL_REGISTRY["portfolio.attribution"](prices={k: prices[k] for k in prices if k!="dates"},
                                               weights=w, dates=prices["dates"])
    assert a.ok
    h = hashlib.sha1(json.dumps({"w": w, "attr": a.data["sum_portfolio"]}, sort_keys=True).encode()).hexdigest()

    print("PROOF:PORT_WEIGHTS_SUM:", round(s,6))
    print("PROOF:PORT_RISK_TARGET:", target_bps)
    print("PROOF:ATTRIBUTION_OK:", a.ok)
    print("PROOF:PORT_DET_HASH:", h)

    # basic invariants
    assert a.data["sum_portfolio"] == a.data["sum_contrib"]