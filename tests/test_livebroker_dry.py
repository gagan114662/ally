import os, json
from pathlib import Path
import pytest

pytestmark = pytest.mark.mlive

def test_place_order_dry():
    os.environ.pop("ALLY_LIVE", None)
    from ally.tools.broker import place_order
    out = place_order("SPY","buy",1.0,100.0, live=False, venue="paper", session_id="TST")
    assert out.ok
    rcpt_sha = out.data["receipt"]["sha1"]
    assert len(rcpt_sha)==40
    p = Path("runs/receipts")/f"{rcpt_sha}.json"
    assert p.exists()