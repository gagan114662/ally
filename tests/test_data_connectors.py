import os, json, hashlib
import pytest
from ally.tools import TOOL_REGISTRY

pytestmark = pytest.mark.mdata

def test_live_guard_offline_by_default(monkeypatch):
    monkeypatch.delenv("ALLY_LIVE", raising=False)
    res = TOOL_REGISTRY["data.fetch_live"](source="polygon", symbol="BTCUSD", interval="1h", live=False)
    assert not res.ok
    assert "live disabled" in (res.data.get("error","") or "")

def test_fixture_fallback_and_hash(monkeypatch):
    # enable live flag but provider falls back to fixture (acceptable in this PR)
    monkeypatch.setenv("ALLY_LIVE","1")
    res = TOOL_REGISTRY["data.fetch_live"](source="polygon", symbol="BTCUSD", interval="1h", live=True)
    assert res.ok
    frame = res.data["frame"]
    assert isinstance(frame, list) and len(frame) >= 2
    s = json.dumps(frame, sort_keys=True).encode()
    h = hashlib.sha1(s).hexdigest()
    assert len(h) == 40

def test_tavily_guard(monkeypatch):
    monkeypatch.delenv("ALLY_LIVE", raising=False)
    res = TOOL_REGISTRY["web.search_tavily"](q="nvda earnings", recency_days=30, live=False)
    assert not res.ok
    assert "live disabled" in (res.data.get("error","") or "")