# tests/test_ops_ask.py
import os, json
from pathlib import Path
import pytest

pytestmark = pytest.mark.mops

from ally.tools.ops import ask, ASK_QUEUE

def test_ops_ask_dry_mode_idempotent(tmp_path, monkeypatch):
    # ensure isolated queue
    monkeypatch.chdir(tmp_path)
    # first ask
    r1 = ask("What is 2+2?", {"note":"demo"}, live=False)
    assert r1.ok and r1.data["mode"] == "dry"
    h1 = r1.data["hash"]
    assert Path("runs/supervisor/ask_queue.json").exists()
    # second identical ask -> same hash, queue grows
    r2 = ask("What is 2+2?", {"note":"demo"}, live=False)
    assert r2.ok and r2.data["mode"] == "dry"
    assert r2.data["hash"] == h1
    # queue length == 2
    q = json.loads(Path("runs/supervisor/ask_queue.json").read_text())
    assert len(q) == 2

def test_ops_ask_live_denied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ALLY_LIVE", raising=False)
    r = ask("Live?", live=True)
    assert r.ok
    assert r.data["mode"] == "dry"
    assert r.data.get("live_denied") is True
    assert ASK_QUEUE.exists()