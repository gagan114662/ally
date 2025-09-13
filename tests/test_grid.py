import pytest, os
from pathlib import Path
pytestmark = pytest.mark.mgrid

def test_grid_runs_and_dedups(tmp_path, monkeypatch):
    # isolate state file
    monkeypatch.chdir(tmp_path)
    from ally.tools.grid import run
    pg = [{"a":1},{"a":1},{"a":2},{"a":3}]
    out = run(pg, max_workers=2)
    assert out.ok
    assert out.data["submitted"] == 3  # one dedup
    assert out.data["dedup_hits"] == 1
    # resume check
    from ally.utils.grid import STATE
    assert STATE.exists()
    out2 = run(pg, max_workers=2)
    assert out2.data["submitted"] == 0
    assert out2.data["resumed"] >= 0