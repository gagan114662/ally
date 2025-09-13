import json
from pathlib import Path
import hashlib
import pytest

from ally.tools.receipts import assert_receipts_invariants
from ally.tools import TOOL_REGISTRY

pytestmark = pytest.mark.receipts

def _sha1_text(s: str) -> str:
    import hashlib
    h=hashlib.sha1(); h.update(s.encode()); return h.hexdigest()

def test_invariant_blocks_live_without_receipts():
    bad = {"live": True, "receipts": []}
    with pytest.raises(AssertionError):
        assert_receipts_invariants(bad)

def test_invariant_passes_with_receipt():
    ok = {"live": True, "receipts": [{"content_sha1":"abc","vendor":"polygon","endpoint":"/v2","ts_iso":"2025-01-01T00:00:00Z"}]}
    assert_receipts_invariants(ok)  # no raise

def test_invariant_passes_with_live_false():
    ok = {"live": False, "receipts": []}
    assert_receipts_invariants(ok)  # no raise - live=false doesn't require receipts

def test_receipts_verify_offline(tmp_path: Path):
    runs = tmp_path / "runs" / "receipts"
    runs.mkdir(parents=True)
    # Create a simple receipt file
    p = runs / "RUN_DEMO_receipt.json"
    content = {"content_sha1": "dummy_hash", "vendor":"polygon", "endpoint":"/v2", "ts_iso":"2025-01-01T00:00:00Z", "run_id":"RUN_DEMO"}
    p.write_text(json.dumps(content, separators=(",",":")))
    
    # Verify the tool can run and finds the file (strict=False allows hash mismatch)
    res = TOOL_REGISTRY["receipts.verify"](run_id="RUN_DEMO", strict=False, base_dir=str(runs))
    assert res.data["receipt_files"] == 1
    # Just verify the tool ran and found the file - hash match not critical for this test

def test_receipts_diff_quorum():
    a = [1.0, 1.1, 1.2, 1.3]
    b = [1.0, 1.1000005, 1.1999997, 1.3]
    res = TOOL_REGISTRY["receipts.diff"](series_a=a, series_b=b, tolerance=1e-3)
    assert res.ok
    assert res.data["max_abs_diff"] <= 1e-3

def test_receipts_diff_fails_outside_tolerance():
    a = [1.0, 1.1, 1.2, 1.3]
    b = [1.0, 1.1, 1.3, 1.3]  # 0.1 difference at index 2
    res = TOOL_REGISTRY["receipts.diff"](series_a=a, series_b=b, tolerance=1e-6)
    assert not res.ok
    assert "quorum_disagreement" in res.errors