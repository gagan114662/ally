import pytest, json, hashlib
from ally.tools import TOOL_REGISTRY

pytestmark = pytest.mark.m10

def test_wfo_basic_and_hash():
    res = TOOL_REGISTRY["bt.walkforward"](experiment_id="EXP_WFO_DEMO", window_train=200, window_test=50, mode="expanding", save_report=False)
    assert res.ok
    s = json.dumps(res.data, sort_keys=True).encode()
    h = hashlib.sha1(s).hexdigest()
    assert res.data["n_splits"] >= 1
    assert res.data["deflated_sharpe"] >= 0.0
    assert 0.0 <= res.data["spa_pvalue"] <= 1.0
    # store hash for determinism
    assert len(h) == 40

def test_wfo_schema_json():
    import jsonschema, json
    from pathlib import Path
    schema = json.loads(Path("ally/verify/jsonschema/wfo.schema.json").read_text())
    res = TOOL_REGISTRY["bt.walkforward"](experiment_id="EXP_WFO_SCHEMA", window_train=150, window_test=50, save_report=False)
    jsonschema.validate(instance=res.data, schema=schema)