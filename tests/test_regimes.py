import json, pytest
from ally.tools.regimes import regime_gate

pytestmark = pytest.mark.mregimes

def test_regime_labels_and_stability(tmp_path):
    fx = {
      "dates":["2022-01-03","2022-01-04","2022-01-05"],
      "realized_vol_bps":[80,210,95],
      "illiq_score_bps":[4,12,6]
    }
    out = regime_gate(**fx).data
    assert set(out["res_alpha_t_per_regime"].keys()) == {"calm","normal","stressed"}
    assert isinstance(out["stable_ok"], bool)