import pytest
from ally.tools.fdr import fdr_gate

pytestmark = pytest.mark.mfdr

def test_fdr_gate_basic():
    cands = [
        {"sid":"A","spa_pvalue":0.01,"resid_alpha_t":2.5},
        {"sid":"B","spa_pvalue":0.02,"resid_alpha_t":2.1},
        {"sid":"C","spa_pvalue":0.2,"resid_alpha_t":1.0},
    ]
    out = fdr_gate(q=0.05,candidates=cands).data
    assert out["psi_ok"] is True
    assert set(out["passed"]) == {"A","B"}