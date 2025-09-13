import json, hashlib
from statistics import mean
from ally.schemas.regimes import RegimeInput, RegimeResult
from ally.schemas.base import ToolResult

def _sha1(o):
    import json,hashlib
    return hashlib.sha1(json.dumps(o,sort_keys=True).encode()).hexdigest()

def _label(vol, illiq):
    if vol < 100 and illiq < 5: return "calm"
    if vol >= 200 or illiq >= 15: return "stressed"
    return "normal"

def analyze_regimes(dates, vol, illiq):
    labels = [_label(v,i) for v,i in zip(vol, illiq)]
    # dummy residual alpha t per regime from deterministic mapping (CI fixtures only)
    by_reg = {"calm": [], "normal": [], "stressed": []}
    for lab,v in zip(labels, vol):
        by_reg[lab].append(v)
    res_t = {k: round( (100 - mean(vs))/100.0 if vs else 0.0, 3) for k,vs in by_reg.items()}
    nonzero = [k for k,v in res_t.items() if abs(v) >= 0.02]
    stable_ok = len(nonzero) >= 2
    h = _sha1({"labels":labels,"res_t":res_t})
    return RegimeResult(labels=labels, res_alpha_t_per_regime=res_t, stable_ok=stable_ok, det_hash=h)

def regime_gate(dates, realized_vol_bps, illiq_score_bps) -> ToolResult:
    r = analyze_regimes(dates, realized_vol_bps, illiq_score_bps)
    return ToolResult(ok=True, data=r.dict())