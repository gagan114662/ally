# ally/utils/grid.py
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, hashlib, time, random

STATE = Path("runs/grid_state.json")

def _sha1_obj(o)->str:
    return hashlib.sha1(json.dumps(o, sort_keys=True, separators=(",",":")).encode()).hexdigest()

def _load_state()->dict:
    if STATE.exists(): return json.loads(STATE.read_text())
    return {"done":{}, "started":{}}

def _save_state(s:dict)->None:
    STATE.parent.mkdir(exist_ok=True, parents=True)
    STATE.write_text(json.dumps(s, indent=2, sort_keys=True))

def _backtest_job(params:dict)->dict:
    # deterministic pseudo "work"
    random.seed(1337 + int(_sha1_obj(params)[:8],16)%10)
    time.sleep(0.01)
    return {"sharpe": round(random.uniform(-0.5, 2.5), 3)}

def run_matrix(param_grid:list[dict], max_workers:int=4)->dict:
    state = _load_state()
    submitted = 0; dedup_hits = 0; resumed = 0
    results = {}

    todo = []
    for p in param_grid:
        k = _sha1_obj(p)
        if k in state["done"]:
            dedup_hits += 1
            continue
        todo.append((k,p))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs={}
        for k,p in todo:
            if k in state["started"]:
                resumed += 1
            state["started"][k]=p
            futs[ex.submit(_backtest_job, p)] = (k,p)
            submitted += 1
        for f in as_completed(futs):
            k,p = futs[f]
            res = f.result()
            results[k]=res
            state["done"][k]=res
    _save_state(state)
    return {"submitted":submitted,"dedup_hits":dedup_hits,"resumed":resumed,"results":results}