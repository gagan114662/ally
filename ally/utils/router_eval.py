from __future__ import annotations
import json, hashlib, math
from typing import Dict, List, Tuple

def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in s.replace("\n"," ").split() if t.strip()]

def _f1(pred: str, gold: str) -> float:
    p = _tokenize(pred); g = _tokenize(gold)
    if not p or not g: return 0.0
    inter = 0
    gp = g.copy()
    for t in p:
        if t in gp:
            inter += 1
            gp.remove(t)
    prec = inter/len(p); rec = inter/len(g)
    if prec+rec == 0: return 0.0
    return 2*prec*rec/(prec+rec)

def _num_score(pred: str, gold: str) -> float:
    # extract all numbers and compare; simple, deterministic
    import re
    pnums = [float(x) for x in re.findall(r"-?\d+\.?\d*", pred)]
    gnums = [float(x) for x in re.findall(r"-?\d+\.?\d*", gold)]
    if not gnums: return 0.0
    # L1 error inverted score
    err = sum(abs((pnums[i] if i < len(pnums) else 0.0) - gnums[i]) for i in range(len(gnums)))
    return 1.0 / (1.0 + err)

def _code_score(pred: str, gold: str) -> float:
    # heuristic: code presence + overlap
    has_def = "def " in pred or "class " in pred
    has_return = "return" in pred or "pass" in pred
    base = 0.3 if (has_def and has_return) else 0.0
    return base + 0.7 * _f1(pred, gold)

def _cv_caption_score(pred: str, gold: str) -> float:
    # textual proxy; token overlap
    return _f1(pred, gold)

def _task_metric(task: str, pred: str, gold: str) -> float:
    if task == "codegen": return _code_score(pred, gold)
    if task == "math":    return _num_score(pred, gold)
    if task == "nlp":     return _f1(pred, gold)
    if task == "cv":      return _cv_caption_score(pred, gold)
    return 0.0

def score_dataset(eval_set: Dict, engines: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, Dict[str,float]], str]:
    """
    eval_set: {"codegen":[{"prompt":..,"gold":..},...], "math":[...], "nlp":[...], "cv":[...]}
    engines:  {engine_id: {task: "joined outputs keyed by item index"}}
              For offline determinism we store per-engine outputs in fixtures keyed by task+index.

    Returns:
      scores_by_task_engine: {task: {engine: avg_score}}
      eval_det_hash: sha1 over inputs+engine outputs to lock determinism
    """
    # Deterministic hash includes prompts, golds, and engine outputs (strings)
    hasher = hashlib.sha1()
    hasher.update(json.dumps(eval_set, sort_keys=True).encode())
    hasher.update(json.dumps(engines,   sort_keys=True).encode())

    scores: Dict[str, Dict[str, float]] = {}
    for task, items in eval_set.items():
        scores[task] = {}
        for eng, outputs in engines.items():
            agg = 0.0; n = 0
            for i, ex in enumerate(items):
                # outputs[task] is a list indexed like eval_set[task]
                pred_list = outputs.get(task, [])
                pred = pred_list[i] if i < len(pred_list) else ""
                gold = ex["gold"]
                s = _task_metric(task, pred, gold)
                agg += s; n += 1
            scores[task][eng] = (agg / max(n,1))
    return scores, hasher.hexdigest()