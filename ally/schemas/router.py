from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

Task = Literal["codegen", "nlp", "math", "cv"]

class RouterConfig(BaseModel):
    # Candidate engines per task (strings only; works offline)
    candidates: Dict[Task, List[str]] = {
        "codegen": ["qwen2.5-coder-7b", "codestral-22b", "llama3.1-8b-instruct"],
        "nlp":     ["llama3.1-8b", "mistral-7b-instruct", "qwen2.5-7b-instruct"],
        "math":    ["deepseek-math-7b", "llama3.1-8b", "qwen2.5-7b-instruct"],
        "cv":      ["moondream-2", "llava-phi-3-mini", "llama-vision-8b"]
    }
    # Fallback order per task (subset of candidates; optional)
    fallback: Dict[Task, List[str]] = {}
    # Offline by default (no network, uses fixtures)
    offline: bool = True
    # Path to deterministic eval set (local json)
    eval_path: str = "data/fixtures/router/eval_set.json"

class RouterDecision(BaseModel):
    task: Task
    winner: str
    scores: Dict[str, float] = Field(default_factory=dict)
    fallback_used: bool = False

class RouterMatrix(BaseModel):
    matrix: Dict[Task, str]                 # best engine per task
    scores: Dict[Task, Dict[str, float]]    # per-task, per-engine scores
    fallback_ok: bool = True
    eval_det_hash: str