from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

_FIX_DIR = Path("data/fixtures/runtime")

def generate_fixture(engine: str, task: str, prompt: str) -> str:
    # For determinism, choose an output list by engine+task; select first that matches a known prompt hash bucket (simple)
    fp = _FIX_DIR / f"{engine}.{task}.json"
    if not fp.exists():
        # fall back to generic fixture
        fp = _FIX_DIR / f"generic.{task}.json"
    try:
        data = json.loads(fp.read_text())
        # very simple deterministic pick: first item
        return data["outputs"][0]
    except Exception:
        return "FIXTURE_OUTPUT"