from __future__ import annotations
import json, subprocess, shutil
from typing import Dict, Any, Optional

def has_ollama() -> bool:
    return shutil.which("ollama") is not None

def generate_ollama(model: str, prompt: str, system: Optional[str], params: Dict[str,Any]) -> str:
    # Deterministic defaults
    body = {
        "model": model,
        "prompt": prompt,
        "system": system or "",
        "options": {
            "temperature": 0,
            "top_p": 0,
            "seed": 1337,
            "repeat_penalty": 1.0
        }
    }
    # Allow safe overrides
    body["options"].update({k:v for k,v in params.items() if k in ("temperature","top_p","seed","repeat_penalty","num_ctx")})
    try:
        p = subprocess.run(["ollama","generate","-j"], input=json.dumps(body).encode(), capture_output=True, check=True)
        # ollama streams JSON lines; we join their "response" fields
        out = ""
        for line in p.stdout.splitlines():
            try:
                obj = json.loads(line.decode())
                out += obj.get("response","")
            except: pass
        return out
    except Exception as e:
        # Bubble up for upper layer to fallback to fixtures
        raise RuntimeError(f"Ollama generation failed: {e}")