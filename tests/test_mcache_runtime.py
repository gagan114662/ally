import re
from ally.tools import TOOL_REGISTRY

def test_runtime_fixture_and_cache_roundtrip(capsys):
    # Clear cache
    TOOL_REGISTRY["cache.clear"](dir="runs/cache")

    # First call: expect cache miss and fixture mode
    r1 = TOOL_REGISTRY["runtime.generate"](task="codegen", prompt="write add", live=False, cache={"dir":"runs/cache"})
    assert r1.ok and r1.data["mode"] in ("fixture","ollama")
    out1 = r1.data["output"]
    captured = capsys.readouterr().out
    assert "PROOF:CACHE_HIT: 0" in captured
    key1 = re.search(r"PROOF:CACHE_KEY_HASH: ([0-9a-f]{40})", captured).group(1)
    mode1 = re.search(r"PROOF:RUNTIME_MODE: (\w+)", captured).group(1)
    assert mode1 in ("fixture","ollama")

    # Second call: same inputs -> cache hit
    r2 = TOOL_REGISTRY["runtime.generate"](task="codegen", prompt="write add", live=False, cache={"dir":"runs/cache"})
    assert r2.ok and r2.data["output"] == out1
    captured2 = capsys.readouterr().out
    assert "PROOF:CACHE_HIT: 1" in captured2
    assert key1 in captured2

def test_cache_key_stability(capsys):
    r = TOOL_REGISTRY["runtime.generate"](task="nlp", prompt="summarize apple news", system=None, params={"seed":1337}, live=False, cache={"dir":"runs/cache"})
    assert r.ok
    out = capsys.readouterr().out
    # deterministic 40-char sha1
    assert re.search(r"PROOF:CACHE_KEY_HASH: [0-9a-f]{40}", out)