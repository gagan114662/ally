# ally/app/autopilot/spec_exec.py
from __future__ import annotations
from pathlib import Path
import subprocess, sys, json, hashlib, textwrap, os
from dataclasses import dataclass

@dataclass
class AutoResult:
    ok: bool
    rounds: int
    det_hash: str
    repo: str

def _sha1(s:str)->str: return hashlib.sha1(s.encode()).hexdigest()

def synthesize_spec(task:str)->str:
    return textwrap.dedent(f"""\
    # Spec
    Task: {task}
    Function: add(a,b) -> int
    Behavior: returns a+b for integers.
    """)

def generate_tests(repo:Path)->None:
    tests = repo/"tests"; tests.mkdir(exist_ok=True, parents=True)
    (tests/"test_autogen.py").write_text(textwrap.dedent("""\
        def test_add():
            from src.auto_impl import add
            assert add(2,3)==5
            assert add(-1,1)==0
    """))

def implement_code(repo:Path)->None:
    src = repo/"src"; src.mkdir(exist_ok=True, parents=True)
    (src/"auto_impl.py").write_text("def add(a,b):\n    return a+b\n")

def run_pytest(repo:Path)->bool:
    r = subprocess.run([sys.executable,"-m","pytest","-q"], cwd=repo)
    return r.returncode==0

def self_repair(repo:Path, max_rounds:int=2)->int:
    rounds=0
    while rounds<max_rounds and not run_pytest(repo):
        # trivial repair placeholder: rewrite impl (already correct)
        rounds+=1
    return rounds

def spec_exec(task_desc:str, max_rounds:int=2)->AutoResult:
    repo = Path("runs/autopilot/demo"); repo.mkdir(parents=True, exist_ok=True)
    (repo/"pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath=["src"]\n')
    (repo/"README.md").write_text(synthesize_spec(task_desc))
    generate_tests(repo); implement_code(repo)
    rounds = self_repair(repo, max_rounds=max_rounds)
    ok = run_pytest(repo)
    det = _sha1((repo/"src/auto_impl.py").read_text() + (repo/"tests/test_autogen.py").read_text())
    return AutoResult(ok=ok, rounds=rounds, det_hash=det, repo=str(repo))