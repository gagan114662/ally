# ally/tools/autopilot.py
from __future__ import annotations
from ally.app.autopilot.spec_exec import spec_exec
from ally.schemas.base import ToolResult

def run(task_desc:str="implement add", max_rounds:int=2)->ToolResult:
    res = spec_exec(task_desc, max_rounds=max_rounds)
    return ToolResult(ok=res.ok, data={"rounds":res.rounds,"det_hash":res.det_hash,"repo":res.repo})