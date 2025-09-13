# ally/tools/grid.py
from __future__ import annotations
from ally.utils.grid import run_matrix
from ally.schemas.base import ToolResult

def run(param_grid:list[dict], max_workers:int=4)->ToolResult:
    out = run_matrix(param_grid, max_workers=max_workers)
    return ToolResult(ok=True, data=out)