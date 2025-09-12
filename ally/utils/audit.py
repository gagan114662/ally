"""
Audit utilities for tool execution tracking and reproducibility
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..schemas.base import ToolResult, Meta
from .hashing import hash_inputs, hash_code, content_hash
from .serialization import convert_timestamps


class AuditLogger:
    """
    Tracks tool executions for reproducibility and debugging
    """
    
    def __init__(self, runs_dir: Path = None):
        self.runs_dir = runs_dir or Path("runs")
        self.runs_dir.mkdir(exist_ok=True)
        self.current_run_id = None
        self.current_run_dir = None
        
    def start_run(self, run_id: str = None) -> str:
        """Start a new audit run"""
        if run_id is None:
            run_id = f"RUN_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"
            
        self.current_run_id = run_id
        self.current_run_dir = self.runs_dir / run_id
        self.current_run_dir.mkdir(exist_ok=True)
        
        # Create manifest file
        manifest = {
            "run_id": run_id,
            "start_time": datetime.now().isoformat(),
            "tools_executed": [],
            "total_duration_ms": 0,
            "status": "running"
        }
        
        self._write_manifest(manifest)
        return run_id
        
    def log_tool_execution(self, tool_name: str, inputs: Dict[str, Any], 
                          result: ToolResult, func_code_hash: str = None) -> None:
        """Log a tool execution"""
        if not self.current_run_id:
            self.start_run()
            
        # Create execution record with serializable inputs
        serializable_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, 'to_dict') and hasattr(value, 'columns'):  # DataFrame-like
                # Convert DataFrame to serializable dict
                converted_df = convert_timestamps(value)
                serializable_inputs[key] = {
                    "_type": "DataFrame",
                    "shape": converted_df.shape,
                    "columns": list(converted_df.columns),
                    "data": converted_df.to_dict(orient='records')[:5]  # Sample first 5 rows
                }
            else:
                serializable_inputs[key] = convert_timestamps(value)
        execution = {
            "tool_name": tool_name,
            "timestamp": datetime.now().isoformat(),
            "inputs": serializable_inputs,
            "inputs_hash": hash_inputs(inputs),
            "code_hash": func_code_hash,
            "result": {
                "ok": result.ok,
                "status": result.status.value,
                "data_keys": list(result.data.keys()),
                "errors": result.errors,
                "warnings": result.meta.warnings,
                "duration_ms": result.meta.duration_ms
            }
        }
        
        # Save individual execution log
        exec_file = self.current_run_dir / f"{tool_name}_{int(time.time())}.json"
        with open(exec_file, 'w') as f:
            json.dump(execution, f, indent=2)
            
        # Update manifest
        self._update_manifest_with_execution(execution)
        
    def end_run(self, status: str = "completed") -> None:
        """End the current audit run"""
        if not self.current_run_id:
            return
            
        manifest = self._read_manifest()
        manifest["end_time"] = datetime.now().isoformat()
        manifest["status"] = status
        
        if "start_time" in manifest:
            start_time = datetime.fromisoformat(manifest["start_time"])
            end_time = datetime.fromisoformat(manifest["end_time"])
            manifest["total_duration_ms"] = int((end_time - start_time).total_seconds() * 1000)
            
        self._write_manifest(manifest)
        self.current_run_id = None
        self.current_run_dir = None
        
    def get_run_summary(self, run_id: str = None) -> Dict[str, Any]:
        """Get summary of a specific run"""
        if run_id is None:
            run_id = self.current_run_id
            
        if not run_id:
            return {}
            
        run_dir = self.runs_dir / run_id
        manifest_file = run_dir / "manifest.json"
        
        if not manifest_file.exists():
            return {}
            
        with open(manifest_file, 'r') as f:
            return json.load(f)
            
    def list_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent runs"""
        runs = []
        
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if run_dir.is_dir() and len(runs) < limit:
                summary = self.get_run_summary(run_dir.name)
                if summary:
                    runs.append(summary)
                    
        return runs
        
    def _read_manifest(self) -> Dict[str, Any]:
        """Read current run manifest"""
        if not self.current_run_dir:
            return {}
            
        manifest_file = self.current_run_dir / "manifest.json"
        if not manifest_file.exists():
            return {}
            
        with open(manifest_file, 'r') as f:
            return json.load(f)
            
    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        """Write manifest to current run directory"""
        if not self.current_run_dir:
            return
            
        manifest_file = self.current_run_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
            
    def _update_manifest_with_execution(self, execution: Dict[str, Any]) -> None:
        """Update manifest with new execution"""
        manifest = self._read_manifest()
        
        if "tools_executed" not in manifest:
            manifest["tools_executed"] = []
            
        # Add execution summary to manifest
        exec_summary = {
            "tool_name": execution["tool_name"],
            "timestamp": execution["timestamp"],
            "inputs_hash": execution["inputs_hash"],
            "code_hash": execution.get("code_hash"),
            "ok": execution["result"]["ok"],
            "duration_ms": execution["result"]["duration_ms"]
        }
        
        manifest["tools_executed"].append(exec_summary)
        self._write_manifest(manifest)