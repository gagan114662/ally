"""
Orchestrator Runtime Integration - Local LLM capabilities
Demonstrates runtime cache integration for local development
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime

from ally.schemas.base import ToolResult, Meta
from ally.tools import TOOL_REGISTRY, register
from ally.utils.db import get_db_manager

def _maybe_runtime(task: str, prompt: str, use_runtime: bool, runtime_live: bool) -> Optional[str]:
    """
    Maybe use runtime generation based on flags
    
    Args:
        task: Task type for router selection
        prompt: Prompt to generate from
        use_runtime: Whether to use runtime at all
        runtime_live: Whether to try Ollama (True) or stick to fixtures (False)
    
    Returns:
        Generated text or None if runtime disabled
    """
    if not use_runtime:
        return None
    
    try:
        res = TOOL_REGISTRY["runtime.generate"](
            task=task, 
            prompt=prompt, 
            live=runtime_live, 
            cache={"dir": "runs/cache"}
        )
        return res.data["output"] if res.ok else None
    except Exception:
        return None

@register("orchestrator.demo")  
def orchestrator_demo(
    experiment_id: str = "demo",
    use_runtime: bool = False,
    runtime_live: bool = False,
    tasks: Optional[List[Dict[str, str]]] = None
) -> ToolResult:
    """
    Orchestrator runtime integration demonstration
    
    Args:
        experiment_id: Unique identifier for this run
        use_runtime: Enable runtime generation (False = skip, True = use cache+fixtures/ollama)
        runtime_live: Try Ollama if available (True), or use fixtures only (False)  
        tasks: List of {"task": "codegen|nlp|math|cv", "prompt": "..."} to demonstrate
        
    Returns:
        ToolResult with orchestration results and runtime statistics
    """
    
    # Default demonstration tasks
    if tasks is None:
        tasks = [
            {"task": "codegen", "prompt": "Write a simple function to calculate fibonacci numbers"},
            {"task": "nlp", "prompt": "Summarize: The market showed strong performance with tech stocks leading gains"},
            {"task": "math", "prompt": "Calculate the compound annual growth rate for 10% over 5 years"},
            {"task": "cv", "prompt": "Describe a bullish candlestick pattern"}
        ]
    
    results = {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "runtime_config": {
            "use_runtime": use_runtime,
            "runtime_live": runtime_live
        },
        "task_results": [],
        "runtime_stats": {
            "cache_hits": 0,
            "cache_misses": 0,
            "runtime_mode": "disabled" if not use_runtime else ("live" if runtime_live else "fixtures")
        }
    }
    
    # Process each task
    for i, task_spec in enumerate(tasks):
        task_type = task_spec["task"]
        prompt = task_spec["prompt"]
        
        print(f"=== Task {i+1}: {task_type.upper()} ===")
        print(f"Prompt: {prompt}")
        
        # Try runtime generation
        generated = _maybe_runtime(task_type, prompt, use_runtime, runtime_live)
        
        task_result = {
            "task": task_type,
            "prompt": prompt,
            "generated": generated,
            "method": "runtime" if generated else "passthrough"
        }
        
        if generated:
            print(f"Generated: {generated[:100]}...")
            results["runtime_stats"]["cache_misses"] += 1
        else:
            print("Generated: [runtime disabled - using passthrough]")
        
        results["task_results"].append(task_result)
        print()
    
    # Runtime integration proofs (only when runtime is enabled)  
    if use_runtime:
        print("PROOF:ORCH_RUNTIME_ENABLED: true")
        print(f"PROOF:ORCH_RUNTIME_MODE: {results['runtime_stats']['runtime_mode']}")
        print(f"PROOF:ORCH_TASK_COUNT: {len(tasks)}")
        print(f"PROOF:ORCH_EXPERIMENT_ID: {experiment_id}")
    else:
        print("PROOF:ORCH_RUNTIME_ENABLED: false")
        print("PROOF:ORCH_RUNTIME_MODE: disabled")
    
    return ToolResult(
        ok=True,
        data=results,
        errors=[],
        meta=Meta(ts=datetime.utcnow(), duration_ms=0)
    )


@register("orchestrator.run")
def orchestrator_run(
    experiment_id: str,
    symbols: Optional[List[str]] = None,
    use_live_data: bool = False,
    live_quorum: Optional[Dict[str, Any]] = None,
    live_budget_cents: int = 100,
    use_runtime: bool = False,
    runtime_live: bool = False
) -> ToolResult:
    """
    Main orchestrator run with live data and runtime integration
    
    Args:
        experiment_id: Unique identifier for this orchestration run
        symbols: List of symbols to analyze (e.g., ["AAPL", "MSFT"])
        use_live_data: Enable live data fetching (requires ALLY_LIVE=1)
        live_quorum: Cross-provider verification config
        live_budget_cents: Maximum cost for live data
        use_runtime: Enable runtime generation
        runtime_live: Use live Ollama vs fixtures
        
    Returns:
        ToolResult with orchestration results and receipt tracking
    """
    
    # Default symbols for demonstration
    if symbols is None:
        symbols = ["AAPL", "MSFT"]
    
    results = {
        "experiment_id": experiment_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "symbols": symbols,
            "use_live_data": use_live_data,
            "live_quorum": live_quorum,
            "live_budget_cents": live_budget_cents,
            "use_runtime": use_runtime,
            "runtime_live": runtime_live
        },
        "data_receipts": [],
        "runtime_tasks": [],
        "analysis_results": []
    }
    
    # Step 1: Fetch live data if enabled
    if use_live_data:
        try:
            for symbol in symbols:
                # Fetch historical data with receipts
                data_result = TOOL_REGISTRY["data.live_history"](
                    symbol=symbol,
                    start="2025-01-01T00:00:00Z",
                    end="2025-01-15T00:00:00Z",
                    interval="1d",
                    vendor="polygon",
                    live=use_live_data,
                    budget_cents=live_budget_cents // len(symbols),  # Split budget
                    quorum=live_quorum
                )
                
                if data_result.ok:
                    if "receipt" in data_result.data:
                        results["data_receipts"].append(data_result.data["receipt"]["content_sha1"])
                    elif "receipts" in data_result.data:
                        for receipt in data_result.data["receipts"]:
                            results["data_receipts"].append(receipt["content_sha1"])
                else:
                    results["analysis_results"].append({
                        "symbol": symbol,
                        "status": "data_fetch_failed",
                        "error": data_result.errors
                    })
                    continue
                    
        except Exception as e:
            print(f"Live data fetch error: {e}")
            results["live_data_error"] = str(e)
    
    # Step 2: Runtime analysis for each symbol
    if use_runtime:
        for symbol in symbols:
            analysis_prompt = f"Analyze {symbol} market conditions and provide trading insights"
            
            runtime_result = _maybe_runtime("nlp", analysis_prompt, use_runtime, runtime_live)
            
            results["runtime_tasks"].append({
                "symbol": symbol,
                "task": "market_analysis",
                "prompt": analysis_prompt,
                "generated": runtime_result is not None,
                "output": runtime_result[:100] + "..." if runtime_result else None
            })
    
    # Step 3: Store run metadata with receipt tracking
    try:
        db_manager = get_db_manager()
        
        # Calculate metrics
        metrics = {
            "symbols_count": len(symbols),
            "receipts_count": len(results["data_receipts"]),
            "runtime_tasks_count": len(results["runtime_tasks"]),
            "live_data_enabled": float(use_live_data),
            "runtime_enabled": float(use_runtime)
        }
        
        # Store run
        db_manager.log_run(
            run_id=experiment_id,
            task="orchestrator_run",
            code_hash="orch_run_v1",  # Version identifier
            inputs_hash=f"symbols={len(symbols)}_live={use_live_data}_runtime={use_runtime}",
            ts=results["timestamp"],
            metrics=metrics,
            events=[{"type": "orchestrator_run", "config": results["config"]}],
            trades=[],  # No trades in this demo
            notes=f"Orchestrator run with {len(results['data_receipts'])} receipts"
        )
        
        results["run_stored"] = True
        
    except Exception as e:
        results["run_stored"] = False
        results["storage_error"] = str(e)
    
    # Generate proof lines
    print("=== ORCHESTRATOR RUN PROOFS ===")
    print(f"PROOF:ORCH_EXPERIMENT_ID: {experiment_id}")
    print(f"PROOF:ORCH_SYMBOLS_COUNT: {len(symbols)}")
    print(f"PROOF:ORCH_LIVE_DATA: {use_live_data}")
    print(f"PROOF:ORCH_RUNTIME: {use_runtime}")
    print(f"PROOF:ORCH_RECEIPTS_COUNT: {len(results['data_receipts'])}")
    
    if results["data_receipts"]:
        print(f"PROOF:ORCH_SAMPLE_RECEIPT: {results['data_receipts'][0]}")
    
    if use_live_data and live_quorum:
        print(f"PROOF:ORCH_QUORUM_VENDORS: {len(live_quorum.get('vendors', []))}")
        print(f"PROOF:ORCH_QUORUM_TOLERANCE: {live_quorum.get('tolerance_bps', 0)}")
    
    return ToolResult(
        ok=True,
        data=results,
        errors=[],
        meta=Meta(ts=datetime.utcnow(), duration_ms=0)
    )