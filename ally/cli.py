"""
Ally CLI using Typer for tool execution
"""

import json
import time
import typer
import orjson
from datetime import datetime
from typing import Optional

from .tools import TOOL_REGISTRY, execute_tool, list_tools, audit_logger
from .schemas.base import ToolResult, Meta
from .utils.serialization import convert_timestamps

app = typer.Typer(help="Ally CLI - Local-first agent tools for quantitative research")


@app.command()
def run(
    tool: str = typer.Argument(..., help="Tool name (e.g., web.fetch, data.load_ohlcv)"),
    json_args: str = typer.Argument(..., help="JSON string with tool arguments"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Audit run ID"),
    pretty: bool = typer.Option(False, "--pretty", help="Pretty print output"),
):
    """
    Execute a tool with JSON arguments
    
    Examples:
        ally run "web.fetch" '{"url": "https://example.com"}'
        ally run "data.load_ohlcv" '{"symbols": ["BTCUSDT"], "interval": "1h"}'
    """
    # Start audit run if provided
    if run_id:
        audit_logger.start_run(run_id)
    
    # Validate tool exists
    if tool not in TOOL_REGISTRY:
        typer.echo(f"❌ Tool '{tool}' not found", err=True)
        typer.echo(f"Available tools: {list(TOOL_REGISTRY.keys())}")
        raise typer.Exit(code=1)
    
    # Parse arguments
    try:
        args = json.loads(json_args)
    except json.JSONDecodeError as e:
        typer.echo(f"❌ Invalid JSON arguments: {e}", err=True)
        raise typer.Exit(code=1)
    
    # Execute tool
    try:
        result = execute_tool(tool, **args)
        
        # Normalize any pandas/numpy datetimes before JSON serialization
        result.data = convert_timestamps(result.data)
        
        # Output result
        if pretty:
            output = orjson.dumps(result.model_dump(), option=orjson.OPT_INDENT_2)
        else:
            output = orjson.dumps(result.model_dump())
            
        typer.echo(output.decode())
        
        # Exit with error code if tool failed
        if not result.ok:
            raise typer.Exit(code=1)
            
    except Exception as e:
        # Create error result
        error_result = ToolResult.error([str(e)])
        error_result.meta.ts = datetime.utcnow()
        
        if pretty:
            output = orjson.dumps(error_result.model_dump(), option=orjson.OPT_INDENT_2)
        else:
            output = orjson.dumps(error_result.model_dump())
            
        typer.echo(output.decode())
        raise typer.Exit(code=1)


@app.command("list")
def list_command():
    """List all available tools"""
    tools = list_tools()
    
    typer.echo("📋 Available Ally Tools:")
    typer.echo()
    
    for name, description in tools.items():
        typer.echo(f"🔧 {name}")
        typer.echo(f"   {description}")
        typer.echo()


@app.command()
def runs(limit: int = typer.Option(10, "--limit", help="Maximum number of runs to show")):
    """List recent audit runs"""
    runs = audit_logger.list_runs(limit=limit)
    
    if not runs:
        typer.echo("No runs found")
        return
    
    typer.echo(f"📊 Recent Runs (last {len(runs)}):")
    typer.echo()
    
    for run in runs:
        status_emoji = "✅" if run.get("status") == "completed" else "🔄"
        typer.echo(f"{status_emoji} {run.get('run_id', 'Unknown')}")
        typer.echo(f"   Started: {run.get('start_time', 'Unknown')}")
        if run.get("end_time"):
            typer.echo(f"   Ended: {run.get('end_time')}")
        typer.echo(f"   Tools: {len(run.get('tools_executed', []))}")
        typer.echo(f"   Duration: {run.get('total_duration_ms', 0)}ms")
        typer.echo()


@app.command()
def run_summary(run_id: str = typer.Argument(..., help="Run ID to summarize")):
    """Get detailed summary of a specific run"""
    summary = audit_logger.get_run_summary(run_id)
    
    if not summary:
        typer.echo(f"❌ Run '{run_id}' not found")
        raise typer.Exit(code=1)
    
    output = orjson.dumps(summary, option=orjson.OPT_INDENT_2)
    typer.echo(output.decode())


@app.command()
def test():
    """Run basic CLI smoke tests"""
    typer.echo("🧪 Running Ally CLI tests...")
    
    # Test 1: List tools (should not crash)
    try:
        tools = list_tools()
        typer.echo(f"✅ Tool registry: {len(tools)} tools found")
    except Exception as e:
        typer.echo(f"❌ Tool registry test failed: {e}")
        return
    
    # Test 2: JSON parsing
    try:
        test_json = '{"test": true, "value": 42}'
        args = json.loads(test_json)
        typer.echo("✅ JSON parsing: OK")
    except Exception as e:
        typer.echo(f"❌ JSON parsing test failed: {e}")
        return
    
    # Test 3: Audit logging
    try:
        run_id = audit_logger.start_run("TEST_CLI")
        audit_logger.end_run("completed")
        typer.echo("✅ Audit logging: OK")
    except Exception as e:
        typer.echo(f"❌ Audit logging test failed: {e}")
        return
    
    typer.echo("🎉 All CLI tests passed!")


if __name__ == "__main__":
    app()