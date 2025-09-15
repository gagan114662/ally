"""
Ally CLI using Typer for tool execution
"""

import json
import time
import typer
from datetime import datetime
from typing import Optional

from .tools import TOOL_REGISTRY, execute_tool, list_tools, audit_logger
from .schemas.base import ToolResult, Meta
from .utils.serialization import convert_timestamps

app = typer.Typer(help="Ally CLI - Local-first agent tools for quantitative research")

# Create portfolio subcommand group
portfolio_app = typer.Typer(help="Portfolio management and reporting commands")
app.add_typer(portfolio_app, name="portfolio")

# Create ops subcommand group
try:
    from ally.cli.ops_cli import ops_app
    app.add_typer(ops_app, name="ops")
except ImportError:
    pass  # Ops CLI not available

# Create execution subcommand group
try:
    from ally.cli.execution_cli import exec_app
    app.add_typer(exec_app, name="exec")
except ImportError:
    pass  # Execution CLI not available


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
            output = json.dumps(result.model_dump(), indent=2)
        else:
            output = json.dumps(result.model_dump())

        typer.echo(output)
        
        # Exit with error code if tool failed
        if not result.ok:
            raise typer.Exit(code=1)
            
    except Exception as e:
        # Create error result
        error_result = ToolResult.error([str(e)])
        error_result.meta.ts = datetime.utcnow()
        
        if pretty:
            output = json.dumps(error_result.model_dump(), indent=2)
        else:
            output = json.dumps(error_result.model_dump())

        typer.echo(output)
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
    
    output = json.dumps(summary, indent=2, default=str)
    typer.echo(output)


@app.command()
def proofs():
    """Show aggregated proof lines from receipts database and commits"""
    import os
    import subprocess
    from pathlib import Path
    
    typer.echo("🔍 Ally Proof Aggregator")
    typer.echo()
    
    # Check for receipt database
    receipts_db = "artifacts/proof_receipts.duckdb"
    if os.path.exists(receipts_db):
        typer.echo(f"✅ Found receipts database: {receipts_db}")
        
        # Try to read receipts from database
        try:
            from .utils.db import DatabaseManager
            db = DatabaseManager(receipts_db)
            receipts_result = db.query("receipts", limit=100)
            receipt_count = receipts_result.get("count", 0)
            
            typer.echo(f"📊 Total receipts: {receipt_count}")
            
            if receipt_count > 0:
                typer.echo("📝 Recent receipt PROOF lines:")
                for receipt in receipts_result.get("rows", [])[:5]:
                    tool_name = receipt.get("tool_name", "unknown")
                    args_hash = receipt.get("args_hash", "")[:8]
                    receipt_hash = receipt.get("receipt_hash", "")[:16] 
                    typer.echo(f"PROOF:run:{tool_name}@{args_hash}:{receipt_hash}")
                    
                if receipt_count > 5:
                    typer.echo(f"... and {receipt_count - 5} more receipts")
            
            db.close()
            
        except Exception as e:
            typer.echo(f"⚠️  Error reading receipts database: {e}")
    else:
        typer.echo(f"ℹ️  No receipts database found at {receipts_db}")
        typer.echo("   (This is normal for Phase 0)")
    
    typer.echo()
    
    # Check for PROOF lines in recent commits
    try:
        # Get last commit message
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%s%n%b"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode == 0:
            commit_msg = result.stdout.strip()
            typer.echo("📋 Checking last commit for PROOF lines...")
            
            proof_lines = []
            for line in commit_msg.split('\n'):
                if line.strip().startswith("PROOF:"):
                    proof_lines.append(line.strip())
            
            if proof_lines:
                typer.echo("✅ Found PROOF lines in last commit:")
                for proof in proof_lines:
                    typer.echo(f"  {proof}")
            else:
                typer.echo("ℹ️  No PROOF lines found in last commit")
        else:
            typer.echo("⚠️  Could not read git commit (not in a git repo?)")
            
    except FileNotFoundError:
        typer.echo("ℹ️  Git not available - skipping commit proof check")
    except Exception as e:
        typer.echo(f"⚠️  Error checking git commits: {e}")
    
    # Show file hashes for key files (demonstration)
    typer.echo()
    typer.echo("🔐 Key file PROOF hashes (first 16 chars):")
    
    key_files = [
        "ally/utils/receipts.py",
        "ally/utils/gating.py", 
        "ally/utils/hashing.py",
        "ally/utils/db.py",
        "ally/.env.example"
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            try:
                from .utils.hashing import hash_code
                file_hash = hash_code(file_path, algorithm="sha1")[:16]
                typer.echo(f"PROOF:file:{file_path}:{file_hash}")
            except Exception as e:
                typer.echo(f"⚠️  Error hashing {file_path}: {e}")
        else:
            typer.echo(f"❌ File not found: {file_path}")


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


@portfolio_app.command("report")
def portfolio_report(
    last: bool = typer.Option(False, "--last", help="Show last portfolio optimization result"),
    format_type: str = typer.Option("table", "--format", help="Output format (table, json)")
):
    """
    Generate portfolio reports and summaries

    Examples:
        ally portfolio report --last           # Show compact table of last optimization
        ally portfolio report --last --format json  # Show detailed JSON output
    """
    import os
    import json
    from pathlib import Path

    if last:
        # Look for the most recent weights artifact
        weights_file = Path("./ally/artifacts/research/portfolio/weights.json")

        if not weights_file.exists():
            typer.echo("❌ No portfolio weights found. Run a portfolio optimization first.")
            typer.echo("   Try: ally run portfolio.optimize '{}'")
            raise typer.Exit(code=1)

        try:
            # Read the weights artifact
            with open(weights_file, 'r') as f:
                portfolio_data = json.load(f)

            if format_type == "json":
                # Output full JSON
                output = json.dumps(portfolio_data, indent=2, default=str)
                typer.echo(output)
            else:
                # Generate compact table view
                typer.echo("📊 Last Portfolio Optimization Report")
                typer.echo("=" * 50)
                typer.echo()

                # Basic info
                typer.echo(f"Method: {portfolio_data.get('method', 'unknown')}")
                typer.echo(f"Timestamp: {portfolio_data.get('timestamp', 'unknown')}")
                typer.echo(f"Universe: {len(portfolio_data.get('universe', []))} assets")
                typer.echo()

                # Risk metrics
                typer.echo("📈 Risk Metrics:")
                typer.echo(f"  Ex-ante Vol:     {portfolio_data.get('ex_ante_vol', 0):.3f}")
                typer.echo(f"  Ex-ante Sharpe:  {portfolio_data.get('ex_ante_sr', 0):.3f}")
                typer.echo(f"  Gross Exposure:  {portfolio_data.get('gross_exposure', 0):.3f}")
                typer.echo(f"  Net Exposure:    {portfolio_data.get('net_exposure', 0):.3f}")
                typer.echo(f"  Turnover:        {portfolio_data.get('turnover', 0):.3f}")
                typer.echo(f"  Cost Drag:       {portfolio_data.get('cost_drag_annual', 0):.3f}")
                typer.echo()

                # Sizing parameters
                typer.echo("⚖️ Sizing Parameters:")
                typer.echo(f"  Kelly Cap:       {portfolio_data.get('kelly_cap', 0):.3f}")
                typer.echo(f"  Vol Target:      {portfolio_data.get('vol_target', 0):.3f}")
                typer.echo()

                # Constraints status
                constraints_ok = portfolio_data.get('constraints_ok', False)
                status_emoji = "✅" if constraints_ok else "❌"
                typer.echo(f"🚦 Constraints: {status_emoji} {'PASS' if constraints_ok else 'VIOLATIONS'}")

                violations = portfolio_data.get('violations', [])
                if violations:
                    typer.echo("   Violations:")
                    for violation in violations[:3]:  # Show first 3
                        typer.echo(f"   - {violation.get('message', 'Unknown violation')}")
                    if len(violations) > 3:
                        typer.echo(f"   ... and {len(violations) - 3} more")

                binding_caps = portfolio_data.get('binding_caps', [])
                if binding_caps:
                    typer.echo(f"   Binding caps: {', '.join(binding_caps)}")
                typer.echo()

                # Top weights
                weights = portfolio_data.get('weights', {})
                if weights:
                    typer.echo("📋 Top Holdings:")
                    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
                    for asset, weight in sorted_weights[:5]:  # Top 5
                        direction = "L" if weight > 0 else "S"
                        typer.echo(f"  {asset:<8} {direction} {abs(weight):>8.3f}")
                    if len(sorted_weights) > 5:
                        typer.echo(f"  ... and {len(sorted_weights) - 5} more positions")

                typer.echo()

                # Link to full artifact
                full_path = os.path.abspath(weights_file)
                typer.echo(f"📁 Full data: {full_path}")

        except json.JSONDecodeError as e:
            typer.echo(f"❌ Error parsing portfolio data: {e}")
            raise typer.Exit(code=1)
        except Exception as e:
            typer.echo(f"❌ Error reading portfolio report: {e}")
            raise typer.Exit(code=1)
    else:
        typer.echo("📊 Portfolio Report")
        typer.echo("Use --last to show the most recent optimization result")
        typer.echo("Other report types coming soon...")


if __name__ == "__main__":
    app()