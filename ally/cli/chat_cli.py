"""
Chat CLI - Command-line interface for Ally chat and TUI

Provides:
- ally chat [prompt] - Interactive chat or one-shot queries
- ally tui - Text-based dashboard

All operations are offline-safe and generate audit receipts.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click

# Import Ally components
try:
    from ally.chat.controller import ChatController
    from ally.ui.tui import AllyTUI
except ImportError:
    # Fallback for testing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from chat.controller import ChatController
    from ui.tui import AllyTUI


@click.group()
def cli():
    """Ally conversational interface commands"""
    pass


@cli.command()
@click.argument('prompt', required=False)
@click.option('--journal-path',
              default='artifacts/status/journal_ci.jsonl',
              help='Path to journal file')
@click.option('--seed',
              default=42,
              help='Deterministic seed')
def chat(prompt: Optional[str], journal_path: str, seed: int):
    """
    Interactive chat with Ally system

    Examples:
        ally chat "show status"
        ally chat "last 5 operations"
        ally chat  # Interactive mode
    """
    controller = ChatController(journal_path=journal_path, seed=seed)

    if prompt:
        # One-shot mode
        response = controller.handle(prompt)
        output = {
            "ok": response.ok,
            "message": response.message,
            "data": response.data,
            "receipt": response.receipt
        }
        click.echo(json.dumps(output, indent=2, sort_keys=True))
        return

    # Interactive REPL mode
    click.echo("🤖 Ally Chat (offline-safe)")
    click.echo("Available commands:")
    for cmd in controller.get_available_commands():
        click.echo(f"  - {cmd}")
    click.echo("Type 'quit' or 'exit' to leave, Ctrl+C to interrupt")
    click.echo()

    try:
        while True:
            try:
                user_input = click.prompt("> ", prompt_suffix="", show_default=False).strip()
            except click.Abort:
                click.echo("\nGoodbye!")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                click.echo("Goodbye!")
                break

            if not user_input:
                continue

            try:
                response = controller.handle(user_input)

                # Format response for interactive display
                if response.ok:
                    click.echo(f"✅ {response.message}")
                    if response.data:
                        # Show key data fields
                        if "ops" in response.data:
                            click.echo(f"   Found {len(response.data['ops'])} operations")
                        if "counters" in response.data:
                            counters = response.data["counters"]
                            if counters:
                                click.echo(f"   Top counters: {dict(list(counters.items())[:3])}")
                        if "phase" in response.data:
                            click.echo(f"   Current: {response.data['phase']} → {response.data.get('substep', 'N/A')}")
                else:
                    click.echo(f"❓ {response.message}")
                    if "help" in response.data:
                        click.echo(f"   {response.data['help']}")

                # Always show receipt for audit
                receipt_hash = response.receipt.get('receipt_hash', 'unknown')
                click.echo(f"   🧾 Receipt: {receipt_hash[:8]}...")
                click.echo()

            except Exception as e:
                click.echo(f"❌ Error: {e}")
                click.echo()

    except KeyboardInterrupt:
        click.echo("\n\nInterrupted. Goodbye!")


@cli.command()
@click.option('--journal-path',
              default='artifacts/status/journal_ci.jsonl',
              help='Path to journal file')
@click.option('--seed',
              default=42,
              help='Deterministic seed')
@click.option('--format',
              type=click.Choice(['full', 'compact', 'json']),
              default='compact',
              help='Output format')
def tui(journal_path: str, seed: int, format: str):
    """
    Launch text-based user interface dashboard

    Shows current system status, recent operations, and metrics.
    """
    try:
        dashboard = AllyTUI(journal_path=journal_path, seed=seed)

        if format == 'full':
            output = dashboard.render_dashboard()
        elif format == 'json':
            output = dashboard.export_json()
        else:  # compact
            output = dashboard.render_compact()

        click.echo(output)

    except FileNotFoundError:
        click.echo(f"❌ Journal file not found: {journal_path}")
        click.echo("   Run 'bash scripts/ci_phase11_status.sh' to generate sample data")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error rendering TUI: {e}")
        sys.exit(1)


@cli.command()
@click.option('--journal-path',
              default='artifacts/status/journal_ci.jsonl',
              help='Path to journal file')
def status(journal_path: str):
    """Quick status check (alias for 'chat "show status"')"""
    controller = ChatController(journal_path=journal_path, seed=42)
    response = controller.handle("show status")

    if response.ok and response.data:
        data = response.data
        click.echo(f"Phase: {data.get('phase', 'Unknown')} → {data.get('substep', 'Unknown')}")
        click.echo(f"Journal entries: {data.get('journal_entries', 0)}")
        click.echo(f"Counters: {len(data.get('counters', {}))}")
        click.echo(f"Recent ops: {len(data.get('recent_ops', []))}")
        click.echo(f"Receipt: {response.receipt.get('receipt_hash', 'unknown')[:8]}...")
    else:
        click.echo(f"❌ Status unavailable: {response.message}")


if __name__ == "__main__":
    cli()