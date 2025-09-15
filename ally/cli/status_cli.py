"""
Status CLI - Command-line interface for Ally status reporting

Provides human and machine-readable status information:
- `ally status` - Human-friendly overview
- `ally status --json` - Machine-readable JSON output
- `ally status --journal` - Recent journal entries
- `ally status --metrics` - Detailed telemetry metrics

Integrates runbook state, journal history, and telemetry data.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
import click

# Import Ally status modules
try:
    from ally.status.runbook import Runbook, get_phase_help, get_substep_help
    from ally.status.journal import Journal
    from ally.status.telemetry import Telemetry
except ImportError:
    # Fallback for testing
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from status.runbook import Runbook, get_phase_help, get_substep_help
    from status.journal import Journal
    from status.telemetry import Telemetry


class StatusCLI:
    """Status command-line interface"""

    def __init__(self, journal_path: str = "artifacts/status/journal.jsonl",
                 artifacts_dir: str = "artifacts/status"):
        """
        Initialize status CLI

        Args:
            journal_path: Path to journal file
            artifacts_dir: Directory for status artifacts
        """
        self.journal_path = journal_path
        self.artifacts_dir = Path(artifacts_dir)

        # Ensure artifacts directory exists
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.runbook = Runbook(deterministic=True, seed=42)
        self.journal = Journal(journal_path, deterministic_seed=42)
        self.telemetry = Telemetry(seed=42)

        # Load any existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load existing state from artifacts"""
        # Try to load last known state
        state_file = self.artifacts_dir / "last_state.json"
        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    state = json.load(f)
                    if "phase" in state and "substep" in state:
                        self.runbook.enter(state["phase"], state["substep"], "Restored state")
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self) -> None:
        """Save current state to artifacts"""
        state = {
            "phase": self.runbook.phase,
            "substep": self.runbook.substep,
            "timestamp": self.runbook.get_status()["timestamp_utc"]
        }

        state_file = self.artifacts_dir / "last_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def get_human_status(self) -> str:
        """Get human-readable status report"""
        status = self.runbook.get_status()
        journal_summary = self.journal.get_summary()
        telemetry_stats = self.telemetry.get_stats()

        # Build human-readable report
        lines = []
        lines.append("🤖 Ally System Status")
        lines.append("=" * 50)

        # Current state
        lines.append(f"📊 Current State: {status['phase']} → {status['substep']}")

        phase_help = get_phase_help().get(status['phase'], "")
        substep_help = get_substep_help().get(status['substep'], "")

        if phase_help:
            lines.append(f"   Phase: {phase_help}")
        if substep_help:
            lines.append(f"   Step:  {substep_help}")

        lines.append("")

        # Recent activity
        last_entry = self.journal.get_last_entry()
        if last_entry:
            lines.append(f"🔄 Last Activity: {last_entry['note']}")
            lines.append(f"   Tool: {last_entry['tool']}")
            lines.append(f"   Time: {last_entry['ts_utc']}")
        else:
            lines.append("🔄 Last Activity: None recorded")

        lines.append("")

        # Operation counts
        counters = telemetry_stats.get('counters', {})
        if counters:
            lines.append("📈 Operation Counts:")
            for name, count in sorted(counters.items()):
                lines.append(f"   {name}: {count}")
        else:
            lines.append("📈 Operation Counts: None")

        lines.append("")

        # Recent receipts
        last_receipts = telemetry_stats.get('last_receipts', [])
        if last_receipts:
            lines.append("🧾 Recent Receipts:")
            for receipt in last_receipts[:3]:  # Show last 3
                lines.append(f"   {receipt['tool']}: {receipt['receipt_hash'][:8]}...")
        else:
            lines.append("🧾 Recent Receipts: None")

        lines.append("")

        # System health
        error_count = counters.get('error.total', 0)
        success_count = counters.get('success.total', 0)
        total_ops = error_count + success_count

        if total_ops > 0:
            success_rate = (success_count / total_ops) * 100
            lines.append(f"💚 Success Rate: {success_rate:.1f}% ({success_count}/{total_ops})")
            if error_count > 0:
                lines.append(f"🔴 Error Count: {error_count}")
        else:
            lines.append("💚 Success Rate: N/A (no operations)")

        lines.append("")

        # Journal stats
        lines.append(f"📚 Journal Entries: {journal_summary['total_entries']}")
        lines.append(f"🔧 Tools Used: {len(journal_summary.get('tools', []))}")

        return "\n".join(lines)

    def get_json_status(self) -> Dict:
        """Get machine-readable status JSON"""
        runbook_status = self.runbook.get_status()
        journal_summary = self.journal.get_summary()
        telemetry_stats = self.telemetry.get_stats()
        last_entry = self.journal.get_last_entry()

        return {
            "phase": runbook_status["phase"],
            "substep": runbook_status["substep"],
            "timestamp": runbook_status["timestamp_utc"],
            "last_tool": last_entry.get("tool", "") if last_entry else "",
            "last_params_hash": last_entry.get("params_hash", "") if last_entry else "",
            "last_receipt_hash": last_entry.get("receipt_hash", "") if last_entry else "",
            "counters": telemetry_stats.get("counters", {}),
            "timers": telemetry_stats.get("timer_stats", {}),
            "success_rate": self._calculate_success_rate(telemetry_stats.get("counters", {})),
            "journal_entries": journal_summary["total_entries"],
            "tools_used": journal_summary.get("tools", []),
            "deterministic": runbook_status["deterministic"],
            "artifacts_dir": str(self.artifacts_dir)
        }

    def get_journal_entries(self, limit: int = 20, phase_filter: str = None) -> List[Dict]:
        """Get recent journal entries"""
        return self.journal.read_entries(limit=limit, filter_phase=phase_filter)

    def get_metrics_detail(self) -> Dict:
        """Get detailed telemetry metrics"""
        return self.telemetry.get_stats()

    def update_status(self, phase: str, substep: str, tool: str = "",
                     params: Dict = None, result: str = "") -> bool:
        """
        Update system status

        Args:
            phase: New phase
            substep: New substep
            tool: Tool being used
            params: Tool parameters
            result: Tool result

        Returns:
            True if update successful
        """
        # Update runbook
        if not self.runbook.enter(phase, substep, f"CLI update to {phase}:{substep}"):
            return False

        # Log to journal
        if tool:
            params_hash = self.journal.hash_params(params or {})
            receipt_hash = self.journal.hash_receipt(result)

            self.journal.append(
                phase=phase,
                step=substep,
                tool=tool,
                params_hash=params_hash,
                receipt_hash=receipt_hash,
                note=f"CLI: {tool} invocation"
            )

            # Update telemetry
            self.telemetry.record_receipt(tool, params_hash, receipt_hash)
            self.telemetry.count(f"cli.{tool}")

        # Save state
        self._save_state()
        return True

    def _calculate_success_rate(self, counters: Dict[str, int]) -> float:
        """Calculate success rate from counters"""
        success = counters.get('success.total', 0)
        errors = counters.get('error.total', 0)
        total = success + errors

        if total == 0:
            return 0.0

        return (success / total) * 100.0


@click.group()
def status():
    """Ally system status commands"""
    pass


@status.command()
@click.option('--json', 'output_json', is_flag=True, help='Output JSON format')
@click.option('--journal', is_flag=True, help='Show recent journal entries')
@click.option('--metrics', is_flag=True, help='Show detailed metrics')
@click.option('--limit', default=20, help='Limit journal entries')
@click.option('--phase', help='Filter journal by phase')
def show(output_json: bool, journal: bool, metrics: bool, limit: int, phase: str):
    """Show current system status"""
    cli = StatusCLI()

    if output_json:
        # Machine-readable JSON output
        click.echo(json.dumps(cli.get_json_status(), indent=2))
    elif journal:
        # Journal entries
        entries = cli.get_journal_entries(limit=limit, phase_filter=phase)
        click.echo(f"📚 Recent Journal Entries ({len(entries)}):")
        click.echo("=" * 50)
        for entry in entries:
            click.echo(f"{entry['ts_utc']} | {entry['phase']}:{entry['step']} | {entry['tool']} | {entry['note']}")
    elif metrics:
        # Detailed metrics
        click.echo(json.dumps(cli.get_metrics_detail(), indent=2))
    else:
        # Human-readable status
        click.echo(cli.get_human_status())


@status.command()
@click.argument('phase')
@click.argument('substep')
@click.option('--tool', help='Tool being used')
@click.option('--result', help='Tool result')
def update(phase: str, substep: str, tool: str, result: str):
    """Update system status"""
    cli = StatusCLI()

    if cli.update_status(phase, substep, tool or "", {}, result or ""):
        click.echo(f"✅ Updated status to {phase}:{substep}")
    else:
        click.echo(f"❌ Failed to update status to {phase}:{substep}")
        sys.exit(1)


@status.command()
def reset():
    """Reset system to idle state"""
    cli = StatusCLI()

    if cli.update_status("Idle", "Ready", "cli.reset", {}, "System reset"):
        click.echo("✅ System reset to Idle:Ready")
    else:
        click.echo("❌ Failed to reset system")
        sys.exit(1)


@status.command()
def export():
    """Export status data to artifacts"""
    cli = StatusCLI()

    # Export current status
    status_file = cli.artifacts_dir / "summary.json"
    with open(status_file, "w") as f:
        json.dump(cli.get_json_status(), f, indent=2, sort_keys=True)

    # Export journal
    journal_file = cli.artifacts_dir / "journal_export.json"
    with open(journal_file, "w") as f:
        f.write(cli.journal.export_json())

    # Export telemetry
    telemetry_file = cli.artifacts_dir / "telemetry_export.json"
    with open(telemetry_file, "w") as f:
        f.write(cli.telemetry.export_json())

    click.echo(f"✅ Status data exported to {cli.artifacts_dir}")


if __name__ == "__main__":
    # Allow running as standalone script
    if len(sys.argv) == 1:
        # Default to show status
        cli = StatusCLI()
        print(cli.get_human_status())
    else:
        # Use Click CLI
        status()