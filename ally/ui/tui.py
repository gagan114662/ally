"""
Text User Interface for Ally System

Provides a real-time dashboard showing:
- Current phase and substep
- Recent journal entries
- Operation counters and timers
- Last receipts and audit trail

Designed for deterministic snapshot testing in CI.
"""

import json
import os
from typing import Dict, List, Optional
from ally.status.runbook import Runbook
from ally.status.journal import Journal
from ally.status.telemetry import Telemetry


class AllyTUI:
    """
    Text-based user interface for Ally system status

    Renders a dashboard view of current system state
    with deterministic output for CI testing.
    """

    def __init__(self, journal_path: str = "artifacts/status/journal_ci.jsonl", seed: int = 42):
        """
        Initialize TUI components

        Args:
            journal_path: Path to journal file
            seed: Deterministic seed for reproducible output
        """
        self.runbook = Runbook(seed=seed, deterministic=True)
        self.telemetry = Telemetry(seed=seed)
        self.journal = Journal(journal_path, deterministic_seed=seed)

    def get_dashboard_data(self) -> Dict:
        """
        Get structured dashboard data

        Returns:
            Dictionary with all dashboard information
        """
        # Load recent journal entries
        recent_entries = self.journal.read_entries(limit=10)

        # Get telemetry stats
        telemetry_stats = self.telemetry.get_stats()

        # Get current status
        runbook_status = self.runbook.get_status()

        return {
            "phase": runbook_status["phase"],
            "substep": runbook_status["substep"],
            "timestamp": runbook_status["timestamp_utc"],
            "journal_entries": len(recent_entries),
            "recent_operations": [
                {
                    "tool": entry["tool"],
                    "note": entry["note"],
                    "timestamp": entry["ts_utc"]
                }
                for entry in recent_entries[-5:]  # Last 5 operations
            ],
            "counters": dict(sorted(telemetry_stats["counters"].items())),
            "active_timers": telemetry_stats["active_timers"],
            "last_receipts": telemetry_stats["last_receipts"][:3],  # Last 3 receipts
            "deterministic": runbook_status["deterministic"]
        }

    def render_dashboard(self) -> str:
        """
        Render dashboard as formatted text

        Returns:
            Formatted dashboard string for display
        """
        data = self.get_dashboard_data()

        lines = []
        lines.append("=" * 80)
        lines.append("🤖 ALLY SYSTEM DASHBOARD")
        lines.append("=" * 80)

        # System state
        lines.append(f"📊 STATE: {data['phase']} → {data['substep']}")
        lines.append(f"⏰ TIME: {data['timestamp']}")
        lines.append(f"📚 JOURNAL: {data['journal_entries']} entries")

        # Recent operations
        lines.append("")
        lines.append("🔄 RECENT OPERATIONS:")
        if data["recent_operations"]:
            for i, op in enumerate(data["recent_operations"], 1):
                lines.append(f"  {i}. {op['tool']} - {op['note']}")
        else:
            lines.append("  No recent operations")

        # Counters
        lines.append("")
        lines.append("📈 COUNTERS:")
        if data["counters"]:
            for name, count in list(data["counters"].items())[:10]:  # Show top 10
                lines.append(f"  {name}: {count}")
        else:
            lines.append("  No counters")

        # Active timers
        lines.append("")
        lines.append(f"⏱️  ACTIVE TIMERS: {len(data['active_timers'])}")
        for timer in data["active_timers"][:5]:  # Show first 5
            lines.append(f"  {timer}")

        # Last receipts
        lines.append("")
        lines.append("🧾 LAST RECEIPTS:")
        if data["last_receipts"]:
            for receipt in data["last_receipts"]:
                lines.append(f"  {receipt['tool']}: {receipt['receipt_hash'][:8]}...")
        else:
            lines.append("  No receipts")

        lines.append("")
        lines.append(f"🔧 MODE: {'Deterministic' if data['deterministic'] else 'Live'}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def render_compact(self) -> str:
        """
        Render compact one-line status for testing

        Returns:
            Compact status string
        """
        data = self.get_dashboard_data()
        return (f"PHASE={data['phase']} "
                f"SUBSTEP={data['substep']} "
                f"OPS={data['journal_entries']} "
                f"COUNTERS={sorted(data['counters'].items())}")

    def export_json(self) -> str:
        """
        Export dashboard data as JSON

        Returns:
            JSON string of dashboard data
        """
        return json.dumps(self.get_dashboard_data(), indent=2, sort_keys=True)


def create_sample_dashboard(seed: int = 42) -> AllyTUI:
    """
    Create sample TUI with test data

    Args:
        seed: Deterministic seed

    Returns:
        TUI instance with sample data
    """
    import tempfile
    import os

    # Create temporary journal with sample data
    temp_dir = tempfile.mkdtemp()
    journal_path = os.path.join(temp_dir, "sample_journal.jsonl")

    tui = AllyTUI(journal_path, seed=seed)

    # Add sample data
    tui.runbook.enter("Researching", "DataLoading", "Sample workflow")

    tui.journal.append(
        phase="Researching",
        step="DataLoading",
        tool="data.load",
        params_hash="sample123",
        receipt_hash="receipt456",
        note="Loading sample data"
    )

    tui.telemetry.count("data.loaded", 5)
    tui.telemetry.time("data.loading", 1500)
    tui.telemetry.record_receipt("data.load", "sample123", "receipt456")

    return tui


if __name__ == "__main__":
    # Demo TUI
    tui = create_sample_dashboard()
    print(tui.render_dashboard())
    print("\nCompact:", tui.render_compact())
    print("\nJSON:", tui.export_json())