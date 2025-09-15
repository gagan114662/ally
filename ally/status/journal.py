"""
Status Journal - Append-only JSONL journal for Ally operations

Provides deterministic, audit-trail logging of all operations:
- Phase transitions and substep changes
- Tool invocations with parameter hashes
- Receipt generation and verification
- Error conditions and recovery actions

Format: {ts_utc, phase, step, tool, params_hash, receipt_hash, note}
All entries are append-only and immutable for compliance.
"""

import json
import os
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path


class Journal:
    """
    Append-only JSONL journal for operation logging

    Maintains deterministic audit trail of all Ally operations.
    In deterministic mode, uses fixed timestamps and seeded hashes.
    """

    def __init__(self, journal_path: str, deterministic_seed: int = None):
        """
        Initialize journal

        Args:
            journal_path: Path to JSONL journal file
            deterministic_seed: Seed for deterministic mode
        """
        self.journal_path = Path(journal_path)
        self.deterministic = os.getenv("ALLY_LIVE", "1") == "0"
        self._counter = 0

        if deterministic_seed is not None and self.deterministic:
            self._seed = deterministic_seed
            self._counter = 0

        # Ensure parent directory exists
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize file if it doesn't exist
        if not self.journal_path.exists():
            self.journal_path.touch()

    def append(self, phase: str, step: str, tool: str = "",
               params_hash: str = "", receipt_hash: str = "",
               note: str = "", metadata: Dict = None) -> Dict:
        """
        Append entry to journal

        Args:
            phase: Current operational phase
            step: Current substep
            tool: Tool/function being invoked
            params_hash: Hash of tool parameters
            receipt_hash: Hash of operation receipt
            note: Human-readable note
            metadata: Additional structured data

        Returns:
            The journal entry that was written
        """
        entry = {
            "ts_utc": self._get_timestamp(),
            "phase": phase,
            "step": step,
            "tool": tool,
            "params_hash": params_hash,
            "receipt_hash": receipt_hash,
            "note": note,
            "metadata": metadata or {}
        }

        # Add deterministic sequence number in test mode
        if self.deterministic:
            entry["seq"] = self._counter

        # Append to file
        with open(self.journal_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

        return entry

    def read_entries(self, limit: int = None, filter_phase: str = None) -> List[Dict]:
        """
        Read journal entries

        Args:
            limit: Maximum number of entries to return
            filter_phase: Only return entries from this phase

        Returns:
            List of journal entries
        """
        entries = []

        if not self.journal_path.exists():
            return entries

        with open(self.journal_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    if filter_phase and entry.get("phase") != filter_phase:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

        # Apply limit if specified
        if limit:
            entries = entries[-limit:]

        return entries

    def get_last_entry(self) -> Optional[Dict]:
        """Get the most recent journal entry"""
        entries = self.read_entries(limit=1)
        return entries[0] if entries else None

    def get_summary(self) -> Dict:
        """
        Get journal summary statistics

        Returns:
            Summary with entry counts, phases, tools used
        """
        entries = self.read_entries()

        if not entries:
            return {
                "total_entries": 0,
                "phases": [],
                "tools": [],
                "first_entry": None,
                "last_entry": None
            }

        phases = list(set(entry["phase"] for entry in entries))
        tools = list(set(entry["tool"] for entry in entries if entry["tool"]))

        return {
            "total_entries": len(entries),
            "phases": sorted(phases),
            "tools": sorted(tools),
            "first_entry": entries[0],
            "last_entry": entries[-1]
        }

    def hash_params(self, params: Dict) -> str:
        """
        Generate deterministic hash of parameters

        Args:
            params: Parameter dictionary

        Returns:
            Hex hash string (first 16 characters)
        """
        if self.deterministic:
            # Use sequence counter for deterministic hashes
            return f"{self._counter:016x}"[:16]
        else:
            # Real hash of parameters
            params_json = json.dumps(params, sort_keys=True)
            return hashlib.sha256(params_json.encode()).hexdigest()[:16]

    def hash_receipt(self, receipt: Any) -> str:
        """
        Generate deterministic hash of receipt

        Args:
            receipt: Receipt data (any JSON-serializable object)

        Returns:
            Hex hash string (first 16 characters)
        """
        if self.deterministic:
            # Use sequence counter for deterministic hashes
            return f"{self._counter + 1000:016x}"[:16]
        else:
            # Real hash of receipt
            if isinstance(receipt, str):
                content = receipt
            else:
                content = json.dumps(receipt, sort_keys=True)
            return hashlib.sha256(content.encode()).hexdigest()[:16]

    def phase_transition(self, from_phase: str, from_step: str,
                        to_phase: str, to_step: str, reason: str = "") -> Dict:
        """
        Log phase transition

        Args:
            from_phase: Source phase
            from_step: Source substep
            to_phase: Target phase
            to_step: Target substep
            reason: Reason for transition

        Returns:
            Journal entry for the transition
        """
        return self.append(
            phase=to_phase,
            step=to_step,
            tool="runbook.transition",
            note=f"Transition: {from_phase}:{from_step} → {to_phase}:{to_step}",
            metadata={
                "from_phase": from_phase,
                "from_step": from_step,
                "to_phase": to_phase,
                "to_step": to_step,
                "reason": reason
            }
        )

    def tool_invocation(self, phase: str, step: str, tool: str,
                       params: Dict, result: Any = None) -> Dict:
        """
        Log tool invocation

        Args:
            phase: Current phase
            step: Current substep
            tool: Tool name
            params: Tool parameters
            result: Tool result/receipt

        Returns:
            Journal entry for the invocation
        """
        params_hash = self.hash_params(params)
        receipt_hash = self.hash_receipt(result) if result is not None else ""

        return self.append(
            phase=phase,
            step=step,
            tool=tool,
            params_hash=params_hash,
            receipt_hash=receipt_hash,
            note=f"Invoked {tool}",
            metadata={
                "param_count": len(params),
                "has_result": result is not None
            }
        )

    def error_event(self, phase: str, step: str, error_type: str,
                   error_msg: str, recovery_action: str = "") -> Dict:
        """
        Log error event

        Args:
            phase: Phase where error occurred
            step: Substep where error occurred
            error_type: Type/category of error
            error_msg: Error message
            recovery_action: Action taken to recover

        Returns:
            Journal entry for the error
        """
        return self.append(
            phase=phase,
            step=step,
            tool="error.handler",
            note=f"Error: {error_type}",
            metadata={
                "error_type": error_type,
                "error_msg": error_msg,
                "recovery_action": recovery_action
            }
        )

    def clear(self) -> None:
        """Clear journal (for testing only)"""
        if self.journal_path.exists():
            self.journal_path.unlink()
        self._counter = 0

    def _get_timestamp(self) -> str:
        """Get current timestamp (deterministic in test mode)"""
        if self.deterministic:
            self._counter += 1
            # Fixed timestamp with incrementing counter for determinism
            return f"2025-01-01T00:00:{self._counter:02d}.000000Z"
        else:
            return datetime.now(timezone.utc).isoformat()

    def export_json(self) -> str:
        """Export all entries as JSON array"""
        entries = self.read_entries()
        return json.dumps(entries, indent=2, sort_keys=True)

    def import_json(self, json_data: str, append: bool = False) -> bool:
        """
        Import entries from JSON

        Args:
            json_data: JSON string with entries array
            append: If True, append to existing entries; if False, replace

        Returns:
            True if successful
        """
        try:
            entries = json.loads(json_data)
            if not isinstance(entries, list):
                return False

            if not append:
                self.clear()

            # Write entries
            for entry in entries:
                with open(self.journal_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, sort_keys=True) + "\n")

            return True
        except (json.JSONDecodeError, IOError):
            return False

    def create_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Create checkpoint copy of journal

        Args:
            checkpoint_path: Path for checkpoint file

        Returns:
            True if successful
        """
        try:
            if self.journal_path.exists():
                import shutil
                shutil.copy2(self.journal_path, checkpoint_path)
            return True
        except IOError:
            return False


def create_sample_journal(path: str, deterministic_seed: int = 42) -> Journal:
    """
    Create sample journal for testing/demo

    Args:
        path: Journal file path
        deterministic_seed: Seed for deterministic entries

    Returns:
        Journal instance with sample data
    """
    journal = Journal(path, deterministic_seed=deterministic_seed)
    journal.clear()  # Start fresh

    # Sample workflow entries
    journal.phase_transition("", "", "Researching", "SpecParsing", "Starting research")
    journal.tool_invocation("Researching", "SpecParsing", "spec.parse",
                           {"strategy": "momentum", "lookback": 20}, {"success": True})

    journal.phase_transition("Researching", "SpecParsing", "Researching", "DataLoading",
                            "Proceeding to data loading")
    journal.tool_invocation("Researching", "DataLoading", "data.load",
                           {"symbols": ["AAPL", "MSFT"], "start": "2023-01-01"},
                           {"rows": 1000, "columns": 10})

    journal.phase_transition("Researching", "DataLoading", "Evaluating", "WalkForward",
                            "Starting evaluation")
    journal.tool_invocation("Evaluating", "WalkForward", "research.walkforward",
                           {"n_splits": 5, "test_size": 0.2}, {"sharpe": 1.2, "returns": 0.15})

    journal.error_event("Evaluating", "WalkForward", "DataError",
                       "Missing data for 2023-03-15", "Forward fill applied")

    journal.phase_transition("Evaluating", "WalkForward", "Idle", "Ready",
                            "Evaluation complete")

    return journal


if __name__ == "__main__":
    # Demo usage
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        journal_path = os.path.join(tmpdir, "test_journal.jsonl")

        # Create sample journal
        journal = create_sample_journal(journal_path)

        print("Journal Summary:")
        print(json.dumps(journal.get_summary(), indent=2))

        print("\nLast Entry:")
        print(json.dumps(journal.get_last_entry(), indent=2))

        print("\nAll Entries:")
        for entry in journal.read_entries():
            print(f"  {entry['ts_utc']} | {entry['phase']}:{entry['step']} | {entry['tool']} | {entry['note']}")