"""
Chat Controller - Natural Language Interface for Ally

Processes natural language prompts and routes them to safe, read-only operations.
All commands generate receipts for audit trail.
Designed to be offline-safe in CI mode.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from ally.status.runbook import Runbook
from ally.status.journal import Journal
from ally.status.telemetry import Telemetry


@dataclass
class ChatResponse:
    """Response from chat controller"""
    ok: bool
    message: str
    data: Dict
    receipt: Dict


def tool_receipt(tool_name: str, params: Dict) -> tuple[str, str]:
    """
    Generate deterministic tool receipt hashes

    Args:
        tool_name: Name of the tool
        params: Parameters passed to tool

    Returns:
        Tuple of (params_hash, receipt_hash)
    """
    # Create deterministic hashes based on tool name and params
    params_json = json.dumps(params, sort_keys=True)
    params_hash = hashlib.sha256((tool_name + params_json).encode()).hexdigest()[:16]

    # Receipt hash includes tool name for uniqueness
    receipt_content = f"{tool_name}:{params_json}:success"
    receipt_hash = hashlib.sha256(receipt_content.encode()).hexdigest()[:16]

    return params_hash, receipt_hash


class ChatController:
    """
    Natural language chat controller for Ally system

    Routes user prompts to safe, read-only operations.
    All commands generate audit receipts.
    """

    def __init__(self, journal_path: str = "artifacts/status/journal_ci.jsonl", seed: int = 42):
        """
        Initialize chat controller

        Args:
            journal_path: Path to journal file
            seed: Deterministic seed for reproducible behavior
        """
        self.runbook = Runbook(seed=seed, deterministic=True)
        self.telemetry = Telemetry(seed=seed)
        self.journal = Journal(journal_path, deterministic_seed=seed)

        # Load existing state if available
        self._load_state()

    def _load_state(self) -> None:
        """Load system state from journal and telemetry"""
        # Get last journal entry to understand current state
        last_entry = self.journal.get_last_entry()
        if last_entry:
            # Try to set runbook state based on last journal entry
            try:
                self.runbook.enter(
                    last_entry["phase"],
                    last_entry["step"],
                    "Loaded from journal"
                )
            except:
                # Fallback to default state if transition invalid
                self.runbook.enter("Idle", "Ready", "Fallback to idle")

    def _recent_ops(self, n: int = 5) -> List[Dict]:
        """
        Get recent operations from journal

        Args:
            n: Number of recent operations to return

        Returns:
            List of recent journal entries
        """
        entries = self.journal.read_entries(limit=n)
        return [
            {
                "timestamp": entry["ts_utc"],
                "phase": entry["phase"],
                "step": entry["step"],
                "tool": entry["tool"],
                "note": entry["note"],
                "params_hash": entry.get("params_hash", ""),
                "receipt_hash": entry.get("receipt_hash", "")
            }
            for entry in entries
        ]

    def handle(self, prompt: str) -> ChatResponse:
        """
        Handle natural language prompt and return response

        Args:
            prompt: User's natural language prompt

        Returns:
            ChatResponse with result and receipt
        """
        p = prompt.strip().lower()

        # Status queries
        if p in ("status", "show status", "where are you at?", "where are you at"):
            summary = {
                "phase": self.runbook.phase,
                "substep": self.runbook.substep,
                "timestamp": self.runbook.get_status()["timestamp_utc"],
                "counters": dict(self.telemetry.counters),
                "timers": {k: len(v) for k, v in self.telemetry.timers.items()},
                "recent_ops": self._recent_ops(5),
                "journal_entries": len(self.journal.read_entries())
            }
            params = {"cmd": "status"}
            ph, rh = tool_receipt("chat.status", params)

            # Log this chat interaction
            self.journal.append(
                phase=self.runbook.phase,
                step=self.runbook.substep,
                tool="chat.status",
                params_hash=ph,
                receipt_hash=rh,
                note="Chat status query"
            )

            return ChatResponse(
                ok=True,
                message="status",
                data=summary,
                receipt={"tool": "chat.status", "params_hash": ph, "receipt_hash": rh}
            )

        # Recent operations queries
        if p.startswith("last ") and ("operations" in p or "ops" in p):
            try:
                # Extract number from prompt like "last 5 operations"
                words = p.split()
                n = int(words[1]) if len(words) > 1 and words[1].isdigit() else 5
                n = min(n, 50)  # Cap at 50 for safety
            except (ValueError, IndexError):
                n = 5

            ops = self._recent_ops(n)
            params = {"cmd": "last_ops", "n": n}
            ph, rh = tool_receipt("chat.last_ops", params)

            # Log this interaction
            self.journal.append(
                phase=self.runbook.phase,
                step=self.runbook.substep,
                tool="chat.last_ops",
                params_hash=ph,
                receipt_hash=rh,
                note=f"Chat last {n} operations query"
            )

            return ChatResponse(
                ok=True,
                message="last_ops",
                data={"ops": ops, "count": len(ops)},
                receipt={"tool": "chat.last_ops", "params_hash": ph, "receipt_hash": rh}
            )

        # Counters query
        if p in ("counters", "show counters", "metrics"):
            counters = dict(self.telemetry.counters)
            params = {"cmd": "counters"}
            ph, rh = tool_receipt("chat.counters", params)

            self.journal.append(
                phase=self.runbook.phase,
                step=self.runbook.substep,
                tool="chat.counters",
                params_hash=ph,
                receipt_hash=rh,
                note="Chat counters query"
            )

            return ChatResponse(
                ok=True,
                message="counters",
                data={"counters": counters, "total": len(counters)},
                receipt={"tool": "chat.counters", "params_hash": ph, "receipt_hash": rh}
            )

        # Timers query
        if p in ("timers", "show timers", "performance"):
            timer_stats = self.telemetry.get_stats()["timer_stats"]
            params = {"cmd": "timers"}
            ph, rh = tool_receipt("chat.timers", params)

            self.journal.append(
                phase=self.runbook.phase,
                step=self.runbook.substep,
                tool="chat.timers",
                params_hash=ph,
                receipt_hash=rh,
                note="Chat timers query"
            )

            return ChatResponse(
                ok=True,
                message="timers",
                data={"timer_stats": timer_stats},
                receipt={"tool": "chat.timers", "params_hash": ph, "receipt_hash": rh}
            )

        # Receipts query
        if p in ("receipts", "show receipts", "audit"):
            receipts = self.telemetry.last_receipts
            params = {"cmd": "receipts"}
            ph, rh = tool_receipt("chat.receipts", params)

            self.journal.append(
                phase=self.runbook.phase,
                step=self.runbook.substep,
                tool="chat.receipts",
                params_hash=ph,
                receipt_hash=rh,
                note="Chat receipts query"
            )

            return ChatResponse(
                ok=True,
                message="receipts",
                data={"receipts": receipts, "count": len(receipts)},
                receipt={"tool": "chat.receipts", "params_hash": ph, "receipt_hash": rh}
            )

        # Help/default response
        params = {"cmd": "help", "prompt": prompt}
        ph, rh = tool_receipt("chat.help", params)

        self.journal.append(
            phase=self.runbook.phase,
            step=self.runbook.substep,
            tool="chat.help",
            params_hash=ph,
            receipt_hash=rh,
            note="Chat help query"
        )

        help_text = ("Available commands:\n"
                    "- 'show status' or 'where are you at?' - System status\n"
                    "- 'last N operations' - Recent operations\n"
                    "- 'counters' - Operation counters\n"
                    "- 'timers' - Performance metrics\n"
                    "- 'receipts' - Audit receipts")

        return ChatResponse(
            ok=False,
            message="help",
            data={"help": help_text, "unknown_prompt": prompt},
            receipt={"tool": "chat.help", "params_hash": ph, "receipt_hash": rh}
        )

    def get_available_commands(self) -> List[str]:
        """Get list of available chat commands"""
        return [
            "show status",
            "where are you at?",
            "last N operations",
            "counters",
            "timers",
            "receipts"
        ]


if __name__ == "__main__":
    # Demo chat controller
    import tempfile
    import os

    # Create test controller
    temp_dir = tempfile.mkdtemp()
    journal_path = os.path.join(temp_dir, "test_journal.jsonl")

    controller = ChatController(journal_path, seed=42)

    # Test commands
    test_prompts = [
        "show status",
        "last 3 operations",
        "counters",
        "help me"
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = controller.handle(prompt)
        print(f"Response: {response.message}")
        print(f"Receipt: {response.receipt['receipt_hash'][:8]}...")
        print(f"OK: {response.ok}")
        if response.data:
            print(f"Data keys: {list(response.data.keys())}")