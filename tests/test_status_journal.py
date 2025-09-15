"""
Test suite for status journal functionality

Tests the append-only JSONL journal for operation logging.
Validates deterministic logging, entry reading, and audit trail integrity.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from ally.status.journal import Journal, create_sample_journal


class TestJournal:
    """Test journal functionality"""

    def setup_method(self):
        """Setup test environment with temporary journal file"""
        self.temp_dir = tempfile.mkdtemp()
        self.journal_path = os.path.join(self.temp_dir, "test_journal.jsonl")

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_journal_initialization(self):
        """Test journal initialization and file creation"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Journal file should be created
        assert os.path.exists(self.journal_path)
        assert journal.journal_path == Path(self.journal_path)
        assert journal.deterministic is True

    def test_append_entry(self):
        """Test appending entries to journal"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        entry = journal.append(
            phase="Researching",
            step="SpecParsing",
            tool="spec.parse",
            params_hash="aa11bb22",
            receipt_hash="cc33dd44",
            note="Parsing strategy specification"
        )

        # Entry should have correct structure
        assert entry["phase"] == "Researching"
        assert entry["step"] == "SpecParsing"
        assert entry["tool"] == "spec.parse"
        assert entry["params_hash"] == "aa11bb22"
        assert entry["receipt_hash"] == "cc33dd44"
        assert entry["note"] == "Parsing strategy specification"
        assert "ts_utc" in entry
        assert "seq" in entry  # Deterministic sequence number

    def test_deterministic_timestamps(self):
        """Test deterministic timestamp generation"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        entry1 = journal.append("Phase1", "Step1", note="First entry")
        entry2 = journal.append("Phase2", "Step2", note="Second entry")
        entry3 = journal.append("Phase3", "Step3", note="Third entry")

        # Timestamps should be deterministic and sequential
        assert entry1["ts_utc"] == "2025-01-01T00:00:01.000000Z"
        assert entry2["ts_utc"] == "2025-01-01T00:00:02.000000Z"
        assert entry3["ts_utc"] == "2025-01-01T00:00:03.000000Z"

        # Sequence numbers should increment
        assert entry1["seq"] == 1
        assert entry2["seq"] == 2
        assert entry3["seq"] == 3

    def test_read_entries(self):
        """Test reading journal entries"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Add several entries
        journal.append("Phase1", "Step1", note="Entry 1")
        journal.append("Phase2", "Step2", note="Entry 2")
        journal.append("Phase1", "Step3", note="Entry 3")

        # Read all entries
        entries = journal.read_entries()
        assert len(entries) == 3
        assert entries[0]["note"] == "Entry 1"
        assert entries[2]["note"] == "Entry 3"

    def test_read_entries_with_limit(self):
        """Test reading entries with limit"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Add many entries
        for i in range(10):
            journal.append("Phase", "Step", note=f"Entry {i}")

        # Read with limit
        entries = journal.read_entries(limit=3)
        assert len(entries) == 3

        # Should get the last 3 entries
        assert entries[0]["note"] == "Entry 7"
        assert entries[1]["note"] == "Entry 8"
        assert entries[2]["note"] == "Entry 9"

    def test_read_entries_with_phase_filter(self):
        """Test reading entries filtered by phase"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Add entries from different phases
        journal.append("Researching", "Step1", note="Research entry 1")
        journal.append("Evaluating", "Step2", note="Eval entry 1")
        journal.append("Researching", "Step3", note="Research entry 2")
        journal.append("Executing", "Step4", note="Exec entry 1")

        # Filter by phase
        research_entries = journal.read_entries(filter_phase="Researching")
        assert len(research_entries) == 2
        assert all(e["phase"] == "Researching" for e in research_entries)

        eval_entries = journal.read_entries(filter_phase="Evaluating")
        assert len(eval_entries) == 1
        assert eval_entries[0]["note"] == "Eval entry 1"

    def test_get_last_entry(self):
        """Test getting the most recent entry"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Empty journal
        assert journal.get_last_entry() is None

        # Add entries
        journal.append("Phase1", "Step1", note="First")
        journal.append("Phase2", "Step2", note="Last")

        last = journal.get_last_entry()
        assert last is not None
        assert last["note"] == "Last"
        assert last["phase"] == "Phase2"

    def test_get_summary(self):
        """Test journal summary generation"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Empty journal
        summary = journal.get_summary()
        assert summary["total_entries"] == 0
        assert summary["phases"] == []
        assert summary["tools"] == []

        # Add entries
        journal.append("Researching", "Step1", tool="tool1", note="Entry 1")
        journal.append("Evaluating", "Step2", tool="tool2", note="Entry 2")
        journal.append("Researching", "Step3", tool="tool1", note="Entry 3")

        summary = journal.get_summary()
        assert summary["total_entries"] == 3
        assert sorted(summary["phases"]) == ["Evaluating", "Researching"]
        assert sorted(summary["tools"]) == ["tool1", "tool2"]
        assert summary["first_entry"]["note"] == "Entry 1"
        assert summary["last_entry"]["note"] == "Entry 3"

    def test_hash_params(self):
        """Test parameter hashing"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # In deterministic mode, should return sequence-based hash
        hash1 = journal.hash_params({"param1": "value1"})
        hash2 = journal.hash_params({"param2": "value2"})

        assert len(hash1) == 16  # 16 hex chars
        assert len(hash2) == 16
        assert hash1 != hash2  # Should be different

        # Check format
        assert all(c in "0123456789abcdef" for c in hash1)
        assert all(c in "0123456789abcdef" for c in hash2)

    def test_hash_receipt(self):
        """Test receipt hashing"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Test different receipt types
        hash1 = journal.hash_receipt("string receipt")
        hash2 = journal.hash_receipt({"result": "success", "value": 42})
        hash3 = journal.hash_receipt(["list", "receipt"])

        assert len(hash1) == 16
        assert len(hash2) == 16
        assert len(hash3) == 16
        assert hash1 != hash2 != hash3

    def test_phase_transition_logging(self):
        """Test phase transition logging helper"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        entry = journal.phase_transition(
            "Idle", "Ready",
            "Researching", "SpecParsing",
            "Starting research workflow"
        )

        assert entry["phase"] == "Researching"
        assert entry["step"] == "SpecParsing"
        assert entry["tool"] == "runbook.transition"
        assert "Transition:" in entry["note"]
        assert entry["metadata"]["from_phase"] == "Idle"
        assert entry["metadata"]["to_phase"] == "Researching"
        assert entry["metadata"]["reason"] == "Starting research workflow"

    def test_tool_invocation_logging(self):
        """Test tool invocation logging helper"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        params = {"strategy": "momentum", "lookback": 20}
        result = {"success": True, "features": 15}

        entry = journal.tool_invocation(
            "Researching", "FeatureEngineering",
            "feature.compute", params, result
        )

        assert entry["phase"] == "Researching"
        assert entry["step"] == "FeatureEngineering"
        assert entry["tool"] == "feature.compute"
        assert len(entry["params_hash"]) == 16
        assert len(entry["receipt_hash"]) == 16
        assert "Invoked feature.compute" in entry["note"]
        assert entry["metadata"]["param_count"] == 2
        assert entry["metadata"]["has_result"] is True

    def test_error_event_logging(self):
        """Test error event logging helper"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        entry = journal.error_event(
            "Evaluating", "WalkForward",
            "DataError", "Missing data for 2023-03-15",
            "Applied forward fill"
        )

        assert entry["phase"] == "Evaluating"
        assert entry["step"] == "WalkForward"
        assert entry["tool"] == "error.handler"
        assert "Error: DataError" in entry["note"]
        assert entry["metadata"]["error_type"] == "DataError"
        assert entry["metadata"]["error_msg"] == "Missing data for 2023-03-15"
        assert entry["metadata"]["recovery_action"] == "Applied forward fill"

    def test_clear_journal(self):
        """Test clearing journal"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Add entries
        journal.append("Phase1", "Step1", note="Entry 1")
        journal.append("Phase2", "Step2", note="Entry 2")

        assert len(journal.read_entries()) == 2

        # Clear
        journal.clear()

        assert len(journal.read_entries()) == 0
        assert not os.path.exists(self.journal_path)

    def test_export_import_json(self):
        """Test exporting and importing journal as JSON"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Add entries
        journal.append("Phase1", "Step1", tool="tool1", note="Entry 1")
        journal.append("Phase2", "Step2", tool="tool2", note="Entry 2")

        # Export
        exported = journal.export_json()
        assert isinstance(exported, str)

        # Import to new journal
        new_journal_path = os.path.join(self.temp_dir, "imported_journal.jsonl")
        new_journal = Journal(new_journal_path, deterministic_seed=42)

        assert new_journal.import_json(exported)

        # Should have same entries
        original_entries = journal.read_entries()
        imported_entries = new_journal.read_entries()

        assert len(imported_entries) == len(original_entries)
        assert imported_entries[0]["note"] == original_entries[0]["note"]
        assert imported_entries[1]["tool"] == original_entries[1]["tool"]

    def test_import_json_append_mode(self):
        """Test importing JSON in append mode"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Add initial entry
        journal.append("Phase1", "Step1", note="Original entry")

        # Create data to import
        import_data = [
            {"phase": "Phase2", "step": "Step2", "tool": "tool2", "note": "Imported entry"}
        ]

        # Import in append mode
        assert journal.import_json(json.dumps(import_data), append=True)

        entries = journal.read_entries()
        assert len(entries) == 2
        assert entries[0]["note"] == "Original entry"
        assert entries[1]["note"] == "Imported entry"

    def test_create_checkpoint(self):
        """Test creating checkpoint copies"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Add entries
        journal.append("Phase1", "Step1", note="Entry 1")
        journal.append("Phase2", "Step2", note="Entry 2")

        # Create checkpoint
        checkpoint_path = os.path.join(self.temp_dir, "checkpoint.jsonl")
        assert journal.create_checkpoint(checkpoint_path)
        assert os.path.exists(checkpoint_path)

        # Checkpoint should have same content
        checkpoint_journal = Journal(checkpoint_path, deterministic_seed=42)
        original_entries = journal.read_entries()
        checkpoint_entries = checkpoint_journal.read_entries()

        assert len(checkpoint_entries) == len(original_entries)

    def test_deterministic_reproducibility(self):
        """Test that journal operations are reproducible with same seed"""
        # Create two journals with same seed
        journal1_path = os.path.join(self.temp_dir, "journal1.jsonl")
        journal2_path = os.path.join(self.temp_dir, "journal2.jsonl")

        journal1 = Journal(journal1_path, deterministic_seed=42)
        journal2 = Journal(journal2_path, deterministic_seed=42)

        # Perform same operations
        operations = [
            ("Researching", "SpecParsing", "spec.parse", {"strategy": "momentum"}),
            ("Researching", "DataLoading", "data.load", {"symbols": ["AAPL"]}),
            ("Evaluating", "WalkForward", "research.wf", {"n_splits": 5})
        ]

        for phase, step, tool, params in operations:
            journal1.tool_invocation(phase, step, tool, params, {"success": True})
            journal2.tool_invocation(phase, step, tool, params, {"success": True})

        # Should have identical entries
        entries1 = journal1.read_entries()
        entries2 = journal2.read_entries()

        assert len(entries1) == len(entries2)
        for e1, e2 in zip(entries1, entries2):
            assert e1["ts_utc"] == e2["ts_utc"]
            assert e1["params_hash"] == e2["params_hash"]
            assert e1["receipt_hash"] == e2["receipt_hash"]


class TestJournalHelpers:
    """Test journal helper functions"""

    def test_create_sample_journal(self):
        """Test sample journal creation"""
        temp_dir = tempfile.mkdtemp()
        try:
            journal_path = os.path.join(temp_dir, "sample.jsonl")
            journal = create_sample_journal(journal_path, deterministic_seed=42)

            # Should have sample entries
            entries = journal.read_entries()
            assert len(entries) > 0

            # Check for expected workflow phases
            phases = [e["phase"] for e in entries]
            assert "Researching" in phases
            assert "Evaluating" in phases
            assert "Idle" in phases

            # Should have tools and transitions
            tools = [e["tool"] for e in entries if e["tool"]]
            assert "spec.parse" in tools
            assert "data.load" in tools
            assert "research.walkforward" in tools

        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestJournalIntegration:
    """Integration tests for journal with realistic workflows"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.journal_path = os.path.join(self.temp_dir, "integration_journal.jsonl")

    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_complete_trading_workflow_logging(self):
        """Test logging complete trading workflow"""
        journal = Journal(self.journal_path, deterministic_seed=42)

        # Research phase
        journal.phase_transition("", "", "Researching", "SpecParsing", "Start research")
        journal.tool_invocation("Researching", "SpecParsing", "spec.parse",
                               {"strategy": "momentum"}, {"parsed": True})

        journal.phase_transition("Researching", "SpecParsing", "Researching", "DataLoading",
                                "Proceed to data loading")
        journal.tool_invocation("Researching", "DataLoading", "data.load",
                               {"symbols": ["AAPL", "MSFT"], "start": "2023-01-01"},
                               {"rows": 1000})

        # Evaluation phase
        journal.phase_transition("Researching", "DataLoading", "Evaluating", "WalkForward",
                                "Start evaluation")
        journal.tool_invocation("Evaluating", "WalkForward", "research.walkforward",
                               {"n_splits": 5, "test_size": 0.2}, {"sharpe": 1.2})

        # Error handling
        journal.error_event("Evaluating", "WalkForward", "DataError",
                          "Missing data point", "Applied interpolation")

        # Complete workflow
        journal.phase_transition("Evaluating", "WalkForward", "Idle", "Ready",
                                "Workflow complete")

        # Verify complete log
        entries = journal.read_entries()
        assert len(entries) >= 6

        summary = journal.get_summary()
        assert summary["total_entries"] >= 6
        assert "Researching" in summary["phases"]
        assert "Evaluating" in summary["phases"]
        assert "Idle" in summary["phases"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])