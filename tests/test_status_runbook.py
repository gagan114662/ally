"""
Test suite for status runbook functionality

Tests the deterministic state machine for Ally operational phases.
Validates transitions, state tracking, and audit trail generation.
"""

import json
import pytest
from ally.status.runbook import Runbook, Phase, Substep, get_phase_help, get_substep_help


class TestRunbook:
    """Test runbook state machine functionality"""

    def test_initial_state(self):
        """Test initial state is Idle:Ready"""
        runbook = Runbook(seed=42, deterministic=True)
        assert runbook.phase == "Idle"
        assert runbook.substep == "Ready"
        assert runbook.state == ("Idle", "Ready")

    def test_deterministic_timestamps(self):
        """Test deterministic timestamp generation"""
        runbook = Runbook(seed=42, deterministic=True)

        # Initial status
        status1 = runbook.get_status()

        # Transition
        runbook.enter("Researching", "SpecParsing")
        status2 = runbook.get_status()

        # Another transition
        runbook.enter("Evaluating", "WalkForward")
        status3 = runbook.get_status()

        # Timestamps should be deterministic and sequential
        assert status1["timestamp_utc"] == "2025-01-01T00:00:01.000000Z"
        assert status2["timestamp_utc"] == "2025-01-01T00:00:02.000000Z"
        assert status3["timestamp_utc"] == "2025-01-01T00:00:03.000000Z"

    def test_valid_phase_transitions(self):
        """Test valid phase transitions"""
        runbook = Runbook(seed=42, deterministic=True)

        # Idle → Researching
        assert runbook.enter("Researching", "SpecParsing", "Starting research")
        assert runbook.phase == "Researching"
        assert runbook.substep == "SpecParsing"

        # Researching → Evaluating
        assert runbook.enter("Evaluating", "WalkForward", "Starting evaluation")
        assert runbook.phase == "Evaluating"

        # Evaluating → Routing
        assert runbook.enter("Routing", "SignalGeneration", "Starting routing")
        assert runbook.phase == "Routing"

        # Routing → Executing
        assert runbook.enter("Executing", "OrderPreparation", "Starting execution")
        assert runbook.phase == "Executing"

        # Executing → Idle
        assert runbook.enter("Idle", "Ready", "Execution complete")
        assert runbook.phase == "Idle"

    def test_invalid_phase_transitions(self):
        """Test invalid phase transitions are rejected"""
        runbook = Runbook(seed=42, deterministic=True)

        # Cannot go directly from Idle to Executing
        assert not runbook.enter("Executing", "OrderPreparation")
        assert runbook.phase == "Idle"  # Should remain in original state

        # Start valid sequence
        runbook.enter("Researching", "SpecParsing")

        # Cannot go from Researching to Executing
        assert not runbook.enter("Executing", "OrderPreparation")
        assert runbook.phase == "Researching"  # Should remain in Researching

    def test_substep_transitions_within_phase(self):
        """Test substep transitions within the same phase"""
        runbook = Runbook(seed=42, deterministic=True)

        # Enter Researching phase
        runbook.enter("Researching", "SpecParsing")

        # Should allow substep changes within same phase
        assert runbook.enter("Researching", "DataLoading")
        assert runbook.substep == "DataLoading"

        assert runbook.enter("Researching", "FeatureEngineering")
        assert runbook.substep == "FeatureEngineering"

    def test_error_recovery_transitions(self):
        """Test transitions to and from Error state"""
        runbook = Runbook(seed=42, deterministic=True)

        # Any phase can transition to Error
        runbook.enter("Researching", "SpecParsing")
        assert runbook.enter("Error", "ErrorDetection", "Data loading failed")
        assert runbook.phase == "Error"

        # Error can transition back to Idle
        assert runbook.enter("Idle", "Ready", "Error resolved")
        assert runbook.phase == "Idle"

        # Error can transition to Researching
        runbook.enter("Error", "ErrorRecovery")
        assert runbook.enter("Researching", "SpecParsing", "Retrying research")
        assert runbook.phase == "Researching"

    def test_transition_history(self):
        """Test transition history tracking"""
        runbook = Runbook(seed=42, deterministic=True)

        assert len(runbook.transitions) == 0

        # Make several transitions
        runbook.enter("Researching", "SpecParsing", "Start research")
        runbook.enter("Researching", "DataLoading", "Load data")
        runbook.enter("Evaluating", "WalkForward", "Start evaluation")

        transitions = runbook.transitions
        assert len(transitions) == 3

        # Check first transition
        first = transitions[0]
        assert first["from_phase"] == "Idle"
        assert first["from_substep"] == "Ready"
        assert first["to_phase"] == "Researching"
        assert first["to_substep"] == "SpecParsing"
        assert first["reason"] == "Start research"

        # Check last transition
        last = transitions[-1]
        assert last["from_phase"] == "Researching"
        assert last["to_phase"] == "Evaluating"

    def test_metadata_handling(self):
        """Test metadata storage and retrieval"""
        runbook = Runbook(seed=42, deterministic=True)

        # Set metadata
        runbook.set_metadata("strategy", "momentum")
        runbook.set_metadata("symbols", ["AAPL", "MSFT"])

        assert runbook.get_metadata("strategy") == "momentum"
        assert runbook.get_metadata("symbols") == ["AAPL", "MSFT"]
        assert runbook.get_metadata("nonexistent") is None
        assert runbook.get_metadata("nonexistent", "default") == "default"

        # Metadata should be included in status
        status = runbook.get_status()
        assert status["metadata"]["strategy"] == "momentum"

    def test_transition_with_metadata(self):
        """Test transitions with metadata"""
        runbook = Runbook(seed=42, deterministic=True)

        metadata = {"strategy_id": "strat_001", "symbols": ["AAPL"]}
        runbook.enter("Researching", "SpecParsing", "Starting strategy", metadata)

        transitions = runbook.transitions
        assert len(transitions) == 1
        assert transitions[0]["metadata"] == metadata

    def test_status_summary(self):
        """Test status summary generation"""
        runbook = Runbook(seed=42, deterministic=True)

        runbook.enter("Researching", "DataLoading", "Loading data")
        status = runbook.get_status()

        assert status["phase"] == "Researching"
        assert status["substep"] == "DataLoading"
        assert status["transition_count"] == 1
        assert status["deterministic"] is True
        assert "timestamp_utc" in status
        assert "last_transition" in status

    def test_reset_functionality(self):
        """Test reset to initial state"""
        runbook = Runbook(seed=42, deterministic=True)

        # Make some transitions
        runbook.enter("Researching", "SpecParsing")
        runbook.enter("Evaluating", "WalkForward")
        runbook.set_metadata("test", "value")

        assert runbook.phase != "Idle"
        assert len(runbook.transitions) > 0
        assert runbook.get_metadata("test") == "value"

        # Reset
        runbook.reset()

        assert runbook.phase == "Idle"
        assert runbook.substep == "Ready"
        assert len(runbook.transitions) == 0
        assert runbook.get_metadata("test") is None

    def test_export_import_transitions(self):
        """Test exporting and importing transitions"""
        runbook = Runbook(seed=42, deterministic=True)

        # Create some transitions
        runbook.enter("Researching", "SpecParsing", "Start")
        runbook.enter("Evaluating", "WalkForward", "Evaluate")

        # Export
        exported = runbook.export_transitions()
        assert isinstance(exported, str)

        # Create new runbook and import
        new_runbook = Runbook(seed=42, deterministic=True)
        assert new_runbook.import_transitions(exported)

        # Should have same state and transitions
        assert new_runbook.phase == runbook.phase
        assert new_runbook.substep == runbook.substep
        assert len(new_runbook.transitions) == len(runbook.transitions)

    def test_invalid_phases_and_substeps(self):
        """Test handling of invalid phase/substep names"""
        runbook = Runbook(seed=42, deterministic=True)

        # Invalid phase name
        assert not runbook.enter("InvalidPhase", "Ready")
        assert runbook.phase == "Idle"

        # Invalid substep name
        assert not runbook.enter("Researching", "InvalidSubstep")
        assert runbook.phase == "Idle"

    def test_deterministic_reproducibility(self):
        """Test that operations are reproducible with same seed"""
        # Create two runbooks with same seed
        runbook1 = Runbook(seed=42, deterministic=True)
        runbook2 = Runbook(seed=42, deterministic=True)

        # Perform same operations
        operations = [
            ("Researching", "SpecParsing", "Start research"),
            ("Researching", "DataLoading", "Load data"),
            ("Evaluating", "WalkForward", "Start evaluation")
        ]

        for phase, substep, reason in operations:
            runbook1.enter(phase, substep, reason)
            runbook2.enter(phase, substep, reason)

        # Should have identical states and transitions
        assert runbook1.phase == runbook2.phase
        assert runbook1.substep == runbook2.substep
        assert len(runbook1.transitions) == len(runbook2.transitions)

        # Transitions should have identical timestamps (deterministic)
        for t1, t2 in zip(runbook1.transitions, runbook2.transitions):
            assert t1["timestamp_utc"] == t2["timestamp_utc"]


class TestPhaseHelpers:
    """Test phase and substep helper functions"""

    def test_phase_help(self):
        """Test phase help text retrieval"""
        help_text = get_phase_help()
        assert isinstance(help_text, dict)
        assert "Idle" in help_text
        assert "Researching" in help_text
        assert "Evaluating" in help_text
        assert "Routing" in help_text
        assert "Executing" in help_text
        assert "Error" in help_text

        # Check help text is meaningful
        assert len(help_text["Idle"]) > 10
        assert "ready" in help_text["Idle"].lower()

    def test_substep_help(self):
        """Test substep help text retrieval"""
        help_text = get_substep_help()
        assert isinstance(help_text, dict)
        assert "Ready" in help_text
        assert "SpecParsing" in help_text
        assert "WalkForward" in help_text
        assert "OrderPreparation" in help_text

        # Check help text is meaningful
        assert len(help_text["Ready"]) > 5


class TestRunbookIntegration:
    """Integration tests for runbook with realistic workflows"""

    def test_complete_trading_workflow(self):
        """Test complete trading workflow from research to execution"""
        runbook = Runbook(seed=42, deterministic=True)

        workflow = [
            ("Researching", "SpecParsing", "Parse strategy specification"),
            ("Researching", "DataLoading", "Load historical market data"),
            ("Researching", "FeatureEngineering", "Compute technical indicators"),
            ("Evaluating", "WalkForward", "Run walk-forward validation"),
            ("Evaluating", "CrossValidation", "Perform cross-validation"),
            ("Evaluating", "RobustnessTesting", "Test strategy robustness"),
            ("Routing", "SignalGeneration", "Generate trading signals"),
            ("Routing", "PortfolioOptimization", "Optimize portfolio weights"),
            ("Routing", "RiskAssessment", "Assess portfolio risk"),
            ("Executing", "OrderPreparation", "Prepare order instructions"),
            ("Executing", "OrderRouting", "Route orders to broker"),
            ("Executing", "PositionMonitoring", "Monitor open positions"),
            ("Idle", "Ready", "Trading workflow complete")
        ]

        for i, (phase, substep, reason) in enumerate(workflow):
            assert runbook.enter(phase, substep, reason), f"Failed at step {i}: {phase}:{substep}"

        # Should end in Idle:Ready
        assert runbook.phase == "Idle"
        assert runbook.substep == "Ready"
        assert len(runbook.transitions) == len(workflow)

    def test_error_handling_workflow(self):
        """Test error handling and recovery workflow"""
        runbook = Runbook(seed=42, deterministic=True)

        # Start normal workflow
        runbook.enter("Researching", "DataLoading", "Loading data")

        # Error occurs
        runbook.enter("Error", "ErrorDetection", "Data feed timeout")
        assert runbook.phase == "Error"

        # Error recovery
        runbook.enter("Error", "ErrorRecovery", "Switching to backup data feed")
        runbook.enter("Error", "ErrorReporting", "Logging error details")

        # Return to normal operation
        runbook.enter("Researching", "DataLoading", "Retrying data load")
        assert runbook.phase == "Researching"

        # Continue workflow
        runbook.enter("Evaluating", "WalkForward", "Proceeding with evaluation")
        assert runbook.phase == "Evaluating"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])