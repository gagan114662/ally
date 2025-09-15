"""
Status Runbook - Lifecycle states & transitions for Ally system

Defines deterministic state machine for Ally's operational phases:
- Idle: System ready, awaiting commands
- Researching: Processing specs, running analysis
- Evaluating: Running walk-forward, cross-validation
- Routing: Determining execution path
- Executing: Placing orders, managing positions
- Error: Handling failures and recovery

All transitions are deterministic and receipt-backed for audit trails.
"""

import json
import random
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


class Phase(Enum):
    """Main operational phases"""
    IDLE = "Idle"
    RESEARCHING = "Researching"
    EVALUATING = "Evaluating"
    ROUTING = "Routing"
    EXECUTING = "Executing"
    ERROR = "Error"


class Substep(Enum):
    """Detailed substeps within each phase"""
    # Idle substeps
    READY = "Ready"
    WAITING = "Waiting"

    # Research substeps
    SPEC_PARSING = "SpecParsing"
    DATA_LOADING = "DataLoading"
    FEATURE_ENGINEERING = "FeatureEngineering"

    # Evaluation substeps
    WALK_FORWARD = "WalkForward"
    CROSS_VALIDATION = "CrossValidation"
    ROBUSTNESS_TESTING = "RobustnessTesting"

    # Routing substeps
    SIGNAL_GENERATION = "SignalGeneration"
    PORTFOLIO_OPTIMIZATION = "PortfolioOptimization"
    RISK_ASSESSMENT = "RiskAssessment"

    # Execution substeps
    ORDER_PREPARATION = "OrderPreparation"
    ORDER_ROUTING = "OrderRouting"
    POSITION_MONITORING = "PositionMonitoring"

    # Error substeps
    ERROR_DETECTION = "ErrorDetection"
    ERROR_RECOVERY = "ErrorRecovery"
    ERROR_REPORTING = "ErrorReporting"


@dataclass
class StateTransition:
    """Records a state transition with metadata"""
    from_phase: str
    from_substep: str
    to_phase: str
    to_substep: str
    timestamp_utc: str
    reason: str
    metadata: Dict


class Runbook:
    """
    Deterministic state machine for Ally operational phases

    Manages transitions between phases and substeps with full audit trail.
    In deterministic mode (ALLY_LIVE=0), uses fixed timestamps and seeded randomness.
    """

    def __init__(self, seed: int = None, deterministic: bool = None):
        """
        Initialize runbook state machine

        Args:
            seed: Random seed for deterministic operation
            deterministic: Force deterministic mode (overrides ALLY_LIVE env)
        """
        import os

        self.deterministic = deterministic if deterministic is not None else os.getenv("ALLY_LIVE", "1") == "0"

        if self.deterministic and seed is not None:
            random.seed(seed)
            self._counter = 0

        # Initial state
        self._phase = Phase.IDLE
        self._substep = Substep.READY
        self._transitions: List[StateTransition] = []
        self._metadata: Dict = {}

        # Valid transitions mapping
        self._valid_transitions = self._build_transition_matrix()

    @property
    def phase(self) -> str:
        """Current phase"""
        return self._phase.value

    @property
    def substep(self) -> str:
        """Current substep"""
        return self._substep.value

    @property
    def state(self) -> Tuple[str, str]:
        """Current (phase, substep) tuple"""
        return (self.phase, self.substep)

    @property
    def transitions(self) -> List[Dict]:
        """History of all transitions"""
        return [asdict(t) for t in self._transitions]

    def enter(self, phase: str, substep: str, reason: str = "", metadata: Dict = None) -> bool:
        """
        Transition to new phase/substep

        Args:
            phase: Target phase name
            substep: Target substep name
            reason: Human-readable reason for transition
            metadata: Additional transition metadata

        Returns:
            True if transition successful, False if invalid
        """
        try:
            new_phase = Phase(phase)
            new_substep = Substep(substep)
        except ValueError:
            return False

        # Validate transition
        if not self._is_valid_transition(self._phase, new_phase, self._substep, new_substep):
            return False

        # Record transition
        transition = StateTransition(
            from_phase=self._phase.value,
            from_substep=self._substep.value,
            to_phase=new_phase.value,
            to_substep=new_substep.value,
            timestamp_utc=self._get_timestamp(),
            reason=reason or f"Transition to {phase}:{substep}",
            metadata=metadata or {}
        )

        self._transitions.append(transition)

        # Update state
        self._phase = new_phase
        self._substep = new_substep

        return True

    def get_status(self) -> Dict:
        """
        Get current status summary

        Returns:
            Dictionary with current state and recent transitions
        """
        return {
            "phase": self.phase,
            "substep": self.substep,
            "timestamp_utc": self._get_timestamp(),
            "transition_count": len(self._transitions),
            "last_transition": asdict(self._transitions[-1]) if self._transitions else None,
            "deterministic": self.deterministic,
            "metadata": self._metadata.copy()
        }

    def set_metadata(self, key: str, value) -> None:
        """Set metadata for current state"""
        self._metadata[key] = value

    def get_metadata(self, key: str, default=None):
        """Get metadata value"""
        return self._metadata.get(key, default)

    def _get_timestamp(self) -> str:
        """Get current timestamp (deterministic in test mode)"""
        if self.deterministic:
            self._counter += 1
            # Fixed timestamp with incrementing counter for determinism
            return f"2025-01-01T00:00:{self._counter:02d}.000000Z"
        else:
            return datetime.now(timezone.utc).isoformat()

    def _build_transition_matrix(self) -> Dict:
        """Build valid state transition matrix"""
        return {
            # From IDLE
            (Phase.IDLE, Phase.RESEARCHING): True,
            (Phase.IDLE, Phase.ERROR): True,

            # From RESEARCHING
            (Phase.RESEARCHING, Phase.EVALUATING): True,
            (Phase.RESEARCHING, Phase.IDLE): True,
            (Phase.RESEARCHING, Phase.ERROR): True,

            # From EVALUATING
            (Phase.EVALUATING, Phase.ROUTING): True,
            (Phase.EVALUATING, Phase.RESEARCHING): True,
            (Phase.EVALUATING, Phase.IDLE): True,
            (Phase.EVALUATING, Phase.ERROR): True,

            # From ROUTING
            (Phase.ROUTING, Phase.EXECUTING): True,
            (Phase.ROUTING, Phase.EVALUATING): True,
            (Phase.ROUTING, Phase.IDLE): True,
            (Phase.ROUTING, Phase.ERROR): True,

            # From EXECUTING
            (Phase.EXECUTING, Phase.IDLE): True,
            (Phase.EXECUTING, Phase.ROUTING): True,
            (Phase.EXECUTING, Phase.ERROR): True,

            # From ERROR
            (Phase.ERROR, Phase.IDLE): True,
            (Phase.ERROR, Phase.RESEARCHING): True,
        }

    def _is_valid_transition(self, from_phase: Phase, to_phase: Phase,
                           from_substep: Substep, to_substep: Substep) -> bool:
        """Check if transition is valid"""
        # Allow transitions within same phase
        if from_phase == to_phase:
            return True

        # Check phase transition matrix
        return self._valid_transitions.get((from_phase, to_phase), False)

    def reset(self) -> None:
        """Reset to initial state"""
        self._phase = Phase.IDLE
        self._substep = Substep.READY
        self._transitions.clear()
        self._metadata.clear()
        if self.deterministic:
            self._counter = 0

    def export_transitions(self) -> str:
        """Export transitions as JSON string"""
        return json.dumps(self.transitions, indent=2, sort_keys=True)

    def import_transitions(self, json_data: str) -> bool:
        """Import transitions from JSON string"""
        try:
            transitions_data = json.loads(json_data)
            self._transitions = [
                StateTransition(**t) for t in transitions_data
            ]
            # Set state to last transition's target
            if self._transitions:
                last = self._transitions[-1]
                self._phase = Phase(last.to_phase)
                self._substep = Substep(last.to_substep)
            return True
        except (json.JSONDecodeError, ValueError, TypeError):
            return False


def get_phase_help() -> Dict[str, str]:
    """Get help text for all phases"""
    return {
        "Idle": "System ready and waiting for commands",
        "Researching": "Processing strategy specs and loading data",
        "Evaluating": "Running backtests and validation",
        "Routing": "Generating signals and optimizing portfolio",
        "Executing": "Placing orders and monitoring positions",
        "Error": "Handling errors and recovery procedures"
    }


def get_substep_help() -> Dict[str, str]:
    """Get help text for all substeps"""
    return {
        "Ready": "System initialized and ready",
        "Waiting": "Idle, waiting for next command",
        "SpecParsing": "Parsing strategy specifications",
        "DataLoading": "Loading market data and features",
        "FeatureEngineering": "Computing technical indicators",
        "WalkForward": "Running walk-forward validation",
        "CrossValidation": "Performing cross-validation",
        "RobustnessTesting": "Testing strategy robustness",
        "SignalGeneration": "Generating trading signals",
        "PortfolioOptimization": "Optimizing portfolio weights",
        "RiskAssessment": "Assessing portfolio risk",
        "OrderPreparation": "Preparing order instructions",
        "OrderRouting": "Routing orders to execution",
        "PositionMonitoring": "Monitoring open positions",
        "ErrorDetection": "Detecting system errors",
        "ErrorRecovery": "Recovering from errors",
        "ErrorReporting": "Reporting error details"
    }


if __name__ == "__main__":
    # Demo usage
    runbook = Runbook(seed=42, deterministic=True)

    print("Initial state:", runbook.state)

    # Simulate typical workflow
    runbook.enter("Researching", "SpecParsing", "Starting new strategy research")
    runbook.enter("Researching", "DataLoading", "Loading historical data")
    runbook.enter("Evaluating", "WalkForward", "Running walk-forward validation")
    runbook.enter("Routing", "SignalGeneration", "Generating trading signals")
    runbook.enter("Executing", "OrderPreparation", "Preparing orders")
    runbook.enter("Idle", "Ready", "Workflow complete")

    print("Final state:", runbook.state)
    print("Transitions:", len(runbook.transitions))
    print("Status:", json.dumps(runbook.get_status(), indent=2))