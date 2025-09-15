"""
Status Telemetry - Deterministic timers, counters, and metrics collection

Provides deterministic telemetry for Ally operations:
- Counters: Operation counts, error tallies, success rates
- Timers: Execution duration tracking with deterministic fake clocks
- Last receipts: Most recent operation receipts and hashes
- Error tracking: Failure patterns and recovery metrics

All metrics are deterministic in CI mode (no system clock drift).
"""

import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Deque
from dataclasses import dataclass, asdict, field


@dataclass
class TimerEntry:
    """Individual timer measurement"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    def stop(self, fake_duration: int = None) -> int:
        """Stop timer and return duration in milliseconds"""
        if fake_duration is not None:
            # Deterministic mode
            self.duration_ms = fake_duration
            self.end_time = self.start_time + (fake_duration / 1000.0)
        else:
            # Real timing
            self.end_time = time.time()
            self.duration_ms = int((self.end_time - self.start_time) * 1000)
        return self.duration_ms


@dataclass
class CounterEntry:
    """Individual counter measurement"""
    name: str
    value: int
    timestamp: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ReceiptEntry:
    """Recent receipt tracking"""
    tool: str
    params_hash: str
    receipt_hash: str
    timestamp: str
    metadata: Dict = field(default_factory=dict)


class Telemetry:
    """
    Deterministic telemetry collection system

    Tracks performance metrics, operation counts, and receipts.
    Uses fake clocks and seeded values in deterministic mode.
    """

    def __init__(self, seed: int = None, max_history: int = 1000):
        """
        Initialize telemetry system

        Args:
            seed: Random seed for deterministic operation
            max_history: Maximum number of historical entries to keep
        """
        self.deterministic = os.getenv("ALLY_LIVE", "1") == "0"
        self.max_history = max_history

        # Initialize counters and timers
        self._counters: Dict[str, int] = defaultdict(int)
        self._counter_history: Deque[CounterEntry] = deque(maxlen=max_history)

        self._active_timers: Dict[str, TimerEntry] = {}
        self._timer_history: Deque[TimerEntry] = deque(maxlen=max_history)

        self._receipts: Deque[ReceiptEntry] = deque(maxlen=max_history)

        # Deterministic state
        if self.deterministic and seed is not None:
            self._seed = seed
            self._fake_time = 1000000  # Start at 1 second in fake time
            self._counter = 0

    @property
    def counters(self) -> Dict[str, int]:
        """Current counter values"""
        return dict(self._counters)

    @property
    def timers(self) -> Dict[str, List[int]]:
        """Timer statistics grouped by name"""
        timer_stats = defaultdict(list)
        for entry in self._timer_history:
            if entry.duration_ms is not None:
                timer_stats[entry.name].append(entry.duration_ms)
        return dict(timer_stats)

    @property
    def last_receipts(self) -> List[Dict]:
        """Recent receipts (most recent first)"""
        return [asdict(receipt) for receipt in reversed(self._receipts)]

    def count(self, name: str, increment: int = 1, metadata: Dict = None) -> int:
        """
        Increment counter

        Args:
            name: Counter name
            increment: Amount to increment
            metadata: Additional metadata

        Returns:
            New counter value
        """
        self._counters[name] += increment

        # Record counter entry
        entry = CounterEntry(
            name=name,
            value=self._counters[name],
            timestamp=self._get_timestamp(),
            metadata=metadata or {}
        )
        self._counter_history.append(entry)

        return self._counters[name]

    def start_timer(self, name: str, metadata: Dict = None) -> TimerEntry:
        """
        Start named timer

        Args:
            name: Timer name
            metadata: Additional metadata

        Returns:
            Timer entry object
        """
        start_time = self._get_fake_time() if self.deterministic else time.time()

        timer = TimerEntry(
            name=name,
            start_time=start_time,
            metadata=metadata or {}
        )

        self._active_timers[name] = timer
        return timer

    def stop_timer(self, name: str, fake_duration: int = None) -> Optional[int]:
        """
        Stop named timer

        Args:
            name: Timer name
            fake_duration: Fixed duration for deterministic mode

        Returns:
            Duration in milliseconds, None if timer not found
        """
        if name not in self._active_timers:
            return None

        timer = self._active_timers.pop(name)

        if self.deterministic and fake_duration is None:
            # Generate deterministic duration
            fake_duration = (hash(name + str(self._counter)) % 5000) + 100
            self._counter += 1

        duration = timer.stop(fake_duration)
        self._timer_history.append(timer)

        return duration

    def time(self, name: str, duration_ms: int, metadata: Dict = None) -> None:
        """
        Record timer measurement directly

        Args:
            name: Timer name
            duration_ms: Duration in milliseconds
            metadata: Additional metadata
        """
        start_time = self._get_fake_time() if self.deterministic else time.time()

        timer = TimerEntry(
            name=name,
            start_time=start_time,
            end_time=start_time + (duration_ms / 1000.0),
            duration_ms=duration_ms,
            metadata=metadata or {}
        )

        self._timer_history.append(timer)

    def record_receipt(self, tool: str, params_hash: str, receipt_hash: str,
                      metadata: Dict = None) -> None:
        """
        Record operation receipt

        Args:
            tool: Tool/function name
            params_hash: Hash of parameters
            receipt_hash: Hash of result/receipt
            metadata: Additional metadata
        """
        receipt = ReceiptEntry(
            tool=tool,
            params_hash=params_hash,
            receipt_hash=receipt_hash,
            timestamp=self._get_timestamp(),
            metadata=metadata or {}
        )

        self._receipts.append(receipt)

    def get_stats(self) -> Dict:
        """
        Get comprehensive telemetry statistics

        Returns:
            Dictionary with all telemetry data
        """
        # Calculate timer statistics
        timer_stats = {}
        for name, durations in self.timers.items():
            if durations:
                timer_stats[name] = {
                    "count": len(durations),
                    "min_ms": min(durations),
                    "max_ms": max(durations),
                    "avg_ms": sum(durations) // len(durations),
                    "total_ms": sum(durations)
                }

        return {
            "timestamp": self._get_timestamp(),
            "counters": self.counters,
            "timer_stats": timer_stats,
            "active_timers": list(self._active_timers.keys()),
            "receipt_count": len(self._receipts),
            "last_receipts": self.last_receipts[:5],  # Last 5 receipts
            "deterministic": self.deterministic,
            "history_size": {
                "counters": len(self._counter_history),
                "timers": len(self._timer_history),
                "receipts": len(self._receipts)
            }
        }

    def get_counter_history(self, name: str, limit: int = 100) -> List[Dict]:
        """Get historical counter values"""
        history = [
            asdict(entry) for entry in self._counter_history
            if entry.name == name
        ]
        return history[-limit:] if limit else history

    def get_timer_history(self, name: str, limit: int = 100) -> List[Dict]:
        """Get historical timer measurements"""
        history = [
            asdict(entry) for entry in self._timer_history
            if entry.name == name and entry.duration_ms is not None
        ]
        return history[-limit:] if limit else history

    def error(self, error_type: str, error_msg: str = "", metadata: Dict = None) -> None:
        """
        Record error event

        Args:
            error_type: Type/category of error
            error_msg: Error message
            metadata: Additional error metadata
        """
        self.count(f"error.{error_type}")
        self.count("error.total")

        # Record as receipt for error tracking
        self.record_receipt(
            tool="error.handler",
            params_hash=str(hash(error_type + error_msg))[:16],
            receipt_hash=str(hash(error_msg + str(self._counter)))[:16],
            metadata={
                "error_type": error_type,
                "error_msg": error_msg,
                **(metadata or {})
            }
        )

    def success(self, operation: str, metadata: Dict = None) -> None:
        """
        Record successful operation

        Args:
            operation: Operation name
            metadata: Additional metadata
        """
        self.count(f"success.{operation}")
        self.count("success.total")

    def reset(self) -> None:
        """Reset all telemetry data"""
        self._counters.clear()
        self._counter_history.clear()
        self._active_timers.clear()
        self._timer_history.clear()
        self._receipts.clear()

        if self.deterministic:
            self._fake_time = 1000000
            self._counter = 0

    def export_json(self) -> str:
        """Export all telemetry data as JSON"""
        data = {
            "counters": list(self._counter_history),
            "timers": [asdict(t) for t in self._timer_history],
            "receipts": [asdict(r) for r in self._receipts],
            "stats": self.get_stats()
        }
        return json.dumps(data, indent=2, sort_keys=True, default=str)

    def import_json(self, json_data: str) -> bool:
        """
        Import telemetry data from JSON

        Args:
            json_data: JSON string with telemetry data

        Returns:
            True if successful
        """
        try:
            data = json.loads(json_data)

            # Clear existing data
            self.reset()

            # Import counter history
            for entry_data in data.get("counters", []):
                entry = CounterEntry(**entry_data)
                self._counter_history.append(entry)
                self._counters[entry.name] = entry.value

            # Import timer history
            for entry_data in data.get("timers", []):
                entry = TimerEntry(**entry_data)
                self._timer_history.append(entry)

            # Import receipts
            for entry_data in data.get("receipts", []):
                entry = ReceiptEntry(**entry_data)
                self._receipts.append(entry)

            return True
        except (json.JSONDecodeError, TypeError, ValueError):
            return False

    def _get_timestamp(self) -> str:
        """Get current timestamp (deterministic in test mode)"""
        if self.deterministic:
            self._counter += 1
            return f"2025-01-01T00:00:{self._counter:02d}.000000Z"
        else:
            return datetime.now(timezone.utc).isoformat()

    def _get_fake_time(self) -> float:
        """Get fake monotonic time for deterministic mode"""
        if self.deterministic:
            self._fake_time += 100  # Advance by 100ms
            return self._fake_time / 1000.0
        else:
            return time.time()


class TelemetryContext:
    """Context manager for automatic timer management"""

    def __init__(self, telemetry: Telemetry, name: str, metadata: Dict = None):
        self.telemetry = telemetry
        self.name = name
        self.metadata = metadata or {}
        self.timer = None

    def __enter__(self):
        self.timer = self.telemetry.start_timer(self.name, self.metadata)
        return self.timer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Error occurred
            self.telemetry.error(
                error_type=exc_type.__name__,
                error_msg=str(exc_val),
                metadata={"timer": self.name}
            )

        self.telemetry.stop_timer(self.name)


def create_sample_telemetry(seed: int = 42) -> Telemetry:
    """
    Create sample telemetry data for testing/demo

    Args:
        seed: Deterministic seed

    Returns:
        Telemetry instance with sample data
    """
    telemetry = Telemetry(seed=seed)

    # Sample operations
    telemetry.count("spec.parsed", 1)
    telemetry.count("data.loaded", 1)
    telemetry.time("data.loading", 1500)

    telemetry.count("walkforward.runs", 5)
    telemetry.time("walkforward.total", 8500)

    telemetry.success("validation")
    telemetry.error("DataError", "Missing data point")

    telemetry.record_receipt(
        tool="research.walkforward",
        params_hash="aa11bb22cc33dd44",
        receipt_hash="ee55ff66gg77hh88"
    )

    return telemetry


if __name__ == "__main__":
    # Demo usage
    telemetry = create_sample_telemetry()

    print("Telemetry Stats:")
    print(json.dumps(telemetry.get_stats(), indent=2))

    print("\nUsing context manager:")
    with TelemetryContext(telemetry, "demo.operation") as timer:
        telemetry.count("demo.steps", 3)
        # Simulate work
        pass

    print("Updated stats:")
    print(f"Demo operations: {telemetry.counters.get('demo.steps', 0)}")
    print(f"Timer measurements: {len(telemetry.timers.get('demo.operation', []))}")