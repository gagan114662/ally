"""
Test suite for status telemetry functionality

Tests deterministic timers, counters, and metrics collection.
Validates telemetry data accuracy, deterministic behavior, and integration.
"""

import json
import pytest
import time
from ally.status.telemetry import Telemetry, TelemetryContext, create_sample_telemetry


class TestTelemetry:
    """Test telemetry functionality"""

    def test_telemetry_initialization(self):
        """Test telemetry initialization"""
        telemetry = Telemetry(seed=42)

        assert telemetry.deterministic is True
        assert telemetry.counters == {}
        assert telemetry.timers == {}
        assert telemetry.last_receipts == []

    def test_counter_operations(self):
        """Test counter increment operations"""
        telemetry = Telemetry(seed=42)

        # Initial count
        count1 = telemetry.count("test.operations")
        assert count1 == 1
        assert telemetry.counters["test.operations"] == 1

        # Increment by default (1)
        count2 = telemetry.count("test.operations")
        assert count2 == 2

        # Increment by custom amount
        count3 = telemetry.count("test.operations", increment=5)
        assert count3 == 7

        # Multiple counters
        telemetry.count("other.operations", increment=3)
        assert telemetry.counters["other.operations"] == 3
        assert telemetry.counters["test.operations"] == 7

    def test_counter_with_metadata(self):
        """Test counter operations with metadata"""
        telemetry = Telemetry(seed=42)

        metadata = {"strategy": "momentum", "symbol": "AAPL"}
        count = telemetry.count("trades.executed", increment=1, metadata=metadata)

        assert count == 1

        # Check counter history includes metadata
        history = telemetry.get_counter_history("trades.executed")
        assert len(history) == 1
        assert history[0]["metadata"] == metadata

    def test_timer_operations(self):
        """Test timer start/stop operations"""
        telemetry = Telemetry(seed=42)

        # Start timer
        timer = telemetry.start_timer("test.duration")
        assert timer.name == "test.duration"
        assert "test.duration" in telemetry.get_stats()["active_timers"]

        # Stop timer with deterministic duration
        duration = telemetry.stop_timer("test.duration", fake_duration=1500)
        assert duration == 1500
        assert "test.duration" not in telemetry.get_stats()["active_timers"]

        # Timer should be in history
        timer_stats = telemetry.timers
        assert "test.duration" in timer_stats
        assert timer_stats["test.duration"] == [1500]

    def test_timer_with_metadata(self):
        """Test timer operations with metadata"""
        telemetry = Telemetry(seed=42)

        metadata = {"operation": "data_load", "symbols": ["AAPL", "MSFT"]}
        timer = telemetry.start_timer("data.loading", metadata=metadata)
        telemetry.stop_timer("data.loading", fake_duration=2000)

        history = telemetry.get_timer_history("data.loading")
        assert len(history) == 1
        assert history[0]["metadata"] == metadata
        assert history[0]["duration_ms"] == 2000

    def test_direct_time_recording(self):
        """Test direct timer measurement recording"""
        telemetry = Telemetry(seed=42)

        metadata = {"batch_size": 100}
        telemetry.time("processing.batch", 750, metadata=metadata)

        timer_stats = telemetry.timers
        assert "processing.batch" in timer_stats
        assert timer_stats["processing.batch"] == [750]

    def test_multiple_timer_measurements(self):
        """Test multiple measurements for same timer"""
        telemetry = Telemetry(seed=42)

        # Record multiple measurements
        durations = [100, 200, 150, 300, 250]
        for duration in durations:
            telemetry.time("api.calls", duration)

        timer_stats = telemetry.timers
        assert timer_stats["api.calls"] == durations

        # Check timer statistics
        stats = telemetry.get_stats()["timer_stats"]
        assert "api.calls" in stats
        api_stats = stats["api.calls"]
        assert api_stats["count"] == 5
        assert api_stats["min_ms"] == 100
        assert api_stats["max_ms"] == 300
        assert api_stats["avg_ms"] == 200  # (100+200+150+300+250)/5
        assert api_stats["total_ms"] == 1000

    def test_receipt_recording(self):
        """Test receipt recording"""
        telemetry = Telemetry(seed=42)

        metadata = {"strategy_id": "strat_001"}
        telemetry.record_receipt(
            "research.walkforward",
            "aa11bb22cc33dd44",
            "ee55ff66gg77hh88",
            metadata=metadata
        )

        receipts = telemetry.last_receipts
        assert len(receipts) == 1

        receipt = receipts[0]
        assert receipt["tool"] == "research.walkforward"
        assert receipt["params_hash"] == "aa11bb22cc33dd44"
        assert receipt["receipt_hash"] == "ee55ff66gg77hh88"
        assert receipt["metadata"] == metadata
        assert "timestamp" in receipt

    def test_multiple_receipts_ordering(self):
        """Test multiple receipts are ordered correctly (most recent first)"""
        telemetry = Telemetry(seed=42)

        # Add receipts in sequence
        receipts_data = [
            ("tool1", "hash1", "receipt1"),
            ("tool2", "hash2", "receipt2"),
            ("tool3", "hash3", "receipt3")
        ]

        for tool, params_hash, receipt_hash in receipts_data:
            telemetry.record_receipt(tool, params_hash, receipt_hash)

        receipts = telemetry.last_receipts
        assert len(receipts) == 3

        # Should be in reverse order (most recent first)
        assert receipts[0]["tool"] == "tool3"
        assert receipts[1]["tool"] == "tool2"
        assert receipts[2]["tool"] == "tool1"

    def test_error_recording(self):
        """Test error event recording"""
        telemetry = Telemetry(seed=42)

        metadata = {"recovery_attempted": True}
        telemetry.error("DataError", "Missing data point", metadata=metadata)

        # Should increment error counters
        counters = telemetry.counters
        assert counters["error.DataError"] == 1
        assert counters["error.total"] == 1

        # Should record as receipt
        receipts = telemetry.last_receipts
        assert len(receipts) == 1
        assert receipts[0]["tool"] == "error.handler"
        assert receipts[0]["metadata"]["error_type"] == "DataError"

    def test_success_recording(self):
        """Test success event recording"""
        telemetry = Telemetry(seed=42)

        metadata = {"duration_ms": 1500}
        telemetry.success("validation", metadata=metadata)

        counters = telemetry.counters
        assert counters["success.validation"] == 1
        assert counters["success.total"] == 1

    def test_get_stats_comprehensive(self):
        """Test comprehensive statistics generation"""
        telemetry = Telemetry(seed=42)

        # Add various metrics
        telemetry.count("operations.completed", 5)
        telemetry.count("operations.failed", 2)
        telemetry.time("execution.duration", 1500)
        telemetry.time("execution.duration", 2000)
        telemetry.record_receipt("tool1", "hash1", "receipt1")
        telemetry.error("NetworkError", "Connection timeout")
        telemetry.success("validation")

        stats = telemetry.get_stats()

        # Check structure
        assert "timestamp" in stats
        assert "counters" in stats
        assert "timer_stats" in stats
        assert "active_timers" in stats
        assert "receipt_count" in stats
        assert "last_receipts" in stats
        assert "deterministic" in stats
        assert "history_size" in stats

        # Check counter data
        assert stats["counters"]["operations.completed"] == 5
        assert stats["counters"]["error.total"] == 1
        assert stats["counters"]["success.total"] == 1

        # Check timer stats
        timer_stats = stats["timer_stats"]["execution.duration"]
        assert timer_stats["count"] == 2
        assert timer_stats["avg_ms"] == 1750

        # Check receipt data
        assert stats["receipt_count"] == 2  # One manual + one from error

    def test_counter_history(self):
        """Test counter history tracking"""
        telemetry = Telemetry(seed=42)

        # Increment counter multiple times
        telemetry.count("test.counter", 1)
        telemetry.count("test.counter", 2)
        telemetry.count("test.counter", 3)

        history = telemetry.get_counter_history("test.counter")
        assert len(history) == 3

        # Check values are cumulative
        assert history[0]["value"] == 1
        assert history[1]["value"] == 3  # 1 + 2
        assert history[2]["value"] == 6  # 1 + 2 + 3

        # Check timestamps
        assert history[0]["timestamp"] < history[1]["timestamp"] < history[2]["timestamp"]

    def test_timer_history(self):
        """Test timer history tracking"""
        telemetry = Telemetry(seed=42)

        # Record timer measurements
        durations = [100, 200, 150]
        for duration in durations:
            telemetry.time("api.request", duration)

        history = telemetry.get_timer_history("api.request")
        assert len(history) == 3

        # Check durations
        for i, entry in enumerate(history):
            assert entry["duration_ms"] == durations[i]
            assert entry["name"] == "api.request"

    def test_history_limits(self):
        """Test history size limits"""
        telemetry = Telemetry(seed=42, max_history=3)

        # Add more entries than limit
        for i in range(5):
            telemetry.count("test.counter")
            telemetry.time("test.timer", 100 + i)
            telemetry.record_receipt(f"tool{i}", f"hash{i}", f"receipt{i}")

        # History should be limited
        counter_history = telemetry.get_counter_history("test.counter")
        timer_history = telemetry.get_timer_history("test.timer")
        receipts = telemetry.last_receipts

        assert len(counter_history) <= 3
        assert len(timer_history) <= 3
        assert len(receipts) <= 3

    def test_reset_functionality(self):
        """Test resetting telemetry data"""
        telemetry = Telemetry(seed=42)

        # Add data
        telemetry.count("test.counter", 5)
        telemetry.time("test.timer", 1000)
        telemetry.record_receipt("test.tool", "hash", "receipt")

        assert len(telemetry.counters) > 0
        assert len(telemetry.timers) > 0
        assert len(telemetry.last_receipts) > 0

        # Reset
        telemetry.reset()

        assert len(telemetry.counters) == 0
        assert len(telemetry.timers) == 0
        assert len(telemetry.last_receipts) == 0

    def test_export_import_json(self):
        """Test exporting and importing telemetry data"""
        telemetry = Telemetry(seed=42)

        # Add data
        telemetry.count("operations", 3)
        telemetry.time("duration", 1500)
        telemetry.record_receipt("tool", "hash", "receipt")

        # Export
        exported = telemetry.export_json()
        assert isinstance(exported, str)

        # Import to new telemetry
        new_telemetry = Telemetry(seed=42)
        assert new_telemetry.import_json(exported)

        # Should have same data
        assert new_telemetry.counters["operations"] == 3
        assert new_telemetry.timers["duration"] == [1500]
        assert len(new_telemetry.last_receipts) == 1

    def test_deterministic_behavior(self):
        """Test deterministic behavior with same seed"""
        # Create two telemetry instances with same seed
        telemetry1 = Telemetry(seed=42)
        telemetry2 = Telemetry(seed=42)

        # Perform same operations
        operations = [
            ("count", "test.ops", 1),
            ("time", "test.duration", 1000),
            ("receipt", "test.tool", "hash1", "receipt1")
        ]

        for op_type, *args in operations:
            if op_type == "count":
                telemetry1.count(args[0], args[1])
                telemetry2.count(args[0], args[1])
            elif op_type == "time":
                telemetry1.time(args[0], args[1])
                telemetry2.time(args[0], args[1])
            elif op_type == "receipt":
                telemetry1.record_receipt(args[0], args[1], args[2])
                telemetry2.record_receipt(args[0], args[1], args[2])

        # Should have identical results
        stats1 = telemetry1.get_stats()
        stats2 = telemetry2.get_stats()

        # Timestamps should be deterministic
        assert stats1["timestamp"] == stats2["timestamp"]

        # Counters should match
        assert stats1["counters"] == stats2["counters"]

        # Timers should match
        assert stats1["timer_stats"] == stats2["timer_stats"]


class TestTelemetryContext:
    """Test telemetry context manager"""

    def test_context_manager_success(self):
        """Test context manager for successful operations"""
        telemetry = Telemetry(seed=42)

        with TelemetryContext(telemetry, "test.operation") as timer:
            assert timer.name == "test.operation"
            # Simulate work
            pass

        # Timer should be stopped and recorded
        timers = telemetry.timers
        assert "test.operation" in timers
        assert len(timers["test.operation"]) == 1

    def test_context_manager_with_exception(self):
        """Test context manager with exception handling"""
        telemetry = Telemetry(seed=42)

        try:
            with TelemetryContext(telemetry, "failing.operation"):
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should record error
        counters = telemetry.counters
        assert "error.ValueError" in counters
        assert counters["error.ValueError"] == 1

        # Timer should still be recorded
        timers = telemetry.timers
        assert "failing.operation" in timers

    def test_context_manager_with_metadata(self):
        """Test context manager with metadata"""
        telemetry = Telemetry(seed=42)

        metadata = {"batch_size": 100, "strategy": "momentum"}

        with TelemetryContext(telemetry, "batch.processing", metadata):
            pass

        # Check timer history includes metadata
        history = telemetry.get_timer_history("batch.processing")
        assert len(history) == 1
        assert history[0]["metadata"] == metadata


class TestTelemetryHelpers:
    """Test telemetry helper functions"""

    def test_create_sample_telemetry(self):
        """Test sample telemetry creation"""
        telemetry = create_sample_telemetry(seed=42)

        # Should have sample data
        counters = telemetry.counters
        assert len(counters) > 0
        assert "spec.parsed" in counters

        timers = telemetry.timers
        assert len(timers) > 0
        assert "data.loading" in timers

        receipts = telemetry.last_receipts
        assert len(receipts) > 0


class TestTelemetryIntegration:
    """Integration tests for telemetry in realistic scenarios"""

    def test_complete_workflow_telemetry(self):
        """Test telemetry throughout complete workflow"""
        telemetry = Telemetry(seed=42)

        # Research phase
        with TelemetryContext(telemetry, "research.phase"):
            telemetry.count("spec.parsed")
            telemetry.success("spec.parsing")

            with TelemetryContext(telemetry, "data.loading"):
                telemetry.count("data.rows", 1000)
                telemetry.record_receipt("data.loader", "params123", "receipt456")

        # Evaluation phase
        with TelemetryContext(telemetry, "evaluation.phase"):
            telemetry.count("walkforward.runs", 5)

            # Simulate some errors and recoveries
            telemetry.error("DataError", "Missing data point")
            telemetry.success("data.recovery")

            telemetry.time("walkforward.total", 8500)

        # Final stats
        stats = telemetry.get_stats()

        # Should have counters from all phases
        counters = stats["counters"]
        assert counters["spec.parsed"] == 1
        assert counters["data.rows"] == 1000
        assert counters["walkforward.runs"] == 5
        assert counters["error.total"] == 1
        assert counters["success.total"] == 2

        # Should have timers
        timer_stats = stats["timer_stats"]
        assert "research.phase" in timer_stats
        assert "evaluation.phase" in timer_stats
        assert "walkforward.total" in timer_stats

        # Should have receipts
        assert stats["receipt_count"] >= 2  # Manual receipt + error receipt

    def test_error_tracking_patterns(self):
        """Test error tracking and recovery patterns"""
        telemetry = Telemetry(seed=42)

        # Different types of errors
        error_scenarios = [
            ("NetworkError", "Connection timeout"),
            ("DataError", "Missing data point"),
            ("ValidationError", "Invalid strategy parameters"),
            ("NetworkError", "Rate limit exceeded"),  # Duplicate type
        ]

        for error_type, error_msg in error_scenarios:
            telemetry.error(error_type, error_msg)

        # Check error counters
        counters = telemetry.counters
        assert counters["error.NetworkError"] == 2  # Two network errors
        assert counters["error.DataError"] == 1
        assert counters["error.ValidationError"] == 1
        assert counters["error.total"] == 4

        # Check error receipts
        receipts = telemetry.last_receipts
        error_receipts = [r for r in receipts if r["tool"] == "error.handler"]
        assert len(error_receipts) == 4

    def test_performance_monitoring(self):
        """Test performance monitoring patterns"""
        telemetry = Telemetry(seed=42)

        # Simulate operations with varying performance
        operations = [
            ("data.load", [100, 150, 120, 200, 180]),  # Data loading times
            ("model.train", [5000, 4800, 5200, 4900]),  # Model training times
            ("signal.generate", [50, 45, 55, 48, 52]),  # Signal generation times
        ]

        for operation, durations in operations:
            for duration in durations:
                telemetry.time(operation, duration)

        # Check performance statistics
        stats = telemetry.get_stats()["timer_stats"]

        # Data loading stats
        data_stats = stats["data.load"]
        assert data_stats["count"] == 5
        assert data_stats["min_ms"] == 100
        assert data_stats["max_ms"] == 200
        assert data_stats["avg_ms"] == 150  # (100+150+120+200+180)/5

        # Model training stats
        model_stats = stats["model.train"]
        assert model_stats["count"] == 4
        assert model_stats["avg_ms"] == 4975  # (5000+4800+5200+4900)/4

        # Signal generation stats
        signal_stats = stats["signal.generate"]
        assert signal_stats["count"] == 5
        assert signal_stats["min_ms"] == 45
        assert signal_stats["max_ms"] == 55


if __name__ == "__main__":
    pytest.main([__file__, "-v"])