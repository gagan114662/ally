import os
import json
import shutil
import pytest
import tempfile
from pathlib import Path
from ally.tools import TOOL_REGISTRY
from ally.schemas.orch import OrchInput, OrchSummary
from ally.schemas.base import ToolResult


@pytest.fixture
def temp_runs_dir():
    """Create temporary runs directory for testing."""
    temp_dir = tempfile.mkdtemp()
    runs_dir = Path(temp_dir) / "runs"
    runs_dir.mkdir(exist_ok=True)
    
    # Set environment to use temp directory
    old_runs_dir = os.environ.get("ALLY_RUNS_DIR")
    os.environ["ALLY_RUNS_DIR"] = str(runs_dir)
    
    yield runs_dir
    
    # Cleanup
    if old_runs_dir:
        os.environ["ALLY_RUNS_DIR"] = old_runs_dir
    else:
        os.environ.pop("ALLY_RUNS_DIR", None)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_orch_001_basic_pipeline_execution(temp_runs_dir):
    """
    ORCH-001: Basic Pipeline Execution
    
    Verify that orchestrator.run executes the full pipeline:
    1. Research phase (CV + NLP)
    2. Signal synthesis
    3. Backtest optimization
    4. Risk policy validation
    5. Paper execution simulation
    6. Memory logging
    7. Report generation
    
    Expected: OrchSummary with valid run_id, KPIs, and report_path
    """
    # Ensure orchestrator tool is available
    assert "orchestrator.run" in TOOL_REGISTRY, "orchestrator.run tool not registered"
    
    # Execute orchestrator with basic config
    result = TOOL_REGISTRY["orchestrator.run"](
        experiment_id="test_basic_001",
        symbols=["BTCUSDT"],
        interval="1h",
        lookback=600,
        targets={"annual_return": 0.12, "sharpe_ratio": 1.2},
        save_run=True,
        make_report=True,
        seed=42
    )
    
    # Verify successful execution
    assert isinstance(result, ToolResult), "Should return ToolResult"
    assert result.ok, f"Pipeline should succeed: {result.errors}"
    assert result.data is not None, "Should have data"
    
    # Validate OrchSummary structure
    summary_data = result.data
    assert "experiment_id" in summary_data
    assert "run_id" in summary_data
    assert "best_params" in summary_data
    assert "kpis" in summary_data
    
    # Verify run_id format
    run_id = summary_data["run_id"]
    assert run_id.startswith("RUN_test_basic_001_"), "Run ID should have correct format"
    assert len(run_id.split("_")) >= 4, "Run ID should include timestamp"
    
    # Verify KPIs structure
    kpis = summary_data["kpis"]
    assert isinstance(kpis, dict), "KPIs should be dict"
    assert len(kpis) > 0, "Should have KPIs"
    
    # Expected KPI keys based on orchestrator logic
    expected_kpis = ["annual_return", "sharpe_ratio", "max_drawdown"]
    for kpi in expected_kpis:
        assert kpi in kpis, f"Should have {kpi} KPI"
        assert isinstance(kpis[kpi], (int, float)), f"{kpi} should be numeric"
    
    # Verify best params
    best_params = summary_data["best_params"]
    assert isinstance(best_params, dict), "Best params should be dict"
    assert len(best_params) > 0, "Should have best params"
    
    # Verify report generation
    report_path = summary_data.get("report_path")
    if report_path:
        assert report_path.endswith(".html"), "Report should be HTML file"


def test_orch_002_memory_integration_verification(temp_runs_dir):
    """
    ORCH-002: Memory Integration Verification
    
    Verify that orchestrator properly integrates with memory system:
    1. Logs run to database with all required fields
    2. Creates deterministic run_id and hashes
    3. Stores metrics, events, and trades
    4. Can query back the logged data
    
    Expected: Memory contains logged run with correct structure
    """
    # Execute orchestrator with memory logging
    result = TOOL_REGISTRY["orchestrator.run"](
        experiment_id="test_memory_002",
        symbols=["ETHUSDT"],
        save_run=True,
        make_report=False,  # Skip report to focus on memory
        seed=123
    )
    
    assert result.ok, f"Orchestrator should succeed: {result.errors}"
    run_id = result.data["run_id"]
    
    # Query memory to verify logging
    memory_query = TOOL_REGISTRY["memory.query"](
        table="runs"
    )
    
    assert memory_query.ok, "Memory query should succeed"
    all_runs = memory_query.data["rows"]
    runs = [r for r in all_runs if r.get("run_id") == run_id]
    assert len(runs) == 1, "Should find exactly one run"
    
    run_row = runs[0]
    # Verify required fields are present
    required_fields = ["run_id", "task", "code_hash", "inputs_hash", "ts"]
    for field in required_fields:
        assert field in run_row, f"Run should have {field}"
        assert run_row[field], f"{field} should not be empty"
    
    # Verify task format
    assert run_row["task"].startswith("orchestrator."), "Task should be orchestrator task"
    
    # Verify metrics are stored (check JSON column in runs table)
    assert "metrics" in run_row, "Run should have metrics column"
    metrics_json = run_row.get("metrics")
    if isinstance(metrics_json, str):
        import json
        metrics = json.loads(metrics_json)
    else:
        metrics = metrics_json or {}
    
    assert isinstance(metrics, dict), "Metrics should be stored as dict"
    assert len(metrics) > 0, "Should have metrics logged"
    
    # Verify events are stored (check JSON column in runs table)  
    assert "events" in run_row, "Run should have events column"
    events_json = run_row.get("events")
    if isinstance(events_json, str):
        import json
        events = json.loads(events_json)
    else:
        events = events_json or []
    
    assert isinstance(events, list), "Events should be stored as list"
    assert len(events) > 0, "Should have events logged"
    
    # Verify event structure and orchestrator events
    orchestrator_events = [e for e in events if e.get("type", "").startswith("orchestrator.")]
    assert len(orchestrator_events) >= 2, "Should have orchestrator lifecycle events"


def test_orch_003_deterministic_behavior_validation(temp_runs_dir):
    """
    ORCH-003: Deterministic Behavior Validation
    
    Verify that orchestrator produces deterministic results:
    1. Same inputs → same run_id format and hashes
    2. Same KPIs with identical configuration
    3. Consistent report generation
    4. Reproducible with same seed
    
    Expected: Two runs with same config produce equivalent results
    """
    common_config = {
        "experiment_id": "test_deterministic_003",
        "symbols": ["BTCUSDT"],
        "interval": "4h",
        "lookback": 300,
        "targets": {"annual_return": 0.15, "sharpe_ratio": 1.5},
        "save_run": True,
        "make_report": True,
        "seed": 999  # Fixed seed for determinism
    }
    
    # Run 1
    result1 = TOOL_REGISTRY["orchestrator.run"](**common_config)
    assert result1.ok, f"First run should succeed: {result1.errors}"
    
    # Run 2 with same config
    result2 = TOOL_REGISTRY["orchestrator.run"](**common_config)
    assert result2.ok, f"Second run should succeed: {result2.errors}"
    
    # Extract summaries
    summary1 = result1.data
    summary2 = result2.data
    
    # Verify experiment IDs match
    assert summary1["experiment_id"] == summary2["experiment_id"]
    
    # Verify run_ids have same format (different timestamps expected)
    run_id1_parts = summary1["run_id"].split("_")
    run_id2_parts = summary2["run_id"].split("_")
    assert run_id1_parts[0] == run_id2_parts[0] == "RUN"  # Prefix
    assert run_id1_parts[1] == run_id2_parts[1]  # Experiment ID
    # Timestamps will differ, that's expected
    
    # Verify KPIs are consistent (deterministic backtest results)
    kpis1 = summary1["kpis"]
    kpis2 = summary2["kpis"]
    
    # With same seed and config, core KPIs should be identical or very close
    for kpi in ["annual_return", "sharpe_ratio", "max_drawdown"]:
        if kpi in kpis1 and kpi in kpis2:
            diff = abs(kpis1[kpi] - kpis2[kpi])
            assert diff < 0.01, f"{kpi} should be deterministic: {kpis1[kpi]} vs {kpis2[kpi]}"
    
    # Verify best_params consistency
    params1 = summary1["best_params"]
    params2 = summary2["best_params"]
    assert params1 == params2, "Best params should be identical with same seed"
    
    # Verify both have reports or both don't
    report1 = summary1.get("report_path")
    report2 = summary2.get("report_path")
    assert bool(report1) == bool(report2), "Report generation should be consistent"


def test_orch_schema_validation():
    """
    ORCH-SCHEMA: Schema Validation
    
    Verify that OrchSummary output conforms to JSON schema.
    """
    # Load JSON schema
    schema_path = Path(__file__).parent.parent / "ally" / "verify" / "jsonschema" / "orch_summary.schema.json"
    assert schema_path.exists(), "OrchSummary schema should exist"
    
    with open(schema_path) as f:
        schema = json.load(f)
    
    # Execute orchestrator
    result = TOOL_REGISTRY["orchestrator.run"](
        experiment_id="test_schema_validation",
        seed=456
    )
    
    assert result.ok, "Orchestrator should succeed for schema test"
    summary_data = result.data
    
    # Validate against schema
    from jsonschema import validate, ValidationError
    try:
        validate(instance=summary_data, schema=schema)
    except ValidationError as e:
        pytest.fail(f"OrchSummary output doesn't match schema: {e}")
    
    # Additional specific validations
    assert isinstance(summary_data["experiment_id"], str)
    assert isinstance(summary_data["run_id"], str)
    assert isinstance(summary_data["best_params"], dict)
    assert isinstance(summary_data["kpis"], dict)
    assert len(summary_data["best_params"]) > 0
    assert len(summary_data["kpis"]) > 0
    
    # Validate run_id format
    import re
    run_id_pattern = r"^RUN_[A-Za-z0-9_-]+_[0-9]{8}T[0-9]{6}Z$"
    assert re.match(run_id_pattern, summary_data["run_id"]), "Run ID format should match schema pattern"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])