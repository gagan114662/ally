import os
import json
import hashlib
import tempfile
import shutil
from datetime import datetime
import pytest
import jsonschema

from ally.tools.memory import memory_log_run, memory_query
from ally.tools.reporting import generate_tearsheet
from ally.utils.db import DatabaseManager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_ally_memory.duckdb")
    db = DatabaseManager(db_path)
    yield db
    db.close()
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_run_data():
    """Sample run data for testing."""
    return {
        "run_id": "TEST_RUN_001",
        "task": "cv.detect_chart_patterns",
        "code_hash": "abc123def456",
        "inputs_hash": "fed654cba321",
        "ts": "2025-01-15T12:00:00Z",
        "metrics": {"annual_return": 0.18, "sharpe_ratio": 1.6, "max_drawdown": -0.12},
        "events": [{"type": "cv.trendline_break", "payload": {"symbol": "BTCUSDT"}}],
        "trades": [
            {"symbol": "BTCUSDT", "side": "buy", "qty": 1.0, "price": 45000.0, "ts": "2025-01-15T12:05:00Z"},
            {"symbol": "BTCUSDT", "side": "sell", "qty": 1.0, "price": 46000.0, "ts": "2025-01-15T12:10:00Z"}
        ],
        "notes": "Test run for memory and reporting"
    }


def test_mem_001_log_run_idempotent(temp_db, sample_run_data):
    """MEM-001: memory.log_run inserts rows; re-logging same (run_id, code_hash, inputs_hash) is idempotent."""
    
    # Patch the database manager
    import ally.tools.memory
    original_get_db = ally.tools.memory.get_db_manager
    ally.tools.memory.get_db_manager = lambda: temp_db
    
    try:
        # First log
        result1 = memory_log_run(**sample_run_data)
        assert result1.ok is True
        assert result1.data["logged"] is True
        
        # Check data was inserted
        query_result = memory_query(table="runs")
        assert query_result.ok is True
        assert query_result.data["count"] == 1
        assert len(query_result.data["rows"]) == 1
        
        # Second log with same hashes (should be idempotent)
        result2 = memory_log_run(**sample_run_data)
        assert result2.ok is True
        assert result2.data["logged"] is True
        
        # Verify no duplicate rows
        query_result = memory_query(table="runs")
        assert query_result.ok is True
        assert query_result.data["count"] == 1  # Still only 1 row
        
        # Different inputs_hash should create new row
        different_data = sample_run_data.copy()
        different_data["run_id"] = "TEST_RUN_002"
        different_data["inputs_hash"] = "different_hash"
        result3 = memory_log_run(**different_data)
        assert result3.ok is True
        
        # Should now have 2 rows
        query_result = memory_query(table="runs")
        assert query_result.ok is True
        assert query_result.data["count"] == 2
        
    finally:
        ally.tools.memory.get_db_manager = original_get_db


def test_mem_002_query_with_where_iso_z_timestamps(temp_db, sample_run_data):
    """MEM-002: memory.query with where clause returns expected count and ISO-Z timestamps."""
    
    # Patch the database manager
    import ally.tools.memory
    original_get_db = ally.tools.memory.get_db_manager
    ally.tools.memory.get_db_manager = lambda: temp_db
    
    try:
        # Log test data
        memory_log_run(**sample_run_data)
        
        # Log another run with different task
        other_data = sample_run_data.copy()
        other_data["run_id"] = "TEST_RUN_002"
        other_data["task"] = "nlp.extract_events"
        other_data["inputs_hash"] = "different_hash"
        memory_log_run(**other_data)
        
        # Query with WHERE clause
        result = memory_query(table="runs", where="task='cv.detect_chart_patterns'", limit=5)
        
        assert result.ok is True
        assert result.data["count"] == 1  # Only one run matches
        assert len(result.data["rows"]) == 1
        
        row = result.data["rows"][0]
        assert row["task"] == "cv.detect_chart_patterns"
        assert row["ts"] == "2025-01-15T12:00:00Z"  # ISO-Z format preserved
        
        # Test limit
        result = memory_query(table="runs", limit=1)
        assert result.ok is True
        assert len(result.data["rows"]) == 1
        assert result.data["count"] == 2  # Total count is still 2
        
    finally:
        ally.tools.memory.get_db_manager = original_get_db


def test_rep_001_tearsheet_generates_html_and_summary(temp_db, sample_run_data):
    """REP-001: reporting.generate_tearsheet writes an HTML file; returns ReportSummary matching schema."""
    
    # Patch the database manager
    import ally.tools.reporting
    original_get_db = ally.tools.reporting.get_db_manager
    ally.tools.reporting.get_db_manager = lambda: temp_db
    
    try:
        # Log test data first
        import ally.tools.memory
        ally.tools.memory.get_db_manager = lambda: temp_db
        memory_log_run(**sample_run_data)
        
        # Generate tearsheet
        result = generate_tearsheet(run_id="TEST_RUN_001")
        
        assert result.ok is True
        
        # Check ReportSummary fields
        summary = result.data
        assert summary["run_id"] == "TEST_RUN_001"
        assert "kpis" in summary
        assert "n_trades" in summary
        assert "sections" in summary
        assert "html_path" in summary
        assert summary["n_trades"] == 2
        
        # Verify HTML file exists
        html_path = summary["html_path"]
        assert os.path.exists(html_path)
        
        # Verify HTML content
        with open(html_path, "r") as f:
            html_content = f.read()
        
        assert "TEST_RUN_001" in html_content
        assert "Tearsheet" in html_content
        assert "BTCUSDT" in html_content
        
        # Cleanup
        if os.path.exists(html_path):
            os.remove(html_path)
    
    finally:
        ally.tools.reporting.get_db_manager = original_get_db
        ally.tools.memory.get_db_manager = lambda: temp_db


def test_rep_002_tearsheet_deterministic_summary_sha1(temp_db, sample_run_data):
    """REP-002: Deterministic summary SHA1 for same inputs (two runs identical)."""
    
    # Patch the database manager
    import ally.tools.reporting
    import ally.tools.memory
    original_get_db_rep = ally.tools.reporting.get_db_manager
    original_get_db_mem = ally.tools.memory.get_db_manager
    ally.tools.reporting.get_db_manager = lambda: temp_db
    ally.tools.memory.get_db_manager = lambda: temp_db
    
    try:
        # Log test data
        memory_log_run(**sample_run_data)
        
        # Generate tearsheet twice
        result1 = generate_tearsheet(run_id="TEST_RUN_001")
        result2 = generate_tearsheet(run_id="TEST_RUN_001")
        
        assert result1.ok is True
        assert result2.ok is True
        
        # Get summaries and remove html_path for comparison
        summary1 = result1.data.copy()
        summary2 = result2.data.copy()
        
        html_path1 = summary1.pop("html_path")
        html_path2 = summary2.pop("html_path")
        
        # Calculate SHA1 of summaries
        summary1_json = json.dumps(summary1, sort_keys=True)
        summary2_json = json.dumps(summary2, sort_keys=True)
        
        sha1_1 = hashlib.sha1(summary1_json.encode()).hexdigest()
        sha1_2 = hashlib.sha1(summary2_json.encode()).hexdigest()
        
        assert sha1_1 == sha1_2, "Tearsheet summaries should be deterministic"
        assert html_path1 == html_path2, "HTML paths should be identical for same run"
        
        # Cleanup
        for path in [html_path1, html_path2]:
            if os.path.exists(path):
                os.remove(path)
                
    finally:
        ally.tools.reporting.get_db_manager = original_get_db_rep
        ally.tools.memory.get_db_manager = original_get_db_mem


def test_schema_001_query_out_validates_against_schema(temp_db, sample_run_data):
    """SCHEMA-001: Validate QueryOut JSON against schema."""
    
    # Load schema
    schema_path = "ally/verify/jsonschema/memory_query.schema.json"
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    # Patch the database manager
    import ally.tools.memory
    original_get_db = ally.tools.memory.get_db_manager
    ally.tools.memory.get_db_manager = lambda: temp_db
    
    try:
        # Log test data
        memory_log_run(**sample_run_data)
        
        # Query data
        result = memory_query(table="runs")
        assert result.ok is True
        
        # Validate against schema
        jsonschema.validate(result.data, schema)
        
        # Should not raise exception if valid
        assert True
        
    finally:
        ally.tools.memory.get_db_manager = original_get_db


def test_schema_002_report_summary_validates_against_schema(temp_db, sample_run_data):
    """SCHEMA-002: Validate ReportSummary JSON against schema."""
    
    # Load schema
    schema_path = "ally/verify/jsonschema/report_summary.schema.json"
    with open(schema_path, "r") as f:
        schema = json.load(f)
    
    # Patch database managers
    import ally.tools.reporting
    import ally.tools.memory
    original_get_db_rep = ally.tools.reporting.get_db_manager
    original_get_db_mem = ally.tools.memory.get_db_manager
    ally.tools.reporting.get_db_manager = lambda: temp_db
    ally.tools.memory.get_db_manager = lambda: temp_db
    
    try:
        # Log test data
        memory_log_run(**sample_run_data)
        
        # Generate tearsheet
        result = generate_tearsheet(run_id="TEST_RUN_001")
        assert result.ok is True
        
        # Validate against schema
        jsonschema.validate(result.data, schema)
        
        # Should not raise exception if valid
        assert True
        
        # Cleanup
        html_path = result.data["html_path"]
        if os.path.exists(html_path):
            os.remove(html_path)
            
    finally:
        ally.tools.reporting.get_db_manager = original_get_db_rep
        ally.tools.memory.get_db_manager = original_get_db_mem