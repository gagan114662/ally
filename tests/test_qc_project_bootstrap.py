#!/usr/bin/env python3
"""
QC project bootstrap tests - offline file creation and receipt generation
"""

import os
import json
import pathlib
import pytest
import tempfile
import shutil

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_qc_bootstrap_offline(tmp_path):
    """Test QC project bootstrap without network calls"""
    # Create temporary templates directory
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    
    # Create mock templates
    config_template = {
        "environment": "live",
        "live-mode-brokerage": "PaperBrokerage",
        "api-access-token": "${QC_API_TOKEN}",
        "api-user-id": "${QC_USER_ID}",
        "data-folder": "${QC_DATA_DIR}",
        "result-destination-folder": "${QC_RESULTS_DIR}",
        "algorithm-language": "Python",
        "parameters": {}
    }
    
    with open(templates_dir / "config.live.paper.json", "w") as f:
        json.dump(config_template, f, indent=2)
    
    algo_template = """from AlgorithmImports import *

class AllyPaperAlgorithm(QCAlgorithm):
    def Initialize(self):
        symbols = self.GetParameter("ALLY_SYMBOLS").split(",")
        for s in symbols:
            self.AddEquity(s.strip(), Resolution.Minute)
        self.inbox_path = self.GetParameter("ALLY_INBOX")
    
    def OnData(self, data):
        pass
"""
    
    with open(templates_dir / "algorithm.py.j2", "w") as f:
        f.write(algo_template)
    
    # Test bootstrap
    from ally.integrations.quantconnect.qc_project_bootstrap import bootstrap_qc_project
    
    out_root = tmp_path / ".ally_qc"
    results_dir = tmp_path / "qc-results"
    data_dir = tmp_path / "qc-data"
    db_path = tmp_path / "test_receipts.duckdb"
    
    meta = bootstrap_qc_project(
        project_slug="ally-paper-test",
        symbols=["AAPL", "MSFT"],
        out_root=str(out_root),
        templates_root=str(templates_dir),
        results_dir=str(results_dir),
        data_dir=str(data_dir),
        params={"ALLY_INBOX": "orders_inbox.jsonl"},
        db_path=str(db_path),
        deterministic=True,
    )
    
    # Verify files created
    assert pathlib.Path(meta.project_dir).exists(), "Project directory should exist"
    assert pathlib.Path(meta.config_path).exists(), "Config file should exist"
    assert pathlib.Path(meta.algorithm_path).exists(), "Algorithm file should exist"
    assert pathlib.Path(meta.inbox_path).exists(), "Inbox file should exist"
    
    # Verify receipts present (16-char)
    assert len(meta.receipts["config"]) == 16, "Config receipt should be 16 chars"
    assert len(meta.receipts["algorithm"]) == 16, "Algorithm receipt should be 16 chars"
    assert len(meta.receipts["inbox"]) == 16, "Inbox receipt should be 16 chars"
    
    # Verify params hash (8-char)
    assert len(meta.params_hash) == 8, "Params hash should be 8 chars"
    
    # Verify config content
    with open(meta.config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    assert cfg["parameters"]["ALLY_INBOX"] == "orders_inbox.jsonl", "Inbox parameter should be set"
    assert "AAPL" in cfg["parameters"]["ALLY_SYMBOLS"], "AAPL should be in symbols"
    assert "MSFT" in cfg["parameters"]["ALLY_SYMBOLS"], "MSFT should be in symbols"
    assert cfg["data-folder"] == str(data_dir), "Data folder should be set"
    assert cfg["result-destination-folder"] == str(results_dir), "Results folder should be set"
    
    # Verify algorithm content
    with open(meta.algorithm_path, "r", encoding="utf-8") as f:
        algo_content = f.read()
    
    assert "orders_inbox.jsonl" in algo_content, "Algorithm should reference inbox file"
    assert "AAPL,MSFT" in algo_content, "Algorithm should reference symbols"
    
    # Verify inbox file is empty
    with open(meta.inbox_path, "r", encoding="utf-8") as f:
        inbox_content = f.read()
    
    assert inbox_content == "", "Inbox should be empty initially"
    
    # Verify database was created
    assert pathlib.Path(db_path).exists(), "Receipt database should be created"


def test_qc_bootstrap_deterministic():
    """Test that bootstrap produces deterministic results"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        
        # Create templates
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()
        
        config_template = {
            "environment": "live",
            "parameters": {}
        }
        
        with open(templates_dir / "config.live.paper.json", "w") as f:
            json.dump(config_template, f, indent=2, sort_keys=True)
        
        with open(templates_dir / "algorithm.py.j2", "w") as f:
            f.write("# Test algorithm")
        
        from ally.integrations.quantconnect.qc_project_bootstrap import bootstrap_qc_project
        
        # Bootstrap twice with same parameters
        common_args = {
            "project_slug": "test-deterministic",
            "symbols": ["TEST"],
            "out_root": str(tmp_path / ".ally_qc"),
            "templates_root": str(templates_dir),
            "results_dir": str(tmp_path / "results"),
            "data_dir": str(tmp_path / "data"),
            "params": {"TEST": "value"},
            "deterministic": True,
        }
        
        meta1 = bootstrap_qc_project(
            **common_args,
            db_path=str(tmp_path / "db1.duckdb")
        )
        
        meta2 = bootstrap_qc_project(
            **common_args,
            db_path=str(tmp_path / "db2.duckdb")
        )
        
        # Should have same params hash
        assert meta1.params_hash == meta2.params_hash, "Params hash should be deterministic"
        
        # Should have same file contents (receipts may differ due to timestamps)
        with open(meta1.config_path, "rb") as f1, open(meta2.config_path, "rb") as f2:
            assert f1.read() == f2.read(), "Config files should be identical"
        
        with open(meta1.algorithm_path, "rb") as f1, open(meta2.algorithm_path, "rb") as f2:
            assert f1.read() == f2.read(), "Algorithm files should be identical"


def test_qc_bootstrap_missing_template():
    """Test bootstrap fails gracefully with missing templates"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        
        from ally.integrations.quantconnect.qc_project_bootstrap import bootstrap_qc_project
        
        # Missing templates directory
        with pytest.raises(FileNotFoundError) as exc_info:
            bootstrap_qc_project(
                project_slug="test-missing",
                symbols=["TEST"],
                templates_root=str(tmp_path / "nonexistent"),
                db_path=str(tmp_path / "test.duckdb")
            )
        
        assert "Missing template" in str(exc_info.value), "Should report missing template"


def test_qc_bootstrap_invalid_inputs():
    """Test bootstrap validates inputs properly"""
    from ally.integrations.quantconnect.qc_project_bootstrap import bootstrap_qc_project
    
    # Empty project slug
    with pytest.raises(AssertionError) as exc_info:
        bootstrap_qc_project(
            project_slug="",
            symbols=["TEST"],
            db_path="/tmp/test.duckdb"
        )
    assert "project_slug is required" in str(exc_info.value)
    
    # Empty symbols
    with pytest.raises(AssertionError) as exc_info:
        bootstrap_qc_project(
            project_slug="test",
            symbols=[],
            db_path="/tmp/test.duckdb"
        )
    assert "symbols must be non-empty" in str(exc_info.value)
    
    # Invalid symbols
    with pytest.raises(AssertionError) as exc_info:
        bootstrap_qc_project(
            project_slug="test",
            symbols=["", "  "],
            db_path="/tmp/test.duckdb"
        )
    assert "symbols must be non-empty" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])