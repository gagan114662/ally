"""
Tests for orchestrator runtime integration
"""

import pytest
from unittest.mock import Mock, patch
from ally.tools.orchestrator_runtime import orchestrator_demo, _maybe_runtime


class TestOrchestratorRuntime:
    
    def test_orchestrator_demo_disabled_runtime(self):
        """Test orchestrator demo with runtime disabled"""
        result = orchestrator_demo(
            experiment_id="test-disabled",
            use_runtime=False,
            runtime_live=False
        )
        
        assert result.ok
        assert result.data["experiment_id"] == "test-disabled"
        assert result.data["runtime_config"]["use_runtime"] is False
        assert result.data["runtime_stats"]["runtime_mode"] == "disabled"
        
        # All tasks should be passthrough
        for task_result in result.data["task_results"]:
            assert task_result["method"] == "passthrough"
            assert task_result["generated"] is None
    
    def test_orchestrator_demo_fixture_runtime(self):
        """Test orchestrator demo with fixture runtime"""
        with patch('ally.tools.orchestrator_runtime.TOOL_REGISTRY') as mock_registry:
            # Mock the runtime.generate tool
            mock_runtime = Mock()
            mock_runtime.return_value.ok = True
            mock_runtime.return_value.data = {"output": "FIXTURE_OUTPUT"}
            mock_registry.__getitem__.return_value = mock_runtime
            
            result = orchestrator_demo(
                experiment_id="test-fixture",
                use_runtime=True,
                runtime_live=False
            )
            
            assert result.ok
            assert result.data["runtime_config"]["use_runtime"] is True
            assert result.data["runtime_config"]["runtime_live"] is False
            assert result.data["runtime_stats"]["runtime_mode"] == "fixtures"
    
    def test_orchestrator_demo_live_runtime(self):
        """Test orchestrator demo with live runtime"""
        with patch('ally.tools.orchestrator_runtime.TOOL_REGISTRY') as mock_registry:
            # Mock the runtime.generate tool
            mock_runtime = Mock()
            mock_runtime.return_value.ok = True
            mock_runtime.return_value.data = {"output": "OLLAMA_OUTPUT"}
            mock_registry.__getitem__.return_value = mock_runtime
            
            result = orchestrator_demo(
                experiment_id="test-live",
                use_runtime=True,
                runtime_live=True
            )
            
            assert result.ok
            assert result.data["runtime_config"]["use_runtime"] is True
            assert result.data["runtime_config"]["runtime_live"] is True
            assert result.data["runtime_stats"]["runtime_mode"] == "live"
    
    def test_orchestrator_demo_custom_tasks(self):
        """Test orchestrator demo with custom tasks"""
        custom_tasks = [
            {"task": "math", "prompt": "Calculate 2+2"},
            {"task": "cv", "prompt": "Describe an image"}
        ]
        
        result = orchestrator_demo(
            experiment_id="test-custom",
            use_runtime=False,
            tasks=custom_tasks
        )
        
        assert result.ok
        assert len(result.data["task_results"]) == 2
        assert result.data["task_results"][0]["task"] == "math"
        assert result.data["task_results"][1]["task"] == "cv"
    
    def test_maybe_runtime_disabled(self):
        """Test _maybe_runtime with runtime disabled"""
        result = _maybe_runtime("codegen", "test prompt", False, False)
        assert result is None
    
    def test_maybe_runtime_enabled_success(self):
        """Test _maybe_runtime with runtime enabled and successful generation"""
        with patch('ally.tools.orchestrator_runtime.TOOL_REGISTRY') as mock_registry:
            mock_runtime = Mock()
            mock_runtime.return_value.ok = True
            mock_runtime.return_value.data = {"output": "SUCCESS_OUTPUT"}
            mock_registry.__getitem__.return_value = mock_runtime
            
            result = _maybe_runtime("codegen", "test prompt", True, False)
            assert result == "SUCCESS_OUTPUT"
            
            # Verify runtime.generate was called with correct params
            mock_registry.__getitem__.assert_called_with("runtime.generate")
            mock_runtime.assert_called_with(
                task="codegen",
                prompt="test prompt", 
                live=False,
                cache={"dir": "runs/cache"}
            )
    
    def test_maybe_runtime_enabled_failure(self):
        """Test _maybe_runtime with runtime enabled but generation fails"""
        with patch('ally.tools.orchestrator_runtime.TOOL_REGISTRY') as mock_registry:
            mock_runtime = Mock()
            mock_runtime.return_value.ok = False
            mock_registry.__getitem__.return_value = mock_runtime
            
            result = _maybe_runtime("codegen", "test prompt", True, False)
            assert result is None
    
    def test_maybe_runtime_exception(self):
        """Test _maybe_runtime with exception during generation"""
        with patch('ally.tools.orchestrator_runtime.TOOL_REGISTRY') as mock_registry:
            mock_registry.__getitem__.side_effect = Exception("Runtime error")
            
            result = _maybe_runtime("codegen", "test prompt", True, False)
            assert result is None
    
    def test_proof_lines_generation(self):
        """Test that orchestrator demo generates proper proof lines"""
        # This test would need to capture stdout to verify PROOF lines
        # For now, we verify the data structure contains the necessary info
        result = orchestrator_demo(
            experiment_id="proof-test",
            use_runtime=True,
            runtime_live=False
        )
        
        assert result.ok
        data = result.data
        
        # Verify all necessary data for proof generation exists
        assert "experiment_id" in data
        assert "runtime_config" in data
        assert "runtime_stats" in data
        assert "task_results" in data
        assert data["runtime_config"]["use_runtime"] is True
        assert data["runtime_config"]["runtime_live"] is False
        assert data["runtime_stats"]["runtime_mode"] == "fixtures"


@pytest.mark.m9
class TestOrchestratorRuntimeM9:
    """M9-specific tests for orchestrator runtime integration"""
    
    def test_m9_orchestrator_runtime_offline(self):
        """Test M9 orchestrator runtime integration in offline mode"""
        result = orchestrator_demo(
            experiment_id="m9-offline",
            use_runtime=True,
            runtime_live=False
        )
        
        assert result.ok
        assert result.data["runtime_stats"]["runtime_mode"] == "fixtures"
        
        # Verify default M9 task coverage
        task_types = {t["task"] for t in result.data["task_results"]}
        expected_tasks = {"codegen", "nlp", "math", "cv"}
        assert task_types == expected_tasks
    
    def test_m9_orchestrator_runtime_integration_proofs(self):
        """Test that M9 orchestrator generates integration proof data"""
        result = orchestrator_demo(
            experiment_id="m9-proofs",
            use_runtime=True,
            runtime_live=True
        )
        
        assert result.ok
        data = result.data
        
        # Verify proof-worthy data structure
        assert data["experiment_id"] == "m9-proofs"
        assert data["runtime_config"]["use_runtime"] is True
        assert data["runtime_config"]["runtime_live"] is True
        assert len(data["task_results"]) == 4  # Default task count
        assert all("task" in tr and "prompt" in tr and "method" in tr 
                  for tr in data["task_results"])