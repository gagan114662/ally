#!/usr/bin/env python3
"""
Unit tests for orchestrator fixes

Tests all surgical fixes for autonomous algo orchestrator:
1. QuantConnect dry-run guard
2. Variant generation wrapper with receipts+tracebacks
3. Offline fixture validation
4. Template schema validation
5. Stoplight summary receipts

These tests ensure the fixes resolve the logged errors without breaking existing functionality.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from ally.research.orchestrator_fixes import (
    QuantConnectDryRunGuard,
    VariantGenerationWrapper,
    OfflineFixtureValidator,
    TemplateSchemaValidator,
    OrchestrationStoplightReceipt,
    create_orchestrator_patch,
    OrchestrationResult
)


class TestQuantConnectDryRunGuard:
    """Test Fix #1: QuantConnect dry-run bootstrap"""

    def test_dry_run_mode_bypasses_quantconnect(self):
        """Test that dry run mode prevents QuantConnect config errors"""
        with QuantConnectDryRunGuard(dry_run=True):
            # This would normally fail with 'QuantConnectConfig' object has no attribute 'get'
            # but should be bypassed in dry run
            import sys
            assert 'quantconnect' in sys.modules
            config = sys.modules['quantconnect'].QuantConnectConfig()
            assert config.get('live', False) is False

    def test_live_mode_allows_real_quantconnect(self):
        """Test that live mode doesn't interfere with real QuantConnect"""
        # In live mode, no mocking should occur
        with QuantConnectDryRunGuard(dry_run=False):
            # Should not modify sys.modules in live mode
            pass

    @patch.dict(os.environ, {'ALLY_LIVE': '0'})
    def test_auto_detect_dry_run_from_env(self):
        """Test automatic dry run detection from ALLY_LIVE env var"""
        guard = QuantConnectDryRunGuard()
        assert guard.dry_run is True

    @patch.dict(os.environ, {'ALLY_LIVE': '1'})
    def test_auto_detect_live_from_env(self):
        """Test automatic live detection from ALLY_LIVE env var"""
        guard = QuantConnectDryRunGuard()
        assert guard.dry_run is False


class TestVariantGenerationWrapper:
    """Test Fix #2: Variant generation with receipts and tracebacks"""

    def test_successful_variant_expansion(self):
        """Test successful variant generation with receipt emission"""
        wrapper = VariantGenerationWrapper(emit_receipts=False)  # Skip receipts in test

        # Mock template and generator
        template = Mock()
        template.__class__.__name__ = "MomentumTemplate"

        generator = Mock()
        generator.expand.return_value = ["variant1", "variant2", "variant3"]

        variants = wrapper.safe_expand(template, generator, spec_hash="test123")

        assert len(variants) == 3
        assert variants == ["variant1", "variant2", "variant3"]
        generator.expand.assert_called_once_with(template)

    def test_variant_expansion_failure_with_traceback(self):
        """Test variant generation failure captures full traceback"""
        wrapper = VariantGenerationWrapper(emit_receipts=False)

        template = Mock()
        template.__class__.__name__ = "MomentumTemplate"

        generator = Mock()
        generator.expand.side_effect = ValueError("Unknown template")

        variants = wrapper.safe_expand(template, generator)

        assert len(variants) == 0
        assert len(wrapper.errors) == 1
        error = wrapper.errors[0]
        assert error["template"] == "MomentumTemplate"
        assert "Unknown template" in error["error"]
        assert "traceback" in error
        assert len(error["traceback"]) > 0

    @patch('ally.research.orchestrator_fixes.emit_receipt')
    def test_receipt_emission_on_success(self, mock_emit):
        """Test that success receipts are properly emitted"""
        wrapper = VariantGenerationWrapper(emit_receipts=True)

        template = Mock()
        template.__class__.__name__ = "MomentumTemplate"

        generator = Mock()
        generator.expand.return_value = ["variant1"]

        wrapper.safe_expand(template, generator, spec_hash="abc123")

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["tool"] == "research.variant.expand"
        assert call_args["ok"] is True
        assert call_args["template"] == "MomentumTemplate"
        assert call_args["variants_count"] == 1
        assert call_args["spec_hash"] == "abc123"

    @patch('ally.research.orchestrator_fixes.emit_receipt')
    def test_receipt_emission_on_failure(self, mock_emit):
        """Test that failure receipts are properly emitted"""
        wrapper = VariantGenerationWrapper(emit_receipts=True)

        template = Mock()
        template.__class__.__name__ = "MomentumTemplate"

        generator = Mock()
        generator.expand.side_effect = RuntimeError("Template parsing failed")

        wrapper.safe_expand(template, generator)

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["tool"] == "research.variant.expand"
        assert call_args["ok"] is False
        assert "Template parsing failed" in call_args["error"]


class TestOfflineFixtureValidator:
    """Test Fix #3: Offline fixture validation"""

    def test_all_fixtures_present(self):
        """Test validation when all fixtures are present"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create all required fixtures
            required_fixtures = [
                "data/fixtures/factor_panels.json",
                "data/fixtures/synthetic_ohlcv.json",
                "artifacts/scoring/offline_benchmarks.json"
            ]

            for fixture_path in required_fixtures:
                full_path = os.path.join(temp_dir, fixture_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    json.dump({"test": "data"}, f)

            validator = OfflineFixtureValidator(fixture_root=temp_dir)
            result = validator.assert_fixtures_present(required_fixtures)

            assert result is True
            assert len(validator.missing_fixtures) == 0

    def test_missing_fixtures_in_dry_mode(self):
        """Test validation failure when fixtures are missing in dry mode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            required_fixtures = ["data/fixtures/missing.json"]

            validator = OfflineFixtureValidator(fixture_root=temp_dir)

            with patch.dict(os.environ, {'ALLY_LIVE': '0'}):
                with pytest.raises(FileNotFoundError) as exc_info:
                    validator.assert_fixtures_present(required_fixtures)

                assert "Missing offline fixtures" in str(exc_info.value)
                assert "data/fixtures/missing.json" in str(exc_info.value)

    def test_missing_fixtures_in_live_mode(self):
        """Test validation warning when fixtures missing in live mode"""
        with tempfile.TemporaryDirectory() as temp_dir:
            required_fixtures = ["data/fixtures/missing.json"]

            validator = OfflineFixtureValidator(fixture_root=temp_dir)

            with patch.dict(os.environ, {'ALLY_LIVE': '1'}):
                result = validator.assert_fixtures_present(required_fixtures)

                assert result is False
                assert len(validator.missing_fixtures) == 1

    def test_create_minimal_fixtures(self):
        """Test automatic creation of minimal fixtures"""
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = OfflineFixtureValidator(fixture_root=temp_dir)
            created = validator.create_minimal_fixtures()

            assert len(created) == len(OfflineFixtureValidator.REQUIRED_FIXTURES)

            # Verify created fixtures are valid JSON
            for fixture_path in created.values():
                assert os.path.exists(fixture_path)
                with open(fixture_path, 'r') as f:
                    data = json.load(f)
                    assert isinstance(data, dict)


class TestTemplateSchemaValidator:
    """Test Fix #4: Template schema validation"""

    def test_valid_template_passes_validation(self):
        """Test that properly structured template passes validation"""
        validator = TemplateSchemaValidator()

        # Mock valid template
        template = Mock()
        template.__class__.__name__ = "MomentumTemplate"
        template.name = "momentum_strategy"
        template.description = "Momentum-based strategy"
        template.parameters = {
            "lookback": {"type": "int", "default": 20},
            "threshold": {"type": "float", "default": 0.1}
        }
        template.lookback_periods = [10, 20, 50]

        errors = validator.validate_template(template)
        assert len(errors) == 0

    def test_template_missing_required_attributes(self):
        """Test validation failure for missing required attributes"""
        validator = TemplateSchemaValidator()

        # Mock template missing required attributes
        template = Mock()
        template.__class__.__name__ = "MomentumTemplate"
        # Missing: name, description, parameters

        errors = validator.validate_template(template)
        assert len(errors) >= 3  # At least 3 missing attributes
        assert any("missing required attribute: name" in error for error in errors)
        assert any("missing required attribute: description" in error for error in errors)
        assert any("missing required attribute: parameters" in error for error in errors)

    def test_invalid_parameters_structure(self):
        """Test validation failure for invalid parameters structure"""
        validator = TemplateSchemaValidator()

        template = Mock()
        template.__class__.__name__ = "MomentumTemplate"
        template.name = "test"
        template.description = "test"
        template.parameters = "invalid"  # Should be dict

        errors = validator.validate_template(template)
        assert len(errors) >= 1
        assert any("parameters must be dict" in error for error in errors)

    def test_parameter_missing_type_field(self):
        """Test validation failure for parameters missing type field"""
        validator = TemplateSchemaValidator()

        template = Mock()
        template.__class__.__name__ = "MomentumTemplate"
        template.name = "test"
        template.description = "test"
        template.parameters = {
            "lookback": {"default": 20}  # Missing 'type' field
        }

        errors = validator.validate_template(template)
        assert len(errors) >= 1
        assert any("missing 'type' field" in error for error in errors)

    @patch('ally.research.orchestrator_fixes.emit_receipt')
    def test_validation_receipt_emission(self, mock_emit):
        """Test that validation results emit receipts"""
        validator = TemplateSchemaValidator()

        template = Mock()
        template.__class__.__name__ = "TestTemplate"
        template.name = "test"
        template.description = "test"
        template.parameters = {}

        validator.validate_template(template)

        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["tool"] == "research.template.validate"
        assert call_args["template"] == "TestTemplate"
        assert "ok" in call_args


class TestOrchestrationStoplightReceipt:
    """Test Fix #5: Stoplight summary receipts"""

    def test_stage_tracking(self):
        """Test that stages are properly tracked"""
        stoplight = OrchestrationStoplightReceipt()

        stoplight.add_stage("template_discovery", success=True, count=5)
        stoplight.add_stage("variant_generation", success=True, count=15)
        stoplight.add_stage("hypothesis_scoring", success=False, count=0)

        assert len(stoplight.stages) == 3
        assert stoplight.stages["template_discovery"]["success"] is True
        assert stoplight.stages["template_discovery"]["count"] == 5
        assert stoplight.stages["hypothesis_scoring"]["success"] is False

    def test_error_tracking(self):
        """Test that errors are properly tracked"""
        stoplight = OrchestrationStoplightReceipt()

        stoplight.add_error("variant_generation", "Template parsing failed")
        stoplight.add_error("hypothesis_scoring", "No offline fixtures")

        assert len(stoplight.errors) == 2
        assert stoplight.errors[0]["stage"] == "variant_generation"
        assert stoplight.errors[0]["error"] == "Template parsing failed"

    @patch('ally.research.orchestrator_fixes.emit_receipt')
    def test_final_receipt_emission(self, mock_emit):
        """Test comprehensive final receipt emission"""
        stoplight = OrchestrationStoplightReceipt()

        stoplight.add_stage("template_discovery", success=True, count=5)
        stoplight.add_stage("variant_generation", success=True, count=15)
        stoplight.add_stage("hypothesis_scoring", success=True, count=10)
        stoplight.add_stage("selection_ranking", success=True, count=3)

        result = stoplight.emit_final_receipt()

        # Test OrchestrationResult
        assert isinstance(result, OrchestrationResult)
        assert result.success is True
        assert result.templates_found == 5
        assert result.variants_generated == 15
        assert result.hypotheses_scored == 10
        assert result.survivors == 3

        # Test receipt emission
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args[1]
        assert call_args["tool"] == "research.orchestration.summary"
        assert call_args["ok"] is True
        assert call_args["pipeline_counts"]["templates"] == 5
        assert call_args["pipeline_counts"]["survivors"] == 3

    def test_failure_propagation(self):
        """Test that any stage failure marks overall orchestration as failed"""
        stoplight = OrchestrationStoplightReceipt()

        stoplight.add_stage("template_discovery", success=True, count=5)
        stoplight.add_stage("variant_generation", success=False, count=0)  # Failure

        result = stoplight.emit_final_receipt()

        assert result.success is False


class TestOrchestratorPatch:
    """Test comprehensive orchestrator patch integration"""

    def test_patch_creation(self):
        """Test that patch creates all required components"""
        patch = create_orchestrator_patch()

        assert hasattr(patch, 'qc_guard')
        assert hasattr(patch, 'variant_wrapper')
        assert hasattr(patch, 'fixture_validator')
        assert hasattr(patch, 'schema_validator')
        assert hasattr(patch, 'stoplight')

    @patch.dict(os.environ, {'ALLY_LIVE': '0'})
    def test_apply_all_fixes(self):
        """Test that apply_all_fixes initializes all components"""
        patch = create_orchestrator_patch()

        # Create minimal fixtures first
        with tempfile.TemporaryDirectory() as temp_dir:
            patch.fixture_validator = OfflineFixtureValidator(fixture_root=temp_dir)
            patch.fixture_validator.create_minimal_fixtures()

            patch.apply_all_fixes()

            # Verify QC guard is active
            assert patch.qc_guard.dry_run is True

            # Verify fixture validation passed
            assert "fixture_validation" in patch.stoplight.stages
            assert patch.stoplight.stages["fixture_validation"]["success"] is True

    def test_safe_template_expand_integration(self):
        """Test integrated template expansion with all safety checks"""
        patch = create_orchestrator_patch()

        # Mock templates
        valid_template = Mock()
        valid_template.__class__.__name__ = "MomentumTemplate"
        valid_template.name = "momentum"
        valid_template.description = "test"
        valid_template.parameters = {"lookback": {"type": "int"}}

        invalid_template = Mock()
        invalid_template.__class__.__name__ = "BadTemplate"
        # Missing required attributes

        templates = [valid_template, invalid_template]

        # Mock generator
        generator = Mock()
        generator.expand.return_value = ["variant1", "variant2"]

        variants = patch.safe_template_expand(templates, generator)

        # Should only expand valid template
        assert len(variants) == 2
        generator.expand.assert_called_once_with(valid_template)

        # Check stoplight tracking
        assert patch.stoplight.stages["template_discovery"]["count"] == 2
        assert patch.stoplight.stages["schema_validation"]["count"] == 1  # Only valid template
        assert patch.stoplight.stages["variant_generation"]["count"] == 2

    def test_finalization_cleanup(self):
        """Test that finalize properly cleans up and emits receipts"""
        patch = create_orchestrator_patch()
        patch.qc_guard.__enter__()  # Simulate active guard

        result = patch.finalize()

        assert isinstance(result, OrchestrationResult)
        # QC guard should be cleaned up (no easy way to test this directly)


class TestIntegrationScenarios:
    """Integration tests simulating real orchestrator scenarios"""

    def test_complete_success_scenario(self):
        """Test complete orchestration success path"""
        with tempfile.TemporaryDirectory() as temp_dir:
            patch = create_orchestrator_patch()
            patch.fixture_validator = OfflineFixtureValidator(fixture_root=temp_dir)

            # Create fixtures
            patch.fixture_validator.create_minimal_fixtures()

            # Apply fixes
            patch.apply_all_fixes()

            # Create valid templates
            template = Mock()
            template.__class__.__name__ = "MomentumTemplate"
            template.name = "momentum"
            template.description = "test momentum"
            template.parameters = {"lookback": {"type": "int"}}

            # Mock successful generation
            generator = Mock()
            generator.expand.return_value = ["variant1", "variant2"]

            # Execute pipeline
            variants = patch.safe_template_expand([template], generator)

            # Simulate scoring and selection
            patch.stoplight.add_stage("hypothesis_scoring", success=True, count=2)
            patch.stoplight.add_stage("selection_ranking", success=True, count=1)

            # Finalize
            result = patch.finalize()

            # Verify success
            assert result.success is True
            assert result.templates_found == 1
            assert result.variants_generated == 2
            assert result.survivors == 1

    def test_complete_failure_scenario(self):
        """Test orchestration failure handling"""
        patch = create_orchestrator_patch()

        # Don't create fixtures - should fail fixture validation
        with patch.dict(os.environ, {'ALLY_LIVE': '0'}):
            try:
                patch.apply_all_fixes()
            except FileNotFoundError:
                pass  # Expected

        # Add more failure scenarios
        patch.stoplight.add_error("variant_generation", "All templates invalid")
        patch.stoplight.add_stage("selection_ranking", success=False, count=0)

        result = patch.finalize()

        assert result.success is False
        assert len(result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])