#!/usr/bin/env python3
"""
Surgical fixes for autonomous algo orchestrator debugging

Based on error analysis from orchestrator logs showing:
1. QuantConnect config errors in dry-run mode
2. "Unknown template" variant generation failures
3. Missing receipts and tracebacks for debugging

This module provides drop-in fixes for the orchestration pipeline.
"""

import os
import logging
import traceback
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timezone

# Assuming these exist in the codebase - adjust imports as needed
try:
    from ..utils.receipts import emit_receipt
except ImportError:
    def emit_receipt(**kwargs):
        """Fallback receipt emitter"""
        print(f"RECEIPT: {kwargs}")


logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result of orchestration run with detailed diagnostics"""
    success: bool
    templates_found: int
    variants_generated: int
    hypotheses_scored: int
    survivors: int
    errors: List[str]
    receipts: List[Dict[str, Any]]
    duration_seconds: float


class QuantConnectDryRunGuard:
    """
    Fix #1: Make QuantConnect completely optional in dry runs

    Usage:
        with QuantConnectDryRunGuard(dry_run=True):
            # Any QuantConnect code here will be bypassed
            config = get_quantconnect_config()  # Returns simulator config
    """

    def __init__(self, dry_run: bool = None):
        self.dry_run = dry_run if dry_run is not None else os.getenv("ALLY_LIVE") != "1"
        self.original_modules = {}

    def __enter__(self):
        if self.dry_run:
            # Mock QuantConnect imports to prevent config errors
            self._patch_quantconnect_imports()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dry_run:
            self._restore_quantconnect_imports()

    def _patch_quantconnect_imports(self):
        """Replace QuantConnect modules with simulators"""
        import sys

        class MockQuantConnectConfig:
            """Simulator config that never fails"""
            def __init__(self, *args, **kwargs):
                self.data = {"mode": "simulator", "live": False}

            def get(self, key, default=None):
                return self.data.get(key, default)

            def __getattr__(self, name):
                return getattr(self.data, name, None)

        # Store original modules if they exist
        qc_modules = ['quantconnect', 'QuantConnect']
        for module_name in qc_modules:
            if module_name in sys.modules:
                self.original_modules[module_name] = sys.modules[module_name]

        # Inject mock config
        sys.modules['quantconnect'] = type('MockModule', (), {
            'QuantConnectConfig': MockQuantConnectConfig
        })()

    def _restore_quantconnect_imports(self):
        """Restore original QuantConnect modules"""
        import sys
        for module_name, original_module in self.original_modules.items():
            sys.modules[module_name] = original_module


class VariantGenerationWrapper:
    """
    Fix #2: Add hard-fail receipts + full tracebacks for variant generation

    Usage:
        wrapper = VariantGenerationWrapper()
        variants = wrapper.safe_expand(template, generator)
    """

    def __init__(self, emit_receipts: bool = True):
        self.emit_receipts = emit_receipts
        self.errors = []

    def safe_expand(self, template, generator, spec_hash: str = None) -> List[Any]:
        """
        Safely expand template variants with full error tracking

        Args:
            template: Hypothesis template object
            generator: Variant generator instance
            spec_hash: Optional hash of spec for receipts

        Returns:
            List of generated variants (empty if failed)
        """
        start_time = time.time()
        template_name = template.__class__.__name__

        try:
            logger.debug(f"Expanding variants for {template_name}")
            variants = generator.expand(template)

            # Success receipt
            if self.emit_receipts:
                emit_receipt(
                    tool="research.variant.expand",
                    ok=True,
                    template=template_name,
                    variants_count=len(variants),
                    spec_hash=spec_hash or "unknown",
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            logger.info(f"Generated {len(variants)} variants for {template_name}")
            return variants

        except Exception as e:
            # Capture full traceback
            tb_str = traceback.format_exc()
            error_msg = f"Variant expansion failed for {template_name}: {str(e)}"

            self.errors.append({
                "template": template_name,
                "error": str(e),
                "traceback": tb_str,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            # Log with full details
            logger.exception(f"variant_expand_failed for {template_name}", extra={
                "template": template_name,
                "error": str(e),
                "spec_hash": spec_hash,
                "duration_ms": int((time.time() - start_time) * 1000)
            })

            # Failure receipt
            if self.emit_receipts:
                emit_receipt(
                    tool="research.variant.expand",
                    ok=False,
                    error=str(e),
                    template=template_name,
                    spec_hash=spec_hash or "unknown",
                    traceback_lines=len(tb_str.split('\n')),
                    duration_ms=int((time.time() - start_time) * 1000)
                )

            return []


class OfflineFixtureValidator:
    """
    Fix #3: Enforce offline fixtures for research scoring in dry mode

    Usage:
        validator = OfflineFixtureValidator()
        validator.assert_fixtures_present()
    """

    REQUIRED_FIXTURES = [
        "data/fixtures/factor_panels.json",
        "data/fixtures/synthetic_ohlcv.json",
        "data/fixtures/walk_forward_results.json",
        "artifacts/scoring/offline_benchmarks.json"
    ]

    def __init__(self, fixture_root: str = "."):
        self.fixture_root = fixture_root
        self.missing_fixtures = []

    def assert_fixtures_present(self, required: List[str] = None) -> bool:
        """
        Assert all required fixtures exist for offline scoring

        Args:
            required: Optional custom list of required fixtures

        Returns:
            True if all present, False otherwise

        Raises:
            FileNotFoundError: If fixtures missing and not in dry mode
        """
        required = required or self.REQUIRED_FIXTURES
        self.missing_fixtures = []

        for fixture_path in required:
            full_path = os.path.join(self.fixture_root, fixture_path)
            if not os.path.exists(full_path):
                self.missing_fixtures.append(fixture_path)

        if self.missing_fixtures:
            error_msg = f"Missing offline fixtures: {self.missing_fixtures}"
            logger.error(error_msg)

            # Emit receipt for missing fixtures
            emit_receipt(
                tool="research.fixture.validation",
                ok=False,
                error=error_msg,
                missing_count=len(self.missing_fixtures),
                missing_fixtures=self.missing_fixtures
            )

            if os.getenv("ALLY_LIVE") != "1":
                raise FileNotFoundError(error_msg)
            return False

        logger.info(f"All {len(required)} fixtures present for offline scoring")
        emit_receipt(
            tool="research.fixture.validation",
            ok=True,
            fixtures_validated=len(required)
        )
        return True

    def create_minimal_fixtures(self) -> Dict[str, str]:
        """
        Create minimal fixtures for testing

        Returns:
            Dict mapping fixture paths to their locations
        """
        fixtures_created = {}

        for fixture_path in self.REQUIRED_FIXTURES:
            full_path = os.path.join(self.fixture_root, fixture_path)

            if not os.path.exists(full_path):
                # Create directory if needed
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                # Create minimal fixture based on type
                if "factor_panels" in fixture_path:
                    content = {"factors": ["momentum", "value"], "timestamps": ["2024-01-01"]}
                elif "synthetic_ohlcv" in fixture_path:
                    content = {"AAPL": {"open": [100], "high": [101], "low": [99], "close": [100.5]}}
                elif "walk_forward" in fixture_path:
                    content = {"periods": [{"start": "2024-01-01", "score": 0.65}]}
                else:
                    content = {"fixture_type": "minimal", "created": datetime.now().isoformat()}

                with open(full_path, 'w') as f:
                    import json
                    json.dump(content, f, indent=2)

                fixtures_created[fixture_path] = full_path
                logger.info(f"Created minimal fixture: {full_path}")

        return fixtures_created


class TemplateSchemaValidator:
    """
    Fix #4: Validate hypothesis template schema before expansion

    Usage:
        validator = TemplateSchemaValidator()
        errors = validator.validate_template(template)
    """

    def __init__(self):
        self.validation_errors = []

    def validate_template(self, template) -> List[str]:
        """
        Validate template schema with clear error messages

        Args:
            template: Hypothesis template object

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        template_name = template.__class__.__name__

        try:
            # Basic attribute validation
            required_attrs = ['name', 'description', 'parameters']
            for attr in required_attrs:
                if not hasattr(template, attr):
                    errors.append(f"{template_name} missing required attribute: {attr}")

            # Parameter validation
            if hasattr(template, 'parameters'):
                params = template.parameters
                if not isinstance(params, dict):
                    errors.append(f"{template_name}.parameters must be dict, got {type(params)}")
                else:
                    # Check for required parameter fields
                    for param_name, param_config in params.items():
                        if not isinstance(param_config, dict):
                            errors.append(f"{template_name}.parameters[{param_name}] must be dict")
                        elif 'type' not in param_config:
                            errors.append(f"{template_name}.parameters[{param_name}] missing 'type' field")

            # Template-specific validation
            if 'Momentum' in template_name:
                self._validate_momentum_template(template, errors)
            elif 'MeanReversion' in template_name:
                self._validate_mean_reversion_template(template, errors)

        except Exception as e:
            errors.append(f"Schema validation failed for {template_name}: {str(e)}")

        # Emit validation receipt
        emit_receipt(
            tool="research.template.validate",
            ok=len(errors) == 0,
            template=template_name,
            error_count=len(errors),
            errors=errors[:5]  # Limit errors in receipt
        )

        if errors:
            logger.error(f"Template validation failed for {template_name}: {errors}")
        else:
            logger.debug(f"Template validation passed for {template_name}")

        return errors

    def _validate_momentum_template(self, template, errors: List[str]):
        """Validate momentum-specific fields"""
        if hasattr(template, 'lookback_periods'):
            if not isinstance(template.lookback_periods, (list, tuple)):
                errors.append("MomentumTemplate.lookback_periods must be list/tuple")

    def _validate_mean_reversion_template(self, template, errors: List[str]):
        """Validate mean reversion-specific fields"""
        if hasattr(template, 'reversion_threshold'):
            if not isinstance(template.reversion_threshold, (int, float)):
                errors.append("MeanReversionTemplate.reversion_threshold must be numeric")


class OrchestrationStoplightReceipt:
    """
    Fix #5: Add stoplight summary receipt at end of orchestration

    Usage:
        stoplight = OrchestrationStoplightReceipt()
        stoplight.add_stage("template_discovery", success=True, count=5)
        stoplight.emit_final_receipt()
    """

    def __init__(self):
        self.stages = {}
        self.start_time = time.time()
        self.errors = []

    def add_stage(self, stage_name: str, success: bool, count: int = 0,
                  details: Dict[str, Any] = None):
        """Record stage results"""
        self.stages[stage_name] = {
            "success": success,
            "count": count,
            "details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def add_error(self, stage_name: str, error: str):
        """Add error for a stage"""
        self.errors.append({"stage": stage_name, "error": error})

    def emit_final_receipt(self) -> OrchestrationResult:
        """Emit comprehensive orchestration summary receipt"""
        duration = time.time() - self.start_time

        # Calculate totals
        templates = self.stages.get("template_discovery", {}).get("count", 0)
        variants = self.stages.get("variant_generation", {}).get("count", 0)
        scored = self.stages.get("hypothesis_scoring", {}).get("count", 0)
        survivors = self.stages.get("selection_ranking", {}).get("count", 0)

        overall_success = all(stage["success"] for stage in self.stages.values())

        # Create result object
        result = OrchestrationResult(
            success=overall_success,
            templates_found=templates,
            variants_generated=variants,
            hypotheses_scored=scored,
            survivors=survivors,
            errors=[e["error"] for e in self.errors],
            receipts=[],  # Would be populated by actual receipt system
            duration_seconds=duration
        )

        # Emit detailed receipt
        receipt_data = {
            "tool": "research.orchestration.summary",
            "ok": overall_success,
            "duration_seconds": duration,
            "pipeline_counts": {
                "templates": templates,
                "variants": variants,
                "scored": scored,
                "survivors": survivors
            },
            "stages": self.stages,
            "error_count": len(self.errors),
            "errors": self.errors[:10]  # Limit errors in receipt
        }

        emit_receipt(**receipt_data)

        # Log summary
        status = "SUCCESS" if overall_success else "FAILED"
        logger.info(f"🚦 Orchestration {status}: {templates}→{variants}→{scored}→{survivors} "
                   f"({duration:.1f}s, {len(self.errors)} errors)")

        return result


def create_orchestrator_patch():
    """
    Factory function to create a comprehensive orchestrator patch

    Usage:
        # At the start of orchestration main():
        patch = create_orchestrator_patch()
        patch.apply_all_fixes()

        # Your existing orchestration logic here

        result = patch.finalize()
    """

    class OrchestratorPatch:
        def __init__(self):
            self.qc_guard = QuantConnectDryRunGuard()
            self.variant_wrapper = VariantGenerationWrapper()
            self.fixture_validator = OfflineFixtureValidator()
            self.schema_validator = TemplateSchemaValidator()
            self.stoplight = OrchestrationStoplightReceipt()

        def apply_all_fixes(self):
            """Apply all surgical fixes"""
            # Fix #1: QuantConnect dry-run guard
            self.qc_guard.__enter__()

            # Fix #3: Validate fixtures
            try:
                self.fixture_validator.assert_fixtures_present()
                self.stoplight.add_stage("fixture_validation", True,
                                       len(OfflineFixtureValidator.REQUIRED_FIXTURES))
            except FileNotFoundError as e:
                self.stoplight.add_stage("fixture_validation", False, 0)
                self.stoplight.add_error("fixture_validation", str(e))

        def safe_template_expand(self, templates, generator):
            """Process templates with all safety checks"""
            all_variants = []
            valid_templates = 0

            for template in templates:
                # Fix #4: Schema validation
                schema_errors = self.schema_validator.validate_template(template)
                if schema_errors:
                    self.stoplight.add_error("schema_validation",
                                           f"{template.__class__.__name__}: {schema_errors}")
                    continue

                valid_templates += 1

                # Fix #2: Safe variant generation
                variants = self.variant_wrapper.safe_expand(template, generator)
                all_variants.extend(variants)

            self.stoplight.add_stage("template_discovery", True, len(templates))
            self.stoplight.add_stage("schema_validation", True, valid_templates)
            self.stoplight.add_stage("variant_generation", True, len(all_variants))

            return all_variants

        def finalize(self) -> OrchestrationResult:
            """Clean up and emit final receipts"""
            # Fix #1: Cleanup QuantConnect guard
            self.qc_guard.__exit__(None, None, None)

            # Fix #5: Emit stoplight summary
            return self.stoplight.emit_final_receipt()

    return OrchestratorPatch()


# Example usage/integration template
def fixed_orchestration_main():
    """
    Example of how to integrate these fixes into existing orchestration
    """
    logger.info("🚀 Starting Autonomous Algo Strategy Developer (FIXED)")

    # Apply all surgical fixes
    patch = create_orchestrator_patch()
    patch.apply_all_fixes()

    try:
        # Your existing template discovery logic
        # templates = discover_hypothesis_templates()
        # patch.stoplight.add_stage("template_discovery", True, len(templates))

        # Your existing variant generation with safety wrapper
        # variants = patch.safe_template_expand(templates, variant_generator)

        # Your existing scoring logic
        # scored_hypotheses = score_hypotheses_offline(variants)
        # patch.stoplight.add_stage("hypothesis_scoring", True, len(scored_hypotheses))

        # Your existing selection logic
        # survivors = select_top_hypotheses(scored_hypotheses, budget=3)
        # patch.stoplight.add_stage("selection_ranking", True, len(survivors))

        pass  # Replace with actual orchestration logic

    except Exception as e:
        logger.exception("Orchestration failed")
        patch.stoplight.add_error("orchestration", str(e))

    finally:
        # Always emit final results
        result = patch.finalize()

        if result.success:
            logger.info(f"✅ Orchestration completed: {result.survivors} strategies generated")
        else:
            logger.error(f"❌ Orchestration failed: {len(result.errors)} errors")

        return result


if __name__ == "__main__":
    # Test the fixes
    result = fixed_orchestration_main()
    print(f"Result: {result}")