#!/usr/bin/env python3
"""
Research CLI commands for orchestrator diagnostics and health checks

Provides 'ally research doctor' command for pre-flight checks before
running autonomous algorithm research loops.
"""

import os
import json
import click
from pathlib import Path
from datetime import datetime, timezone

from ..research.orchestrator_fixes import OfflineFixtureValidator

# Fallback receipt emitter if main utils not available
try:
    from ..utils.receipts import emit_receipt
except ImportError:
    def emit_receipt(**kwargs):
        """Fallback receipt emitter"""
        print(f"RECEIPT: {kwargs}")


@click.group()
def research():
    """Research orchestrator commands"""
    pass


@research.command()
@click.option('--fixtures', default="data/fixtures", help="Fixtures directory to validate")
@click.option('--json-output', is_flag=True, help="Output results as JSON")
@click.option('--create-missing', is_flag=True, help="Auto-create minimal fixtures if missing")
def doctor(fixtures: str, json_output: bool, create_missing: bool):
    """
    Run research orchestrator health check

    Validates all required fixtures, schema definitions, and dependencies
    needed for autonomous algorithm research loops.
    """
    click.echo("🩺 Ally Research Doctor - Pre-flight Health Check")
    click.echo(f"   Fixtures directory: {fixtures}")
    click.echo(f"   ALLY_LIVE: {os.getenv('ALLY_LIVE', '0')}")
    click.echo()

    validator = OfflineFixtureValidator(fixture_root=".")

    # Check fixture health
    try:
        fixtures_ok = validator.assert_fixtures_present()
        fixture_status = "✅ HEALTHY"
        fixture_details = f"All {len(OfflineFixtureValidator.REQUIRED_FIXTURES)} fixtures present"
    except FileNotFoundError as e:
        fixtures_ok = False
        fixture_status = "❌ MISSING FIXTURES"
        fixture_details = str(e)

        if create_missing:
            click.echo("🔧 Auto-creating minimal fixtures...")
            created = validator.create_minimal_fixtures()
            click.echo(f"   Created {len(created)} fixtures")
            fixtures_ok = True
            fixture_status = "✅ FIXED (created minimal fixtures)"
            fixture_details = f"Created: {list(created.keys())}"

    # Check dependencies
    try:
        import typer, duckdb
        deps_ok = True
        deps_status = "✅ HEALTHY"
        deps_details = "Core dependencies available"
    except ImportError as e:
        deps_ok = False
        deps_status = "❌ MISSING DEPS"
        deps_details = f"Import failed: {e}"

    # Check CI environment
    is_ci = os.getenv("ALLY_LIVE") == "0" or os.getenv("CI") == "true"
    ci_status = "✅ DETERMINISTIC" if is_ci else "⚠️ LIVE MODE"
    ci_details = f"ALLY_LIVE={os.getenv('ALLY_LIVE', '0')}, CI={os.getenv('CI', 'false')}"

    # Overall health
    overall_ok = fixtures_ok and deps_ok
    overall_status = "✅ READY FOR RESEARCH" if overall_ok else "❌ NOT READY"

    # Display results
    if not json_output:
        click.echo("📋 Health Check Results:")
        click.echo(f"   Fixtures: {fixture_status}")
        click.echo(f"   Dependencies: {deps_status}")
        click.echo(f"   Environment: {ci_status}")
        click.echo()
        click.echo(f"🎯 Overall: {overall_status}")

        if not overall_ok:
            click.echo()
            click.echo("🔧 Recommended actions:")
            if not fixtures_ok:
                click.echo(f"   - Run with --create-missing to auto-create fixtures")
                click.echo(f"   - Or manually create: {validator.missing_fixtures}")
            if not deps_ok:
                click.echo(f"   - Install missing dependencies: {deps_details}")

    # Create diagnostic report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "overall_ok": overall_ok,
        "checks": {
            "fixtures": {
                "status": fixture_status,
                "ok": fixtures_ok,
                "details": fixture_details,
                "missing": validator.missing_fixtures
            },
            "dependencies": {
                "status": deps_status,
                "ok": deps_ok,
                "details": deps_details
            },
            "environment": {
                "status": ci_status,
                "ok": is_ci,
                "details": ci_details,
                "ally_live": os.getenv('ALLY_LIVE', '0'),
                "is_ci": os.getenv('CI') == 'true'
            }
        }
    }

    # Save diagnostic report
    os.makedirs("artifacts/research", exist_ok=True)
    report_path = "artifacts/research/doctor_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    if json_output:
        click.echo(json.dumps(report, indent=2))
    else:
        click.echo(f"📄 Full report saved to: {report_path}")

    # Emit receipt for CI verification
    emit_receipt(
        tool="research.doctor",
        ok=overall_ok,
        fixtures_ok=fixtures_ok,
        deps_ok=deps_ok,
        missing_fixtures=len(validator.missing_fixtures),
        environment=ci_status,
        report_path=report_path
    )

    # Exit with error code if unhealthy
    if not overall_ok:
        click.echo()
        raise click.ClickException("Research environment not ready. Run with --create-missing or fix issues above.")

    click.echo("✅ Research environment healthy - ready for orchestration!")


@research.command()
@click.option('--dry-run', is_flag=True, default=True, help="Run in dry mode (default)")
@click.option('--budget', default=3, help="Number of strategies to generate")
def test_loop(dry_run: bool, budget: int):
    """
    Test research orchestrator loop with all fixes applied

    Runs a minimal orchestration with the surgical fixes to verify
    the pipeline works end-to-end without crashing.
    """
    click.echo("🧪 Testing Research Orchestrator Loop")
    click.echo(f"   Budget: {budget} strategies")
    click.echo(f"   Dry run: {dry_run}")
    click.echo()

    from ..research.orchestrator_fixes import create_orchestrator_patch

    # Apply all fixes
    patch = create_orchestrator_patch()
    patch.apply_all_fixes()

    try:
        # Simulate template discovery
        mock_templates = [
            type('MockMomentumTemplate', (), {
                '__class__': type('MomentumTemplate', (), {'__name__': 'MomentumTemplate'}),
                'name': 'momentum_test',
                'description': 'Test momentum strategy',
                'parameters': {'lookback': {'type': 'int', 'default': 20}}
            })(),
            type('MockMeanRevTemplate', (), {
                '__class__': type('MeanReversionTemplate', (), {'__name__': 'MeanReversionTemplate'}),
                'name': 'mean_reversion_test',
                'description': 'Test mean reversion strategy',
                'parameters': {'threshold': {'type': 'float', 'default': 0.1}}
            })()
        ]

        # Simulate variant generator
        class MockGenerator:
            def expand(self, template):
                return [f"variant_{i}" for i in range(2)]  # 2 variants per template

        generator = MockGenerator()

        # Test the pipeline
        patch.stoplight.add_stage("template_discovery", True, len(mock_templates))
        variants = patch.safe_template_expand(mock_templates, generator)

        # Simulate scoring
        scored = [{"variant": v, "score": 0.7} for v in variants[:3]]  # Top 3
        patch.stoplight.add_stage("hypothesis_scoring", True, len(scored))

        # Simulate selection
        survivors = scored[:min(budget, len(scored))]
        patch.stoplight.add_stage("selection_ranking", True, len(survivors))

        click.echo(f"✅ Pipeline Success: {len(mock_templates)}→{len(variants)}→{len(scored)}→{len(survivors)}")

    except Exception as e:
        click.echo(f"❌ Pipeline Failed: {e}")
        patch.stoplight.add_error("test_orchestration", str(e))

    finally:
        # Always finalize and emit results
        result = patch.finalize()

        # Save test results
        os.makedirs("artifacts/research", exist_ok=True)
        test_path = "artifacts/research/test_loop_results.json"
        with open(test_path, 'w') as f:
            json.dump({
                "success": result.success,
                "templates": result.templates_found,
                "variants": result.variants_generated,
                "scored": result.hypotheses_scored,
                "survivors": result.survivors,
                "errors": result.errors,
                "duration": result.duration_seconds
            }, f, indent=2)

        click.echo(f"📄 Results saved to: {test_path}")

        if result.success:
            click.echo("🎉 Test orchestration completed successfully!")
        else:
            click.echo("❌ Test orchestration failed - check diagnostics above")
            raise click.ClickException(f"Orchestration failed with {len(result.errors)} errors")


if __name__ == "__main__":
    research()