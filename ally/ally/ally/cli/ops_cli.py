#!/usr/bin/env python3
"""
Ops CLI for drift detection and promotion guards - Phase 8

Provides command-line interface for all drift detection and promotion guard operations.
"""

import typer
import json
from typing import Optional
from pathlib import Path

# Import ops modules
try:
    from ally.ops.drift_data import ops_drift_data
    from ally.ops.drift_strategy import ops_drift_strategy
    from ally.ops.drift_ops import ops_drift_ops
    from ally.ops.promotion_guard import ops_promote_guard
    from ally.ops.heartbeat import ops_heartbeat
    OPS_MODULES_AVAILABLE = True
except ImportError:
    OPS_MODULES_AVAILABLE = False

# Create ops CLI app
ops_app = typer.Typer(help="Ops commands for drift detection and promotion guards")


@ops_app.command("drift")
def drift_command(
    type: str = typer.Argument(..., help="Drift type: data, strategy, or ops"),
    panel: Optional[str] = typer.Option(None, "--panel", help="Data panel path for data drift"),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="Strategy hash for strategy drift"),
    fixture: Optional[str] = typer.Option(None, "--fixture", help="Fixture path for ops drift"),
    policy: str = typer.Option("ally/ops/policy.yaml", "--policy", help="Policy configuration path"),
    ref_window: Optional[int] = typer.Option(None, "--ref-window", help="Reference window days"),
    test_window: Optional[int] = typer.Option(None, "--test-window", help="Test window days"),
    psi_thresh: Optional[float] = typer.Option(None, "--psi-thresh", help="PSI threshold"),
    window: Optional[int] = typer.Option(None, "--window", help="Rolling window days"),
    te_max: Optional[float] = typer.Option(None, "--te-max", help="Max tracking error"),
    z_min: Optional[float] = typer.Option(None, "--z-min", help="Min Z-score threshold"),
    recon_band: Optional[float] = typer.Option(None, "--recon-band", help="Reconciliation band"),
    days: Optional[int] = typer.Option(None, "--days", help="Min days for reconciliation"),
    repeats: Optional[int] = typer.Option(None, "--repeats", help="Number of repeat runs"),
    live: bool = typer.Option(False, "--live", help="Enable live mode"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format")
):
    """
    Run drift detection

    Examples:
        ally ops drift data --panel artifacts/research/features.parquet --live=false
        ally ops drift strategy --strategy <hash> --window 21 --live=false
        ally ops drift ops --fixture artifacts/fixtures/determinism.pkl --repeats 3 --live=false
    """
    if not OPS_MODULES_AVAILABLE:
        typer.echo("❌ Ops modules not available", err=True)
        raise typer.Exit(code=1)

    try:
        if type == "data":
            result = ops_drift_data(
                panel_path=panel,
                policy_path=policy,
                ref_window=ref_window,
                test_window=test_window,
                psi_thresh=psi_thresh,
                live=live
            )
        elif type == "strategy":
            if not strategy:
                typer.echo("❌ Strategy hash required for strategy drift", err=True)
                raise typer.Exit(code=1)

            result = ops_drift_strategy(
                strategy_hash=strategy,
                policy_path=policy,
                window=window,
                te_max=te_max,
                z_min=z_min,
                recon_band=recon_band,
                days=days,
                live=live
            )
        elif type == "ops":
            result = ops_drift_ops(
                fixture_path=fixture or "artifacts/fixtures/determinism.pkl",
                policy_path=policy,
                repeats=repeats,
                live=live
            )
        else:
            typer.echo(f"❌ Unknown drift type: {type}. Use: data, strategy, or ops", err=True)
            raise typer.Exit(code=1)

        if json_output:
            # Output full JSON result
            typer.echo(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            # Formatted output
            if result.ok:
                status = result.data.get("status", "UNKNOWN")
                status_emoji = "✅" if status == "OK" else "❌"

                typer.echo(f"{status_emoji} {type.title()} drift detection completed")
                typer.echo(f"Status: {status}")
                typer.echo(f"Receipt: {result.data.get('drift_receipt', 'N/A')}")

                if type == "data":
                    typer.echo(f"Features checked: {result.data.get('summary', {}).get('features_checked', 0)}")
                    typer.echo(f"Features with drift: {result.data.get('summary', {}).get('features_with_drift', 0)}")
                    typer.echo(f"Schema OK: {result.data.get('summary', {}).get('schema_ok', False)}")
                elif type == "strategy":
                    typer.echo(f"Tracking error: {result.data.get('tracking_analysis', {}).get('tracking_error', 0):.4f}")
                    typer.echo(f"Min Z-score: {result.data.get('zscore_analysis', {}).get('min_zscore', 0):.2f}")
                    typer.echo(f"Reconciliation OK: {result.data.get('reconciliation_analysis', {}).get('recon_pass', False)}")
                elif type == "ops":
                    typer.echo(f"Deterministic: {result.data.get('summary', {}).get('deterministic', False)}")
                    typer.echo(f"PSD compliant: {result.data.get('summary', {}).get('psd_compliant', False)}")

                violations = result.data.get("violations", [])
                if violations:
                    typer.echo(f"Violations ({len(violations)}):")
                    for violation in violations[:5]:  # Show first 5
                        typer.echo(f"  - {violation}")
                    if len(violations) > 5:
                        typer.echo(f"  ... and {len(violations) - 5} more")
            else:
                typer.echo("❌ Drift detection failed", err=True)
                for error in result.errors:
                    typer.echo(f"Error: {error}", err=True)
                raise typer.Exit(code=1)

    except Exception as e:
        typer.echo(f"❌ Drift detection failed: {str(e)}", err=True)
        raise typer.Exit(code=1)


@ops_app.command("promote-guard")
def promote_guard_command(
    bundle: str = typer.Argument(..., help="Bundle SHA1 to evaluate"),
    policy: str = typer.Option("ally/ops/policy.yaml", "--policy", help="Policy configuration path"),
    strategy: Optional[str] = typer.Option(None, "--strategy", help="Strategy hash for drift check"),
    panel: Optional[str] = typer.Option(None, "--panel", help="Data panel path"),
    fixture: Optional[str] = typer.Option(None, "--fixture", help="Ops fixture path"),
    live: bool = typer.Option(False, "--live", help="Enable live mode"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format")
):
    """
    Run promotion guard to check if bundle can be promoted

    Examples:
        ally ops promote-guard <bundle_sha1> --live=false
        ally ops promote-guard <bundle_sha1> --strategy <hash> --live=false
    """
    if not OPS_MODULES_AVAILABLE:
        typer.echo("❌ Ops modules not available", err=True)
        raise typer.Exit(code=1)

    try:
        result = ops_promote_guard(
            bundle_sha1=bundle,
            policy_path=policy,
            strategy_hash=strategy,
            panel_path=panel,
            fixture_path=fixture,
            live=live
        )

        if json_output:
            # Output full JSON result
            typer.echo(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            # Formatted output
            decision = result.data.get("promotion_decision", "UNKNOWN")
            decision_emoji = "✅" if decision == "ALLOW" else "❌"

            typer.echo(f"{decision_emoji} Promotion guard: {decision}")
            typer.echo(f"Bundle: {result.data.get('bundle_sha1', 'unknown')}")
            typer.echo(f"Receipt: {result.data.get('guard_receipt', 'N/A')}")

            summary = result.data.get("guard_summary", {})
            typer.echo(f"Sentinels OK: {summary.get('sentinels_ok', 0)}/{summary.get('total_sentinels', 0)}")

            if decision == "BLOCK":
                typer.echo("Blocking reasons:")
                for reason in result.errors[:5]:  # Show first 5
                    typer.echo(f"  - {reason}")
                if len(result.errors) > 5:
                    typer.echo(f"  ... and {len(result.errors) - 5} more")

                failed_sentinels = summary.get("sentinels_failed", 0)
                error_sentinels = summary.get("sentinels_error", 0)

                if failed_sentinels > 0:
                    typer.echo(f"Failed sentinels: {failed_sentinels}")
                if error_sentinels > 0:
                    typer.echo(f"Error sentinels: {error_sentinels}")

        # Exit with error code if promotion is blocked
        if not result.ok:
            raise typer.Exit(code=1)

    except Exception as e:
        typer.echo(f"❌ Promotion guard failed: {str(e)}", err=True)
        raise typer.Exit(code=1)


@ops_app.command("heartbeat")
def heartbeat_command(
    since: str = typer.Option("24h", "--since", help="Time period to analyze (e.g., 24h, 48h)"),
    policy: str = typer.Option("ally/ops/policy.yaml", "--policy", help="Policy configuration path"),
    details: bool = typer.Option(True, "--details/--no-details", help="Include detailed analysis"),
    live: bool = typer.Option(False, "--live", help="Enable live mode"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON format")
):
    """
    Generate system heartbeat and status report

    Examples:
        ally ops heartbeat --since 24h --live=false
        ally ops heartbeat --since 48h --details --live=false
    """
    if not OPS_MODULES_AVAILABLE:
        typer.echo("❌ Ops modules not available", err=True)
        raise typer.Exit(code=1)

    try:
        # Parse time period
        if since.endswith('h'):
            since_hours = int(since[:-1])
        else:
            since_hours = 24  # Default

        result = ops_heartbeat(
            since_hours=since_hours,
            policy_path=policy,
            include_details=details,
            live=live
        )

        if json_output:
            # Output full JSON result
            typer.echo(json.dumps(result.model_dump(), indent=2, default=str))
        else:
            # Formatted output
            if result.ok:
                snapshot = result.data.get("snapshot", {})
                system_status = snapshot.get("system_status", "UNKNOWN")
                health_score = snapshot.get("health_score", 0)

                status_emoji = "✅" if system_status == "HEALTHY" else "⚠️" if "DEGRADED" in system_status else "❌"

                typer.echo(f"{status_emoji} System Heartbeat Report")
                typer.echo(f"Status: {system_status}")
                typer.echo(f"Health Score: {health_score:.1f}/100")
                typer.echo(f"Analysis Period: {snapshot.get('analysis_period', 'unknown')}")
                typer.echo(f"Receipt: {result.data.get('heartbeat_receipt', 'N/A')}")

                drift_analysis = result.data.get("drift_analysis", {})
                portfolio_analysis = result.data.get("portfolio_analysis", {})

                typer.echo(f"\n📊 Drift Status: {drift_analysis.get('overall_health', 'UNKNOWN')}")
                typer.echo(f"Tools with issues: {drift_analysis.get('tools_with_issues', 0)}")
                typer.echo(f"Total drift checks: {drift_analysis.get('total_drift_checks', 0)}")

                typer.echo(f"\n📈 Portfolio Performance:")
                typer.echo(f"Optimizations: {portfolio_analysis.get('optimizations_count', 0)}")
                typer.echo(f"Constraint violations: {portfolio_analysis.get('constraints_violations', 0)}")

                if portfolio_analysis.get('avg_sharpe'):
                    typer.echo(f"Avg Sharpe: {portfolio_analysis['avg_sharpe']:.3f}")
                if portfolio_analysis.get('avg_volatility'):
                    typer.echo(f"Avg Volatility: {portfolio_analysis['avg_volatility']:.3f}")

                daily_narrative = result.data.get("daily_narrative", {})
                action_items = daily_narrative.get("action_items", [])

                if action_items:
                    typer.echo(f"\n🔧 Action Items:")
                    for item in action_items:
                        typer.echo(f"  - {item}")

                typer.echo(f"\n🕐 Next Review: {daily_narrative.get('next_review', 'unknown')}")
            else:
                typer.echo("❌ Heartbeat generation failed", err=True)
                for error in result.errors:
                    typer.echo(f"Error: {error}", err=True)
                raise typer.Exit(code=1)

    except Exception as e:
        typer.echo(f"❌ Heartbeat generation failed: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    ops_app()