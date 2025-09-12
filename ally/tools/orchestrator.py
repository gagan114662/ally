from ally.orchestrator.run import OrchestrationEngine
from ally.schemas.orch import OrchInput
from ally.schemas.base import ToolResult
from ally.tools import register


@register("orchestrator.run")
def orchestrator_run(
    experiment_id: str,
    symbols: list[str] = None,
    interval: str = "1h",
    lookback: int = 600,
    targets: dict = None,
    risk_policy_yaml: str = None,
    save_run: bool = True,
    make_report: bool = True,
    seed: int = 42
) -> ToolResult:
    """
    Execute the full research → backtest → risk/exec → memory → report pipeline.
    
    This is the master orchestrator that coordinates all Ally tools into a complete
    trading research and execution workflow.
    
    Args:
        experiment_id: Unique identifier for this experiment
        symbols: List of trading symbols (default: ["BTCUSDT"])
        interval: Time interval for analysis (default: "1h")
        lookback: Number of periods to look back (default: 600)
        targets: Performance targets dict (default: {"annual_return": 0.10, "sharpe_ratio": 1.0})
        risk_policy_yaml: YAML risk policy string (default: basic policy)
        save_run: Whether to save results to memory (default: True)
        make_report: Whether to generate HTML report (default: True)
        seed: Random seed for deterministic results (default: 42)
        
    Returns:
        ToolResult containing OrchSummary with experiment results
        
    Raises:
        Orchestration errors are caught and returned as ToolResult.error
    """
    
    # Set defaults
    if symbols is None:
        symbols = ["BTCUSDT"]
    if targets is None:
        targets = {"annual_return": 0.10, "sharpe_ratio": 1.0}
    if risk_policy_yaml is None:
        risk_policy_yaml = "max_leverage: 3.0\nmax_single_order_notional: 25000"
    
    # Create input configuration
    config = OrchInput(
        experiment_id=experiment_id,
        symbols=symbols,
        interval=interval,
        lookback=lookback,
        targets=targets,
        risk_policy_yaml=risk_policy_yaml,
        save_run=save_run,
        make_report=make_report
    )
    
    # Run orchestration
    engine = OrchestrationEngine(seed=seed)
    return engine.run_pipeline(config)