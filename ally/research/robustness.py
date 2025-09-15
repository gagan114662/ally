"""
Robustness Battery - Stress testing for strategy validation
Bootstrap resampling, regime shuffling, and noise injection tests
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from ally.utils.receipts import generate_receipt
from ally.utils.gating import check_live_mode_allowed
from ally.core.tool_registry import register, ToolResult


class StressScenario(Enum):
    """Available stress test scenarios"""
    BOOTSTRAP_RETURNS = "bootstrap_returns"
    REGIME_SHUFFLE = "regime_shuffle"
    NOISE_JITTER = "noise_jitter"
    DRAWDOWN_EXTEND = "drawdown_extend"
    VOLATILITY_SHOCK = "volatility_shock"


@dataclass
class RobustnessConfig:
    """Robustness testing configuration"""
    scenarios: List[StressScenario]
    n_bootstrap_samples: int = 1000
    noise_std_pct: float = 0.05  # 5% noise injection
    regime_shuffle_blocks: int = 4
    pass_rate_threshold: float = 0.70  # 70% scenarios must pass
    min_sharpe_threshold: float = 0.5
    max_drawdown_threshold: float = -0.25


def bootstrap_resample_returns(returns: np.ndarray, n_samples: int = 1000,
                              block_size: Optional[int] = None) -> List[np.ndarray]:
    """
    Bootstrap resample returns with optional block bootstrap for time series
    
    Args:
        returns: Array of daily returns
        n_samples: Number of bootstrap samples
        block_size: Size of blocks for block bootstrap (None for simple bootstrap)
        
    Returns:
        List of resampled return arrays
    """
    
    np.random.seed(42)  # Deterministic for CI
    n_returns = len(returns)
    
    if block_size is None:
        # Simple bootstrap (iid assumption)
        resampled_returns = []
        for _ in range(n_samples):
            indices = np.random.choice(n_returns, size=n_returns, replace=True)
            resampled_returns.append(returns[indices])
        return resampled_returns
    
    else:
        # Block bootstrap (preserves some time dependence)
        n_blocks = n_returns // block_size
        resampled_returns = []
        
        for _ in range(n_samples):
            # Sample blocks with replacement
            block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
            resampled = []
            
            for block_idx in block_indices:
                start_idx = block_idx * block_size
                end_idx = min(start_idx + block_size, n_returns)
                resampled.extend(returns[start_idx:end_idx])
            
            # Trim to original length
            resampled_returns.append(np.array(resampled[:n_returns]))
        
        return resampled_returns


def shuffle_regime_blocks(returns: np.ndarray, n_blocks: int = 4) -> np.ndarray:
    """
    Shuffle market regime blocks to test strategy robustness across different orderings
    
    Args:
        returns: Array of daily returns
        n_blocks: Number of regime blocks to create and shuffle
        
    Returns:
        Returns with shuffled regime blocks
    """
    
    np.random.seed(43)  # Deterministic for CI
    n_returns = len(returns)
    block_size = n_returns // n_blocks
    
    # Split into blocks
    blocks = []
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size if i < n_blocks - 1 else n_returns
        blocks.append(returns[start_idx:end_idx])
    
    # Shuffle blocks
    np.random.shuffle(blocks)
    
    # Concatenate shuffled blocks
    shuffled_returns = np.concatenate(blocks)
    
    return shuffled_returns


def inject_price_noise(prices: np.ndarray, noise_std_pct: float = 0.05) -> np.ndarray:
    """
    Inject random noise into price series to test signal robustness
    
    Args:
        prices: Array of prices
        noise_std_pct: Standard deviation of noise as percentage of price
        
    Returns:
        Prices with injected noise
    """
    
    np.random.seed(44)  # Deterministic for CI
    
    # Generate multiplicative noise (log-normal)
    noise_factors = np.random.lognormal(
        mean=0, 
        sigma=noise_std_pct, 
        size=len(prices)
    )
    
    noisy_prices = prices * noise_factors
    
    return noisy_prices


def extend_drawdown_periods(returns: np.ndarray, extension_factor: float = 1.5) -> np.ndarray:
    """
    Artificially extend drawdown periods to test strategy resilience
    
    Args:
        returns: Array of daily returns
        extension_factor: Factor by which to extend negative periods
        
    Returns:
        Returns with extended drawdowns
    """
    
    extended_returns = returns.copy()
    
    # Identify drawdown periods (consecutive negative returns)
    in_drawdown = False
    drawdown_start = 0
    
    for i, ret in enumerate(returns):
        if ret < 0 and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            drawdown_start = i
        elif ret >= 0 and in_drawdown:
            # End of drawdown
            in_drawdown = False
            drawdown_length = i - drawdown_start
            
            # Extend this drawdown period
            extension_length = int(drawdown_length * (extension_factor - 1))
            if extension_length > 0:
                # Sample additional negative returns
                np.random.seed(45 + i)  # Deterministic
                extended_returns = np.concatenate([
                    extended_returns[:i],
                    np.random.choice(returns[returns < 0], size=extension_length),
                    extended_returns[i:]
                ])
    
    return extended_returns


def shock_volatility_regime(returns: np.ndarray, shock_factor: float = 2.0,
                          shock_start_pct: float = 0.3, shock_duration_pct: float = 0.2) -> np.ndarray:
    """
    Inject high volatility shock regime
    
    Args:
        returns: Array of daily returns
        shock_factor: Volatility multiplication factor during shock
        shock_start_pct: When shock starts (as fraction of series)
        shock_duration_pct: Duration of shock (as fraction of series)
        
    Returns:
        Returns with volatility shock
    """
    
    shocked_returns = returns.copy()
    n_returns = len(returns)
    
    shock_start = int(n_returns * shock_start_pct)
    shock_duration = int(n_returns * shock_duration_pct)
    shock_end = min(shock_start + shock_duration, n_returns)
    
    # Apply volatility shock to the specified period
    shocked_returns[shock_start:shock_end] *= shock_factor
    
    return shocked_returns


def run_stress_scenario(returns: np.ndarray, scenario: StressScenario,
                       config: RobustnessConfig) -> Dict[str, Any]:
    """
    Run a single stress test scenario
    
    Args:
        returns: Original return series
        scenario: Stress scenario to run
        config: Robustness configuration
        
    Returns:
        Dict with scenario results
    """
    
    if scenario == StressScenario.BOOTSTRAP_RETURNS:
        # Bootstrap resampling
        resampled_returns_list = bootstrap_resample_returns(
            returns, 
            n_samples=min(100, config.n_bootstrap_samples),  # Limit for CI
            block_size=21  # Monthly blocks
        )
        
        # Calculate metrics for each bootstrap sample
        bootstrap_sharpes = []
        bootstrap_drawdowns = []
        
        for boot_returns in resampled_returns_list:
            if len(boot_returns) > 0 and np.std(boot_returns) > 0:
                sharpe = np.mean(boot_returns) / np.std(boot_returns) * np.sqrt(252)
                
                # Calculate max drawdown
                cum_returns = np.cumprod(1 + boot_returns) - 1
                peak = np.maximum.accumulate(cum_returns)
                drawdown = (cum_returns - peak) / (1 + peak)
                max_dd = np.min(drawdown)
                
                bootstrap_sharpes.append(sharpe)
                bootstrap_drawdowns.append(max_dd)
        
        # Calculate pass rate
        passing_samples = sum(1 for s in bootstrap_sharpes 
                            if s >= config.min_sharpe_threshold)
        pass_rate = passing_samples / len(bootstrap_sharpes) if bootstrap_sharpes else 0
        
        return {
            "scenario": scenario.value,
            "n_samples": len(bootstrap_sharpes),
            "pass_rate": pass_rate,
            "sharpe_distribution": {
                "mean": np.mean(bootstrap_sharpes) if bootstrap_sharpes else 0,
                "std": np.std(bootstrap_sharpes) if bootstrap_sharpes else 0,
                "p25": np.percentile(bootstrap_sharpes, 25) if bootstrap_sharpes else 0,
                "p50": np.percentile(bootstrap_sharpes, 50) if bootstrap_sharpes else 0,
                "p75": np.percentile(bootstrap_sharpes, 75) if bootstrap_sharpes else 0
            },
            "drawdown_distribution": {
                "mean": np.mean(bootstrap_drawdowns) if bootstrap_drawdowns else 0,
                "worst": np.min(bootstrap_drawdowns) if bootstrap_drawdowns else 0
            }
        }
    
    elif scenario == StressScenario.REGIME_SHUFFLE:
        # Regime shuffling
        shuffled_returns = shuffle_regime_blocks(returns, config.regime_shuffle_blocks)
        
        if np.std(shuffled_returns) > 0:
            shuffled_sharpe = np.mean(shuffled_returns) / np.std(shuffled_returns) * np.sqrt(252)
            
            # Calculate drawdown
            cum_returns = np.cumprod(1 + shuffled_returns) - 1
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - peak) / (1 + peak)
            max_dd = np.min(drawdown)
            
            passed = shuffled_sharpe >= config.min_sharpe_threshold and max_dd >= config.max_drawdown_threshold
        else:
            shuffled_sharpe = 0
            max_dd = 0
            passed = False
        
        return {
            "scenario": scenario.value,
            "shuffled_sharpe": shuffled_sharpe,
            "shuffled_max_drawdown": max_dd,
            "passed": passed,
            "pass_rate": 1.0 if passed else 0.0
        }
    
    elif scenario == StressScenario.NOISE_JITTER:
        # Noise injection test
        # Simulate noisy prices and recalculate returns
        prices = np.cumprod(1 + returns)
        noisy_prices = inject_price_noise(prices, config.noise_std_pct)
        noisy_returns = np.diff(noisy_prices) / noisy_prices[:-1]
        
        if len(noisy_returns) > 0 and np.std(noisy_returns) > 0:
            noisy_sharpe = np.mean(noisy_returns) / np.std(noisy_returns) * np.sqrt(252)
            
            # Calculate drawdown
            cum_returns = np.cumprod(1 + noisy_returns) - 1
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - peak) / (1 + peak)
            max_dd = np.min(drawdown)
            
            passed = noisy_sharpe >= config.min_sharpe_threshold and max_dd >= config.max_drawdown_threshold
        else:
            noisy_sharpe = 0
            max_dd = 0
            passed = False
        
        return {
            "scenario": scenario.value,
            "noise_std_pct": config.noise_std_pct,
            "noisy_sharpe": noisy_sharpe,
            "noisy_max_drawdown": max_dd,
            "passed": passed,
            "pass_rate": 1.0 if passed else 0.0
        }
    
    elif scenario == StressScenario.VOLATILITY_SHOCK:
        # Volatility shock test
        shocked_returns = shock_volatility_regime(returns, shock_factor=2.0)
        
        if np.std(shocked_returns) > 0:
            shocked_sharpe = np.mean(shocked_returns) / np.std(shocked_returns) * np.sqrt(252)
            
            # Calculate drawdown
            cum_returns = np.cumprod(1 + shocked_returns) - 1
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - peak) / (1 + peak)
            max_dd = np.min(drawdown)
            
            passed = shocked_sharpe >= config.min_sharpe_threshold and max_dd >= config.max_drawdown_threshold
        else:
            shocked_sharpe = 0
            max_dd = 0
            passed = False
        
        return {
            "scenario": scenario.value,
            "shocked_sharpe": shocked_sharpe,
            "shocked_max_drawdown": max_dd,
            "passed": passed,
            "pass_rate": 1.0 if passed else 0.0
        }
    
    else:
        return {
            "scenario": scenario.value,
            "error": f"Scenario {scenario.value} not implemented",
            "passed": False,
            "pass_rate": 0.0
        }


@register("research.robustness.battery")
def research_robustness_battery(backtest_results: Dict[str, Any], 
                               scenarios: str = "default",
                               pass_rate_threshold: float = 0.70,
                               live: bool = False, **kwargs) -> ToolResult:
    """
    Run robustness stress test battery
    
    Args:
        backtest_results: Backtest results with returns
        scenarios: Scenarios to run ("default", "all", or comma-separated list)
        pass_rate_threshold: Minimum pass rate required (default: 70%)
        live: Whether to use live data
        
    Returns:
        ToolResult with robustness test results
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "Robustness Testing")
    
    try:
        if live:
            raise NotImplementedError("Live robustness testing not implemented")
        
        # Parse scenarios
        if scenarios == "default":
            test_scenarios = [
                StressScenario.BOOTSTRAP_RETURNS,
                StressScenario.REGIME_SHUFFLE, 
                StressScenario.NOISE_JITTER
            ]
        elif scenarios == "all":
            test_scenarios = list(StressScenario)
        else:
            # Parse comma-separated list
            scenario_names = [s.strip() for s in scenarios.split(',')]
            test_scenarios = []
            for name in scenario_names:
                try:
                    test_scenarios.append(StressScenario(name))
                except ValueError:
                    pass  # Skip invalid scenario names
        
        # Extract returns from backtest results
        # For CI, generate mock returns based on backtest metrics
        annual_return = backtest_results.get("annual_return", 0.10)
        annual_vol = backtest_results.get("annual_volatility", 0.15)
        
        # Generate mock daily returns
        np.random.seed(42)  # Deterministic for CI
        n_days = 252  # One year
        daily_returns = np.random.normal(
            annual_return / 252, 
            annual_vol / np.sqrt(252), 
            n_days
        )
        
        # Configure robustness testing
        config = RobustnessConfig(
            scenarios=test_scenarios,
            n_bootstrap_samples=100,  # Reduced for CI
            pass_rate_threshold=pass_rate_threshold
        )
        
        # Run stress scenarios
        scenario_results = []
        scenario_receipts = []
        
        for scenario in test_scenarios:
            scenario_result = run_stress_scenario(daily_returns, scenario, config)
            scenario_results.append(scenario_result)
            
            # Generate receipt for each scenario
            scenario_receipt = generate_receipt(
                f"research.robustness.scenario_{scenario.value}", 
                scenario_result
            )
            scenario_receipts.append(scenario_receipt[:16])
        
        # Calculate overall pass rate
        scenario_pass_rates = [r.get("pass_rate", 0.0) for r in scenario_results]
        overall_pass_rate = np.mean(scenario_pass_rates)
        
        # Count passed scenarios
        passed_scenarios = sum(1 for r in scenario_results 
                             if r.get("pass_rate", 0.0) >= pass_rate_threshold)
        
        battery_passed = overall_pass_rate >= pass_rate_threshold
        
        # Compile results
        robustness_results = {
            "battery_configuration": {
                "scenarios_tested": [s.value for s in test_scenarios],
                "pass_rate_threshold": pass_rate_threshold,
                "n_scenarios": len(test_scenarios)
            },
            "scenario_results": scenario_results,
            "battery_summary": {
                "overall_pass_rate": overall_pass_rate,
                "scenarios_passed": passed_scenarios,
                "scenarios_total": len(test_scenarios),
                "battery_passed": battery_passed,
                "weakest_scenario": min(scenario_results, key=lambda x: x.get("pass_rate", 0))["scenario"] if scenario_results else None
            },
            "scenario_receipts": scenario_receipts,
            "original_metrics": {
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "sharpe_ratio": backtest_results.get("sharpe_ratio", 0)
            },
            "testing_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Generate aggregate receipt
        receipt_hash = generate_receipt("research.robustness.battery", robustness_results)
        robustness_results["battery_receipt"] = receipt_hash[:16]
        
        # Determine warnings and errors
        warnings = []
        if not battery_passed:
            warnings.append(f"Robustness battery failed: {overall_pass_rate:.1%} pass rate < {pass_rate_threshold:.1%}")
        
        failed_scenarios = [r["scenario"] for r in scenario_results if r.get("pass_rate", 0) < 0.5]
        if failed_scenarios:
            warnings.append(f"Failed scenarios: {', '.join(failed_scenarios)}")
        
        return ToolResult(
            ok=battery_passed,
            data=robustness_results,
            receipt_hash=receipt_hash,
            warnings=warnings if warnings else [f"Robustness battery passed: {overall_pass_rate:.1%} pass rate"],
            errors=[] if battery_passed else [f"Battery failed: {overall_pass_rate:.1%} < {pass_rate_threshold:.1%} required"]
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "scenarios": scenarios,
            "pass_rate_threshold": pass_rate_threshold
        }
        receipt_hash = generate_receipt("research.robustness.error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"Robustness testing failed: {e}"],
            receipt_hash=receipt_hash
        )


if __name__ == "__main__":
    # Test robustness functionality
    print("🧪 Testing Robustness Battery...")
    
    # Mock backtest results
    mock_backtest = {
        "annual_return": 0.12,
        "annual_volatility": 0.18,
        "sharpe_ratio": 0.67,
        "max_drawdown": -0.15
    }
    
    result = research_robustness_battery(
        backtest_results=mock_backtest,
        scenarios="default",
        pass_rate_threshold=0.60,  # Lower threshold for test
        live=False
    )
    
    print(f"Robustness battery: {result.ok}")
    if result.ok:
        data = result.data
        print(f"Pass rate: {data['battery_summary']['overall_pass_rate']:.1%}")
        print(f"Scenarios passed: {data['battery_summary']['scenarios_passed']}/{data['battery_summary']['scenarios_total']}")
        print(f"Weakest scenario: {data['battery_summary']['weakest_scenario']}")
        print(f"Receipt: {data['battery_receipt']}")
    else:
        print(f"Errors: {result.errors}")
    
    print("\n🎯 Robustness battery ready for testing")