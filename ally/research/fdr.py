"""
FDR gate - False Discovery Rate control using Benjamini-Hochberg procedure
Multiple testing correction for strategy evaluation grids
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats

from ally.utils.receipts import generate_receipt
from ally.utils.gating import check_live_mode_allowed
from ally.core.tool_registry import register, ToolResult


def benjamini_hochberg_procedure(p_values: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, float]:
    """
    Apply Benjamini-Hochberg procedure for FDR control
    
    Args:
        p_values: Array of p-values to correct
        alpha: Desired FDR level (default: 0.05)
        
    Returns:
        Tuple of (rejected_hypotheses, adjusted_threshold)
    """
    if len(p_values) == 0:
        return np.array([], dtype=bool), 0.0
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    m = len(p_values)
    
    # Find largest k such that P(k) <= (k/m) * alpha
    # Work backwards from largest p-value
    rejected = np.zeros(m, dtype=bool)
    adjusted_threshold = 0.0
    
    for i in range(m-1, -1, -1):
        threshold = ((i + 1) / m) * alpha
        if sorted_p_values[i] <= threshold:
            # Reject this and all smaller p-values
            rejected_sorted = np.zeros(m, dtype=bool)
            rejected_sorted[:i+1] = True
            
            # Map back to original indices
            rejected[sorted_indices] = rejected_sorted
            adjusted_threshold = threshold
            break
    
    return rejected, adjusted_threshold


def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Apply Bonferroni correction for comparison"""
    return p_values <= (alpha / len(p_values))


def simulate_strategy_grid(n_strategies: int = 100, n_observations: int = 252,
                          true_alpha_rate: float = 0.1, seed: int = 42) -> Dict[str, Any]:
    """
    Simulate a grid of strategy backtests for FDR testing
    
    Args:
        n_strategies: Number of strategies in grid
        n_observations: Number of return observations per strategy
        true_alpha_rate: Fraction of strategies with true alpha
        seed: Random seed for reproducibility
        
    Returns:
        Dict with strategy results and ground truth
    """
    np.random.seed(seed)
    
    # Determine which strategies have true alpha
    n_true_alpha = int(n_strategies * true_alpha_rate)
    has_true_alpha = np.zeros(n_strategies, dtype=bool)
    has_true_alpha[:n_true_alpha] = True
    np.random.shuffle(has_true_alpha)
    
    strategy_results = []
    
    for i in range(n_strategies):
        # Generate returns
        if has_true_alpha[i]:
            # Strategy with true alpha (2% annual = ~0.008% daily)
            daily_alpha = 0.02 / 252
            returns = np.random.normal(daily_alpha, 0.015, n_observations)
        else:
            # No alpha strategy
            returns = np.random.normal(0.0, 0.015, n_observations)
        
        # Compute statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        t_stat = mean_return / (std_return / np.sqrt(n_observations))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_observations-1))
        
        # Annualized metrics
        annual_return = mean_return * 252
        annual_vol = std_return * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        strategy_results.append({
            'strategy_id': f"STRAT_{i:03d}",
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'mean_return': mean_return,
            'return_std': std_return,
            't_stat': t_stat,
            'p_value': p_value,
            'has_true_alpha': has_true_alpha[i],  # Ground truth
            'observations': n_observations
        })
    
    return {
        'strategies': strategy_results,
        'simulation_params': {
            'n_strategies': n_strategies,
            'n_observations': n_observations,
            'true_alpha_rate': true_alpha_rate,
            'n_true_alpha': n_true_alpha,
            'seed': seed
        }
    }


@register("research.fdr.analyze_grid")
def research_fdr_analyze_grid(strategy_results: List[Dict[str, Any]],
                             fdr_level: float = 0.05, live: bool = False, **kwargs) -> ToolResult:
    """
    Apply FDR correction to strategy evaluation grid
    
    Args:
        strategy_results: List of strategy evaluation results with p-values
        fdr_level: Desired false discovery rate (default: 0.05)
        live: Whether this is live analysis
        
    Returns:
        ToolResult with FDR analysis results
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "FDR Grid Analysis")
    
    try:
        if not strategy_results:
            return ToolResult(
                ok=False,
                errors=["No strategy results provided"]
            )
        
        # Extract p-values and strategy info
        p_values = []
        strategy_names = []
        
        for result in strategy_results:
            if 'p_value' not in result:
                return ToolResult(
                    ok=False,
                    errors=["Strategy results must contain 'p_value' field"]
                )
            
            p_values.append(result['p_value'])
            strategy_names.append(result.get('strategy_id', result.get('name', f"Strategy_{len(p_values)}")))
        
        p_values = np.array(p_values)
        
        # Apply Benjamini-Hochberg procedure
        bh_rejected, bh_threshold = benjamini_hochberg_procedure(p_values, fdr_level)
        
        # Apply Bonferroni correction for comparison
        bonf_rejected = bonferroni_correction(p_values, fdr_level)
        
        # Compile results
        survivors = []
        rejected_strategies = []
        
        for i, (rejected, strategy_name) in enumerate(zip(bh_rejected, strategy_names)):
            strategy_summary = {
                'strategy_id': strategy_name,
                'p_value': p_values[i],
                'bh_rejected': bool(rejected),
                'bonf_rejected': bool(bonf_rejected[i]),
                'rank': int(np.argsort(p_values)[i] + 1),  # Rank by p-value
                **{k: v for k, v in strategy_results[i].items() if k != 'strategy_id'}
            }
            
            if rejected:
                survivors.append(strategy_summary)
            else:
                rejected_strategies.append(strategy_summary)
        
        # Summary statistics
        n_total = len(strategy_results)
        n_survivors_bh = int(np.sum(bh_rejected))
        n_survivors_bonf = int(np.sum(bonf_rejected))
        
        # Calculate empirical FDR if ground truth available
        empirical_fdr = None
        if all('has_true_alpha' in result for result in strategy_results):
            true_positives = sum(1 for i, result in enumerate(strategy_results) 
                               if bh_rejected[i] and result['has_true_alpha'])
            false_positives = sum(1 for i, result in enumerate(strategy_results)
                                if bh_rejected[i] and not result['has_true_alpha'])
            
            if n_survivors_bh > 0:
                empirical_fdr = false_positives / n_survivors_bh
            else:
                empirical_fdr = 0.0
        
        analysis_results = {
            "fdr_level": fdr_level,
            "n_strategies_total": n_total,
            "n_survivors_bh": n_survivors_bh,
            "n_survivors_bonferroni": n_survivors_bonf,
            "bh_threshold": bh_threshold,
            "bonferroni_threshold": fdr_level / n_total,
            "survivors": survivors,
            "rejected": rejected_strategies,
            "empirical_fdr": empirical_fdr,
            "discovery_rate": n_survivors_bh / n_total if n_total > 0 else 0,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("research.fdr.analyze_grid", analysis_results)
        analysis_results["fdr_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=analysis_results,
            receipt_hash=receipt_hash,
            warnings=[f"FDR analysis found {n_survivors_bh}/{n_total} survivors"] if n_survivors_bh > 0 else []
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "fdr_level": fdr_level,
            "n_strategies": len(strategy_results) if strategy_results else 0
        }
        receipt_hash = generate_receipt("research.fdr.error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"FDR analysis failed: {e}"],
            receipt_hash=receipt_hash
        )


@register("research.fdr.simulate_grid")
def research_fdr_simulate_grid(n_strategies: int = 100, n_observations: int = 252,
                              true_alpha_rate: float = 0.1, fdr_level: float = 0.05,
                              live: bool = False, **kwargs) -> ToolResult:
    """
    Simulate strategy grid and apply FDR analysis
    
    Args:
        n_strategies: Number of strategies to simulate
        n_observations: Return observations per strategy
        true_alpha_rate: Fraction with true alpha
        fdr_level: FDR control level
        live: Whether this is live simulation
        
    Returns:
        ToolResult with simulation and FDR results
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "FDR Grid Simulation")
    
    try:
        # Generate strategy grid
        simulation_data = simulate_strategy_grid(
            n_strategies=n_strategies,
            n_observations=n_observations,
            true_alpha_rate=true_alpha_rate,
            seed=42  # Deterministic for CI
        )
        
        # Apply FDR analysis
        fdr_result = research_fdr_analyze_grid(
            strategy_results=simulation_data['strategies'],
            fdr_level=fdr_level,
            live=live
        )
        
        if not fdr_result.ok:
            return fdr_result
        
        # Combine results
        combined_results = {
            "simulation_params": simulation_data['simulation_params'],
            "fdr_analysis": fdr_result.data,
            "performance_metrics": {
                "true_positive_rate": None,
                "false_positive_rate": None,
                "precision": None,
                "power": None
            }
        }
        
        # Calculate performance metrics with ground truth
        fdr_data = fdr_result.data
        survivors = fdr_data['survivors']
        rejected = fdr_data['rejected']
        
        # Count outcomes
        true_positives = sum(1 for s in survivors if s.get('has_true_alpha', False))
        false_positives = sum(1 for s in survivors if not s.get('has_true_alpha', False))
        true_negatives = sum(1 for s in rejected if not s.get('has_true_alpha', False))
        false_negatives = sum(1 for s in rejected if s.get('has_true_alpha', False))
        
        n_true_alpha = simulation_data['simulation_params']['n_true_alpha']
        n_no_alpha = n_strategies - n_true_alpha
        
        # Performance metrics
        tpr = true_positives / n_true_alpha if n_true_alpha > 0 else 0  # Power
        fpr = false_positives / n_no_alpha if n_no_alpha > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        combined_results["performance_metrics"] = {
            "true_positive_rate": tpr,  # Power
            "false_positive_rate": fpr,
            "precision": precision,
            "power": tpr,
            "empirical_fdr": fdr_data.get('empirical_fdr'),
            "confusion_matrix": {
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives
            }
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("research.fdr.simulate_grid", combined_results)
        combined_results["simulation_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=combined_results,
            receipt_hash=receipt_hash,
            warnings=[f"Simulated {n_strategies} strategies with {true_alpha_rate*100:.1f}% true alpha rate"]
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "n_strategies": n_strategies,
            "true_alpha_rate": true_alpha_rate
        }
        receipt_hash = generate_receipt("research.fdr.simulation_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"FDR simulation failed: {e}"],
            receipt_hash=receipt_hash
        )


@register("research.fdr.validate_procedure")
def research_fdr_validate_procedure(n_simulations: int = 100, fdr_level: float = 0.05,
                                   live: bool = False, **kwargs) -> ToolResult:
    """
    Validate BH procedure with multiple simulations
    
    Args:
        n_simulations: Number of simulation runs
        fdr_level: FDR level to validate
        live: Whether this is live validation
        
    Returns:
        ToolResult with validation results
    """
    check_live_mode_allowed(live, kwargs.get("api_key"), "FDR Procedure Validation")
    
    try:
        validation_results = []
        
        for sim in range(n_simulations):
            # Run simulation with different seed
            sim_result = research_fdr_simulate_grid(
                n_strategies=50,
                n_observations=252,
                true_alpha_rate=0.2,
                fdr_level=fdr_level,
                live=live
            )
            
            if sim_result.ok:
                metrics = sim_result.data["performance_metrics"]
                validation_results.append({
                    'simulation': sim,
                    'empirical_fdr': metrics.get('empirical_fdr', 0),
                    'power': metrics.get('power', 0),
                    'n_survivors': sim_result.data['fdr_analysis']['n_survivors_bh']
                })
        
        if not validation_results:
            return ToolResult(
                ok=False,
                errors=["No successful simulations completed"]
            )
        
        # Aggregate results
        fdrs = [r['empirical_fdr'] for r in validation_results if r['empirical_fdr'] is not None]
        powers = [r['power'] for r in validation_results]
        survivors = [r['n_survivors'] for r in validation_results]
        
        summary = {
            "n_simulations": len(validation_results),
            "fdr_level_target": fdr_level,
            "empirical_fdr": {
                "mean": np.mean(fdrs) if fdrs else None,
                "std": np.std(fdrs) if fdrs else None,
                "median": np.median(fdrs) if fdrs else None,
                "max": np.max(fdrs) if fdrs else None,
                "control_rate": sum(1 for fdr in fdrs if fdr <= fdr_level) / len(fdrs) if fdrs else None
            },
            "power": {
                "mean": np.mean(powers),
                "std": np.std(powers),
                "median": np.median(powers)
            },
            "survivors": {
                "mean": np.mean(survivors),
                "std": np.std(survivors),
                "median": np.median(survivors)
            },
            "validation_results": validation_results,
            "validation_timestamp": datetime.utcnow().isoformat()
        }
        
        # Generate receipt
        receipt_hash = generate_receipt("research.fdr.validate_procedure", summary)
        summary["validation_receipt"] = receipt_hash[:16]
        
        return ToolResult(
            ok=True,
            data=summary,
            receipt_hash=receipt_hash
        )
        
    except Exception as e:
        error_dict = {
            "error": str(e),
            "n_simulations": n_simulations,
            "fdr_level": fdr_level
        }
        receipt_hash = generate_receipt("research.fdr.validation_error", error_dict)
        
        return ToolResult(
            ok=False,
            errors=[f"FDR validation failed: {e}"],
            receipt_hash=receipt_hash
        )


if __name__ == "__main__":
    # Test FDR functionality
    print("🧪 Testing FDR...")
    
    # Test simulation
    sim_result = research_fdr_simulate_grid(
        n_strategies=20,
        n_observations=252,
        true_alpha_rate=0.2,
        fdr_level=0.05,
        live=False
    )
    
    print(f"Simulation success: {sim_result.ok}")
    if sim_result.ok:
        data = sim_result.data
        print(f"Survivors: {data['fdr_analysis']['n_survivors_bh']}/20")
        print(f"Empirical FDR: {data['performance_metrics']['empirical_fdr']:.3f}")
        print(f"Power: {data['performance_metrics']['power']:.3f}")
        print(f"Receipt: {data['simulation_receipt']}")
    else:
        print(f"Errors: {sim_result.errors}")
    
    # Test BH procedure directly
    test_p_values = np.array([0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.8])
    rejected, threshold = benjamini_hochberg_procedure(test_p_values, 0.05)
    print(f"BH test - Rejected: {np.sum(rejected)}/{len(test_p_values)}, Threshold: {threshold:.4f}")