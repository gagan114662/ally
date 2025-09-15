#!/usr/bin/env python3
"""
Meta-learner for budget allocation across strategy candidates - Phase 6.2

Implements an intelligent budget allocator that learns which strategy branches
are most promising and dynamically allocates computational resources for
maximum alpha discovery efficiency.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Handle missing dependencies gracefully for CI
try:
    import numpy as np
    import pandas as pd
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    # Mock implementations for CI
    np = type('np', (), {
        'random': type('random', (), {
            'seed': lambda x: None,
            'random': lambda: 0.5,
            'choice': lambda x: x[0] if x else None,
            'uniform': lambda a, b: (a + b) / 2,
            'normal': lambda mu, sigma: mu,
            'exponential': lambda scale: scale
        })(),
        'exp': lambda x: 2.718 ** x,
        'log': lambda x: x if x > 0 else 0,
        'clip': lambda x, a, b: max(a, min(b, x)),
        'sum': lambda x: sum(x) if x else 0,
        'mean': lambda x: sum(x) / len(x) if x else 0
    })()
    pd = type('pd', (), {
        'DataFrame': lambda data: data,
        'Series': lambda data: data
    })()

from ally.utils.result import Result
from ally.utils.gating import check_live_mode_allowed, LiveModeError
from ally.utils.receipt import generate_receipt
from ally.utils.registry import register_tool


@dataclass
class StrategyCandidate:
    """Represents a strategy candidate for budget allocation"""
    strategy_hash: str
    strategy_type: str
    fitness_history: List[float]
    resource_consumed: float  # CPU-hours or equivalent
    last_updated: datetime
    validation_status: Dict[str, bool]  # Phase 5.x gate results
    metadata: Dict[str, Any]
    
    @property
    def current_fitness(self) -> float:
        """Get most recent fitness score"""
        return self.fitness_history[-1] if self.fitness_history else 0.0
    
    @property
    def fitness_trend(self) -> float:
        """Calculate fitness improvement trend"""
        if len(self.fitness_history) < 2:
            return 0.0
        
        recent = self.fitness_history[-3:]  # Last 3 evaluations
        if len(recent) < 2:
            return 0.0
            
        # Simple linear trend
        return (recent[-1] - recent[0]) / len(recent)
    
    @property
    def efficiency_score(self) -> float:
        """Calculate resource efficiency (fitness per resource unit)"""
        if self.resource_consumed <= 0:
            return self.current_fitness
        return self.current_fitness / self.resource_consumed
    
    @property
    def validation_score(self) -> float:
        """Calculate validation completeness score"""
        if not self.validation_status:
            return 0.0
        
        total_gates = len(self.validation_status)
        passed_gates = sum(1 for passed in self.validation_status.values() if passed)
        return passed_gates / total_gates


@dataclass
class BudgetAllocation:
    """Represents budget allocation decision"""
    strategy_hash: str
    allocated_budget: float
    allocation_type: str  # 'explore', 'exploit', 'validate', 'retire'
    confidence: float
    expected_value: float
    risk_adjusted_value: float
    justification: str


@dataclass
class MetaLearnerConfig:
    """Configuration for meta-learner"""
    total_budget: float = 100.0  # Total computational budget
    explore_ratio: float = 0.3  # Fraction for exploration
    exploit_ratio: float = 0.5  # Fraction for exploitation
    validate_ratio: float = 0.2  # Fraction for validation
    min_allocation: float = 1.0  # Minimum allocation per strategy
    max_allocation: float = 20.0  # Maximum allocation per strategy
    decay_factor: float = 0.9  # Budget decay for underperforming strategies
    novelty_bonus: float = 1.2  # Bonus multiplier for novel strategies
    risk_aversion: float = 0.1  # Risk adjustment parameter


class MetaLearner:
    """Meta-learner for intelligent budget allocation"""
    
    def __init__(self, config: MetaLearnerConfig):
        self.config = config
        self.candidates: Dict[str, StrategyCandidate] = {}
        self.allocation_history: List[Dict[str, BudgetAllocation]] = []
        self.performance_tracker: Dict[str, List[float]] = {}
        
    def add_candidate(self, candidate: StrategyCandidate):
        """Add a strategy candidate to the pool"""
        self.candidates[candidate.strategy_hash] = candidate
        if candidate.strategy_hash not in self.performance_tracker:
            self.performance_tracker[candidate.strategy_hash] = []
    
    def update_candidate_fitness(self, strategy_hash: str, fitness: float, resource_cost: float):
        """Update candidate fitness and resource consumption"""
        if strategy_hash in self.candidates:
            candidate = self.candidates[strategy_hash]
            candidate.fitness_history.append(fitness)
            candidate.resource_consumed += resource_cost
            candidate.last_updated = datetime.now()
            self.performance_tracker[strategy_hash].append(fitness)
    
    def update_validation_status(self, strategy_hash: str, gate_results: Dict[str, bool]):
        """Update validation gate results for a candidate"""
        if strategy_hash in self.candidates:
            self.candidates[strategy_hash].validation_status.update(gate_results)
    
    def calculate_exploration_value(self, candidate: StrategyCandidate) -> float:
        """Calculate exploration value using Upper Confidence Bound (UCB)"""
        if not candidate.fitness_history:
            return float('inf')  # Unexplored candidates get highest priority
        
        n_evaluations = len(candidate.fitness_history)
        total_evaluations = sum(len(c.fitness_history) for c in self.candidates.values())
        
        if total_evaluations == 0 or n_evaluations == 0:
            return candidate.current_fitness
        
        # UCB formula: mean + sqrt(2 * log(total_evals) / n_evals)
        mean_fitness = np.mean(candidate.fitness_history)
        confidence_interval = np.sqrt(2 * np.log(total_evaluations) / n_evaluations)
        
        return mean_fitness + confidence_interval
    
    def calculate_exploitation_value(self, candidate: StrategyCandidate) -> float:
        """Calculate exploitation value based on current performance"""
        base_value = candidate.current_fitness
        
        # Adjust for trend
        trend_bonus = candidate.fitness_trend * 0.5
        
        # Adjust for efficiency
        efficiency_bonus = candidate.efficiency_score * 0.3
        
        # Adjust for validation completeness
        validation_bonus = candidate.validation_score * 0.2
        
        return base_value + trend_bonus + efficiency_bonus + validation_bonus
    
    def calculate_novelty_score(self, candidate: StrategyCandidate) -> float:
        """Calculate novelty score based on strategy uniqueness"""
        # Simple novelty based on strategy type diversity
        strategy_types = [c.strategy_type for c in self.candidates.values()]
        type_count = strategy_types.count(candidate.strategy_type)
        total_candidates = len(self.candidates)
        
        if total_candidates == 0:
            return 1.0
        
        # Higher novelty for less common strategy types
        novelty = 1.0 - (type_count - 1) / total_candidates
        return max(0.1, novelty)
    
    def calculate_risk_adjusted_value(self, candidate: StrategyCandidate, raw_value: float) -> float:
        """Apply risk adjustment to value calculation"""
        if not candidate.fitness_history:
            return raw_value
        
        # Calculate fitness volatility as risk measure
        fitness_std = np.std(candidate.fitness_history) if len(candidate.fitness_history) > 1 else 0
        
        # Risk-adjusted value: value - risk_aversion * volatility
        risk_penalty = self.config.risk_aversion * fitness_std
        return raw_value - risk_penalty
    
    def allocate_budget(self, seed: int = 42) -> Dict[str, BudgetAllocation]:
        """Allocate budget across strategy candidates"""
        if seed is not None:
            np.random.seed(seed)
        
        if not self.candidates:
            return {}
        
        allocations = {}
        remaining_budget = self.config.total_budget
        
        # Split budget into explore/exploit/validate pools
        explore_budget = self.config.total_budget * self.config.explore_ratio
        exploit_budget = self.config.total_budget * self.config.exploit_ratio
        validate_budget = self.config.total_budget * self.config.validate_ratio
        
        # 1. Exploration allocation (for high UCB candidates)
        exploration_candidates = list(self.candidates.values())
        exploration_values = {
            c.strategy_hash: self.calculate_exploration_value(c) 
            for c in exploration_candidates
        }
        
        # Sort by exploration value (descending)
        sorted_explore = sorted(
            exploration_candidates,
            key=lambda c: exploration_values[c.strategy_hash],
            reverse=True
        )
        
        allocated_explore = 0
        for candidate in sorted_explore:
            if allocated_explore >= explore_budget:
                break
                
            allocation_amount = min(
                self.config.max_allocation,
                explore_budget - allocated_explore,
                self.config.min_allocation
            )
            
            if allocation_amount >= self.config.min_allocation:
                novelty_score = self.calculate_novelty_score(candidate)
                expected_value = exploration_values[candidate.strategy_hash]
                risk_adjusted = self.calculate_risk_adjusted_value(candidate, expected_value)
                
                allocations[candidate.strategy_hash] = BudgetAllocation(
                    strategy_hash=candidate.strategy_hash,
                    allocated_budget=allocation_amount * novelty_score,
                    allocation_type='explore',
                    confidence=0.6,  # Medium confidence for exploration
                    expected_value=expected_value,
                    risk_adjusted_value=risk_adjusted,
                    justification=f"High exploration value: {expected_value:.3f}, novelty: {novelty_score:.3f}"
                )
                allocated_explore += allocation_amount
        
        # 2. Exploitation allocation (for high-performing candidates)
        exploitation_candidates = [
            c for c in self.candidates.values() 
            if c.fitness_history and c.current_fitness > 0
        ]
        
        exploitation_values = {
            c.strategy_hash: self.calculate_exploitation_value(c)
            for c in exploitation_candidates
        }
        
        sorted_exploit = sorted(
            exploitation_candidates,
            key=lambda c: exploitation_values[c.strategy_hash],
            reverse=True
        )
        
        allocated_exploit = 0
        for candidate in sorted_exploit:
            if allocated_exploit >= exploit_budget:
                break
            
            # Skip if already allocated in exploration
            if candidate.strategy_hash in allocations:
                continue
                
            allocation_amount = min(
                self.config.max_allocation,
                exploit_budget - allocated_exploit,
                self.config.min_allocation
            )
            
            if allocation_amount >= self.config.min_allocation:
                expected_value = exploitation_values[candidate.strategy_hash]
                risk_adjusted = self.calculate_risk_adjusted_value(candidate, expected_value)
                
                allocations[candidate.strategy_hash] = BudgetAllocation(
                    strategy_hash=candidate.strategy_hash,
                    allocated_budget=allocation_amount,
                    allocation_type='exploit',
                    confidence=0.8,  # High confidence for exploitation
                    expected_value=expected_value,
                    risk_adjusted_value=risk_adjusted,
                    justification=f"High exploitation value: {expected_value:.3f}, trend: {candidate.fitness_trend:.3f}"
                )
                allocated_exploit += allocation_amount
        
        # 3. Validation allocation (for candidates needing gate validation)
        validation_candidates = [
            c for c in self.candidates.values()
            if c.validation_score < 1.0 and c.current_fitness > 0.3  # Promising but incomplete
        ]
        
        allocated_validate = 0
        for candidate in validation_candidates:
            if allocated_validate >= validate_budget:
                break
                
            # Skip if already allocated
            if candidate.strategy_hash in allocations:
                continue
                
            allocation_amount = min(
                self.config.max_allocation,
                validate_budget - allocated_validate,
                self.config.min_allocation
            )
            
            if allocation_amount >= self.config.min_allocation:
                expected_value = candidate.current_fitness
                risk_adjusted = self.calculate_risk_adjusted_value(candidate, expected_value)
                
                allocations[candidate.strategy_hash] = BudgetAllocation(
                    strategy_hash=candidate.strategy_hash,
                    allocated_budget=allocation_amount,
                    allocation_type='validate',
                    confidence=0.7,  # Medium-high confidence for validation
                    expected_value=expected_value,
                    risk_adjusted_value=risk_adjusted,
                    justification=f"Validation needed: {candidate.validation_score:.2f} completeness"
                )
                allocated_validate += allocation_amount
        
        # Store allocation history
        self.allocation_history.append(allocations.copy())
        
        return allocations
    
    def retire_underperformers(self, performance_threshold: float = 0.1) -> List[str]:
        """Retire consistently underperforming strategies"""
        retired = []
        
        for strategy_hash, candidate in list(self.candidates.items()):
            if (len(candidate.fitness_history) >= 3 and
                candidate.current_fitness < performance_threshold and
                candidate.fitness_trend <= 0):
                
                retired.append(strategy_hash)
                del self.candidates[strategy_hash]
        
        return retired
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of current allocation state"""
        if not self.allocation_history:
            return {"total_candidates": len(self.candidates), "total_allocations": 0}
        
        latest_allocations = self.allocation_history[-1]
        
        total_allocated = sum(alloc.allocated_budget for alloc in latest_allocations.values())
        allocation_by_type = {}
        
        for alloc in latest_allocations.values():
            alloc_type = alloc.allocation_type
            if alloc_type not in allocation_by_type:
                allocation_by_type[alloc_type] = 0
            allocation_by_type[alloc_type] += alloc.allocated_budget
        
        return {
            "total_candidates": len(self.candidates),
            "total_allocations": len(latest_allocations),
            "total_budget_allocated": total_allocated,
            "budget_utilization": total_allocated / self.config.total_budget,
            "allocation_by_type": allocation_by_type,
            "avg_allocation": total_allocated / len(latest_allocations) if latest_allocations else 0
        }


@register_tool("meta.learner")
def research_meta_learner_allocation(
    candidates_data: Optional[List[Dict]] = None,
    config: Optional[Dict] = None,
    total_budget: float = 100.0,
    live: bool = True
) -> Result:
    """
    Run meta-learner budget allocation across strategy candidates
    
    Args:
        candidates_data: List of strategy candidate specifications
        config: Meta-learner configuration parameters
        total_budget: Total computational budget to allocate
        live: Enable live mode (requires ALLY_LIVE=1)
    
    Returns:
        Result with budget allocations and allocation summary
    """
    try:
        # Gate live mode
        if live:
            check_live_mode_allowed(
                live=live,
                api_key=os.getenv("META_LEARNER_API_KEY", "not_set"),
                service_name="Meta-Learner"
            )
        
        # Default configuration
        learner_config = MetaLearnerConfig(total_budget=total_budget)
        
        if config:
            for key, value in config.items():
                if hasattr(learner_config, key):
                    setattr(learner_config, key, value)
        
        # Initialize meta-learner
        meta_learner = MetaLearner(learner_config)
        
        # Add candidates (use mock data if none provided)
        if not candidates_data:
            # Create mock candidates for CI
            candidates_data = [
                {
                    "strategy_hash": "mock_momentum_001",
                    "strategy_type": "momentum",
                    "fitness_history": [0.6, 0.65, 0.7],
                    "resource_consumed": 5.0,
                    "validation_status": {"wf_pass": True, "costs_pass": True, "robust_pass": False}
                },
                {
                    "strategy_hash": "mock_reversal_001", 
                    "strategy_type": "reversal",
                    "fitness_history": [0.4, 0.5, 0.45],
                    "resource_consumed": 3.0,
                    "validation_status": {"wf_pass": True, "costs_pass": False, "robust_pass": False}
                },
                {
                    "strategy_hash": "mock_value_001",
                    "strategy_type": "value", 
                    "fitness_history": [0.8, 0.85, 0.9],
                    "resource_consumed": 8.0,
                    "validation_status": {"wf_pass": True, "costs_pass": True, "robust_pass": True}
                }
            ]
        
        # Create candidate objects
        for cand_data in candidates_data:
            candidate = StrategyCandidate(
                strategy_hash=cand_data.get("strategy_hash", f"unknown_{hash(str(cand_data))}"),
                strategy_type=cand_data.get("strategy_type", "unknown"),
                fitness_history=cand_data.get("fitness_history", []),
                resource_consumed=cand_data.get("resource_consumed", 0.0),
                last_updated=datetime.now(),
                validation_status=cand_data.get("validation_status", {}),
                metadata=cand_data.get("metadata", {})
            )
            meta_learner.add_candidate(candidate)
        
        # Allocate budget
        allocations = meta_learner.allocate_budget(seed=42)
        
        # Get allocation summary
        summary = meta_learner.get_allocation_summary()
        
        # Retire underperformers
        retired = meta_learner.retire_underperformers()
        
        # Generate receipt
        allocation_data = {
            "total_budget": total_budget,
            "total_candidates": len(meta_learner.candidates),
            "total_allocations": len(allocations),
            "budget_utilization": summary["budget_utilization"],
            "retired_strategies": len(retired),
            "config": asdict(learner_config)
        }
        
        receipt_hash = generate_receipt("meta.learner", allocation_data)
        
        return Result(
            ok=True,
            data={
                "meta_learner_receipt": receipt_hash[:16],
                "allocations": {k: asdict(v) for k, v in allocations.items()},
                "allocation_summary": summary,
                "retired_strategies": retired,
                "candidate_count": len(meta_learner.candidates),
                "budget_utilization": summary["budget_utilization"],
                "allocation_breakdown": summary["allocation_by_type"],
                "config_used": asdict(learner_config)
            },
            receipt_hash=receipt_hash
        )
        
    except LiveModeError as e:
        return Result(ok=False, errors=[str(e)])
    except Exception as e:
        return Result(ok=False, errors=[f"Meta-learner allocation failed: {str(e)}"])


if __name__ == "__main__":
    # Test meta-learner allocation
    result = research_meta_learner_allocation(
        total_budget=50.0,
        live=False
    )
    
    if result.ok:
        print("✅ Meta-learner allocation completed")
        print(f"Receipt: {result.data['meta_learner_receipt']}")
        print(f"Candidates: {result.data['candidate_count']}")
        print(f"Allocations: {len(result.data['allocations'])}")
        print(f"Budget utilization: {result.data['budget_utilization']:.2%}")
    else:
        print("❌ Meta-learner allocation failed")
        for error in result.errors:
            print(f"Error: {error}")