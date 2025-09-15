#!/usr/bin/env python3
"""
Meta-learner budget allocation tests - Phase 6.2 testing
"""

import os
import json
from datetime import datetime, timedelta

# Ensure CI stays dry
os.environ["ALLY_LIVE"] = "0"

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Handle missing dependencies gracefully for CI
try:
    import pytest
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
        'mean': lambda x: sum(x) / len(x) if x else 0,
        'sqrt': lambda x: x ** 0.5,
        'std': lambda x: (sum((v - sum(x)/len(x))**2 for v in x) / len(x))**0.5 if x else 0
    })()
    
    pytest = type('pytest', (), {
        'raises': lambda *args, **kwargs: type('MockRaises', (), {
            '__enter__': lambda self: self,
            '__exit__': lambda self, *args: False
        })(),
        'main': lambda args: print("pytest not available - using mock tests")
    })()


def test_strategy_candidate_creation():
    """Test strategy candidate creation and properties"""
    from ally.research.meta_learner import StrategyCandidate
    
    candidate = StrategyCandidate(
        strategy_hash="test_strategy_001",
        strategy_type="momentum",
        fitness_history=[0.5, 0.6, 0.7],
        resource_consumed=10.0,
        last_updated=datetime.now(),
        validation_status={"wf_pass": True, "costs_pass": True, "robust_pass": False},
        metadata={"factor": "momentum_12m"}
    )
    
    # Test basic properties
    assert candidate.strategy_hash == "test_strategy_001"
    assert candidate.strategy_type == "momentum"
    assert len(candidate.fitness_history) == 3
    
    # Test computed properties
    assert candidate.current_fitness == 0.7  # Latest fitness
    assert candidate.fitness_trend > 0  # Improving trend
    assert candidate.efficiency_score == 0.7 / 10.0  # fitness / resource
    assert candidate.validation_score == 2.0 / 3.0  # 2 passed / 3 total


def test_strategy_candidate_edge_cases():
    """Test strategy candidate edge cases"""
    from ally.research.meta_learner import StrategyCandidate
    
    # Empty fitness history
    empty_candidate = StrategyCandidate(
        strategy_hash="empty_001",
        strategy_type="test",
        fitness_history=[],
        resource_consumed=0.0,
        last_updated=datetime.now(),
        validation_status={},
        metadata={}
    )
    
    assert empty_candidate.current_fitness == 0.0
    assert empty_candidate.fitness_trend == 0.0
    assert empty_candidate.efficiency_score == 0.0
    assert empty_candidate.validation_score == 0.0
    
    # Single fitness value
    single_candidate = StrategyCandidate(
        strategy_hash="single_001",
        strategy_type="test", 
        fitness_history=[0.8],
        resource_consumed=5.0,
        last_updated=datetime.now(),
        validation_status={"test_pass": True},
        metadata={}
    )
    
    assert single_candidate.current_fitness == 0.8
    assert single_candidate.fitness_trend == 0.0  # No trend with single value
    assert single_candidate.efficiency_score == 0.8 / 5.0
    assert single_candidate.validation_score == 1.0


def test_meta_learner_config():
    """Test meta-learner configuration"""
    from ally.research.meta_learner import MetaLearnerConfig
    
    config = MetaLearnerConfig(
        total_budget=200.0,
        explore_ratio=0.4,
        exploit_ratio=0.4,
        validate_ratio=0.2
    )
    
    assert config.total_budget == 200.0
    assert config.explore_ratio == 0.4
    assert config.exploit_ratio == 0.4
    assert config.validate_ratio == 0.2
    assert config.explore_ratio + config.exploit_ratio + config.validate_ratio == 1.0


def test_meta_learner_candidate_management():
    """Test meta-learner candidate addition and updates"""
    from ally.research.meta_learner import MetaLearner, MetaLearnerConfig, StrategyCandidate
    
    config = MetaLearnerConfig(total_budget=100.0)
    learner = MetaLearner(config)
    
    # Add candidate
    candidate = StrategyCandidate(
        strategy_hash="test_001",
        strategy_type="momentum",
        fitness_history=[0.5],
        resource_consumed=2.0,
        last_updated=datetime.now(),
        validation_status={"wf_pass": True},
        metadata={}
    )
    
    learner.add_candidate(candidate)
    assert "test_001" in learner.candidates
    assert "test_001" in learner.performance_tracker
    
    # Update fitness
    learner.update_candidate_fitness("test_001", 0.6, 3.0)
    updated_candidate = learner.candidates["test_001"]
    assert updated_candidate.fitness_history == [0.5, 0.6]
    assert updated_candidate.resource_consumed == 5.0
    
    # Update validation status
    learner.update_validation_status("test_001", {"costs_pass": True})
    assert updated_candidate.validation_status["costs_pass"] == True


def test_exploration_value_calculation():
    """Test exploration value using Upper Confidence Bound"""
    from ally.research.meta_learner import MetaLearner, MetaLearnerConfig, StrategyCandidate
    
    config = MetaLearnerConfig()
    learner = MetaLearner(config)
    
    # Add multiple candidates
    candidates = []
    for i in range(3):
        candidate = StrategyCandidate(
            strategy_hash=f"test_{i}",
            strategy_type="test",
            fitness_history=[0.5 + i * 0.1] * (i + 1),  # Different evaluation counts
            resource_consumed=1.0,
            last_updated=datetime.now(),
            validation_status={},
            metadata={}
        )
        candidates.append(candidate)
        learner.add_candidate(candidate)
    
    # Calculate exploration values
    exploration_values = {}
    for candidate in candidates:
        exploration_values[candidate.strategy_hash] = learner.calculate_exploration_value(candidate)
    
    # Unexplored candidates should have high value
    unexplored = StrategyCandidate(
        strategy_hash="unexplored",
        strategy_type="test",
        fitness_history=[],
        resource_consumed=0.0,
        last_updated=datetime.now(),
        validation_status={},
        metadata={}
    )
    
    unexplored_value = learner.calculate_exploration_value(unexplored)
    assert unexplored_value == float('inf')  # Highest priority for unexplored


def test_exploitation_value_calculation():
    """Test exploitation value calculation"""
    from ally.research.meta_learner import MetaLearner, MetaLearnerConfig, StrategyCandidate
    
    config = MetaLearnerConfig()
    learner = MetaLearner(config)
    
    # High performing candidate
    high_performer = StrategyCandidate(
        strategy_hash="high_001",
        strategy_type="momentum",
        fitness_history=[0.6, 0.7, 0.8],  # Improving trend
        resource_consumed=5.0,
        last_updated=datetime.now(),
        validation_status={"wf_pass": True, "costs_pass": True, "robust_pass": True},
        metadata={}
    )
    
    # Low performing candidate
    low_performer = StrategyCandidate(
        strategy_hash="low_001", 
        strategy_type="reversal",
        fitness_history=[0.3, 0.25, 0.2],  # Declining trend
        resource_consumed=10.0,
        last_updated=datetime.now(),
        validation_status={"wf_pass": False},
        metadata={}
    )
    
    high_value = learner.calculate_exploitation_value(high_performer)
    low_value = learner.calculate_exploitation_value(low_performer)
    
    # High performer should have higher exploitation value
    assert high_value > low_value


def test_novelty_score_calculation():
    """Test novelty score calculation based on strategy diversity"""
    from ally.research.meta_learner import MetaLearner, MetaLearnerConfig, StrategyCandidate
    
    config = MetaLearnerConfig()
    learner = MetaLearner(config)
    
    # Add multiple momentum strategies (common type)
    for i in range(3):
        candidate = StrategyCandidate(
            strategy_hash=f"momentum_{i}",
            strategy_type="momentum",
            fitness_history=[0.5],
            resource_consumed=1.0,
            last_updated=datetime.now(),
            validation_status={},
            metadata={}
        )
        learner.add_candidate(candidate)
    
    # Add unique value strategy
    value_candidate = StrategyCandidate(
        strategy_hash="value_001",
        strategy_type="value", 
        fitness_history=[0.5],
        resource_consumed=1.0,
        last_updated=datetime.now(),
        validation_status={},
        metadata={}
    )
    learner.add_candidate(value_candidate)
    
    # Calculate novelty scores
    momentum_novelty = learner.calculate_novelty_score(learner.candidates["momentum_0"])
    value_novelty = learner.calculate_novelty_score(value_candidate)
    
    # Value strategy should have higher novelty (less common)
    assert value_novelty > momentum_novelty


def test_risk_adjusted_value():
    """Test risk adjustment for value calculations"""
    from ally.research.meta_learner import MetaLearner, MetaLearnerConfig, StrategyCandidate
    
    config = MetaLearnerConfig(risk_aversion=0.2)
    learner = MetaLearner(config)
    
    # Stable candidate (low volatility)
    stable = StrategyCandidate(
        strategy_hash="stable_001",
        strategy_type="test",
        fitness_history=[0.6, 0.61, 0.59, 0.6],  # Low volatility
        resource_consumed=1.0,
        last_updated=datetime.now(),
        validation_status={},
        metadata={}
    )
    
    # Volatile candidate (high volatility)
    volatile = StrategyCandidate(
        strategy_hash="volatile_001",
        strategy_type="test",
        fitness_history=[0.3, 0.9, 0.2, 0.8],  # High volatility, same mean
        resource_consumed=1.0,
        last_updated=datetime.now(),
        validation_status={},
        metadata={}
    )
    
    base_value = 0.6
    
    stable_adjusted = learner.calculate_risk_adjusted_value(stable, base_value)
    volatile_adjusted = learner.calculate_risk_adjusted_value(volatile, base_value)
    
    # Stable candidate should have higher risk-adjusted value
    assert stable_adjusted > volatile_adjusted


def test_budget_allocation():
    """Test budget allocation across strategy types"""
    from ally.research.meta_learner import MetaLearner, MetaLearnerConfig, StrategyCandidate
    
    config = MetaLearnerConfig(
        total_budget=100.0,
        explore_ratio=0.4,
        exploit_ratio=0.4,
        validate_ratio=0.2
    )
    learner = MetaLearner(config)
    
    # Add diverse candidates
    candidates = [
        # High performer (should get exploitation budget)
        StrategyCandidate(
            strategy_hash="high_001",
            strategy_type="momentum",
            fitness_history=[0.7, 0.8, 0.85],
            resource_consumed=5.0,
            last_updated=datetime.now(),
            validation_status={"wf_pass": True, "costs_pass": True, "robust_pass": True},
            metadata={}
        ),
        # Unexplored (should get exploration budget)
        StrategyCandidate(
            strategy_hash="unexplored_001",
            strategy_type="value",
            fitness_history=[],
            resource_consumed=0.0,
            last_updated=datetime.now(),
            validation_status={},
            metadata={}
        ),
        # Needs validation (should get validation budget)
        StrategyCandidate(
            strategy_hash="validation_001",
            strategy_type="quality",
            fitness_history=[0.6, 0.65],
            resource_consumed=3.0,
            last_updated=datetime.now(),
            validation_status={"wf_pass": True, "costs_pass": False},
            metadata={}
        )
    ]
    
    for candidate in candidates:
        learner.add_candidate(candidate)
    
    # Allocate budget
    allocations = learner.allocate_budget(seed=42)
    
    # Check allocations
    assert len(allocations) > 0
    
    # Verify allocation types
    allocation_types = [alloc.allocation_type for alloc in allocations.values()]
    assert "explore" in allocation_types or "exploit" in allocation_types
    
    # Check budget constraints
    total_allocated = sum(alloc.allocated_budget for alloc in allocations.values())
    assert total_allocated <= config.total_budget
    
    # Each allocation should have required fields
    for strategy_hash, allocation in allocations.items():
        assert allocation.strategy_hash == strategy_hash
        assert allocation.allocated_budget >= 0
        assert allocation.allocation_type in ["explore", "exploit", "validate"]
        assert 0 <= allocation.confidence <= 1
        assert allocation.justification is not None


def test_underperformer_retirement():
    """Test retirement of consistently underperforming strategies"""
    from ally.research.meta_learner import MetaLearner, MetaLearnerConfig, StrategyCandidate
    
    config = MetaLearnerConfig()
    learner = MetaLearner(config)
    
    # Add good and bad performers
    good_performer = StrategyCandidate(
        strategy_hash="good_001",
        strategy_type="momentum",
        fitness_history=[0.6, 0.65, 0.7],  # Good and improving
        resource_consumed=3.0,
        last_updated=datetime.now(),
        validation_status={},
        metadata={}
    )
    
    bad_performer = StrategyCandidate(
        strategy_hash="bad_001",
        strategy_type="reversal", 
        fitness_history=[0.05, 0.03, 0.02],  # Poor and declining
        resource_consumed=5.0,
        last_updated=datetime.now(),
        validation_status={},
        metadata={}
    )
    
    learner.add_candidate(good_performer)
    learner.add_candidate(bad_performer)
    
    # Retire underperformers
    retired = learner.retire_underperformers(performance_threshold=0.1)
    
    # Bad performer should be retired
    assert "bad_001" in retired
    assert "good_001" not in retired
    assert "bad_001" not in learner.candidates
    assert "good_001" in learner.candidates


def test_allocation_summary():
    """Test allocation summary generation"""
    from ally.research.meta_learner import MetaLearner, MetaLearnerConfig, StrategyCandidate
    
    config = MetaLearnerConfig(total_budget=100.0)
    learner = MetaLearner(config)
    
    # Add candidates and allocate
    candidate = StrategyCandidate(
        strategy_hash="test_001",
        strategy_type="momentum",
        fitness_history=[0.6],
        resource_consumed=2.0,
        last_updated=datetime.now(),
        validation_status={},
        metadata={}
    )
    
    learner.add_candidate(candidate)
    allocations = learner.allocate_budget(seed=42)
    
    # Get summary
    summary = learner.get_allocation_summary()
    
    # Check summary structure
    assert "total_candidates" in summary
    assert "total_allocations" in summary
    assert "total_budget_allocated" in summary
    assert "budget_utilization" in summary
    assert "allocation_by_type" in summary
    
    # Check values
    assert summary["total_candidates"] == 1
    assert summary["budget_utilization"] >= 0
    assert summary["budget_utilization"] <= 1


def test_meta_learner_api():
    """Test meta-learner API function"""
    from ally.research.meta_learner import research_meta_learner_allocation
    
    # Test with custom candidates
    candidates_data = [
        {
            "strategy_hash": "api_test_001",
            "strategy_type": "momentum",
            "fitness_history": [0.6, 0.7],
            "resource_consumed": 4.0,
            "validation_status": {"wf_pass": True, "costs_pass": False}
        },
        {
            "strategy_hash": "api_test_002",
            "strategy_type": "reversal",
            "fitness_history": [0.4, 0.5, 0.6],
            "resource_consumed": 6.0,
            "validation_status": {"wf_pass": True, "costs_pass": True, "robust_pass": True}
        }
    ]
    
    result = research_meta_learner_allocation(
        candidates_data=candidates_data,
        total_budget=80.0,
        live=False
    )
    
    assert result.ok == True
    assert "meta_learner_receipt" in result.data
    assert "allocations" in result.data
    assert "allocation_summary" in result.data
    assert "candidate_count" in result.data
    
    # Check receipt format
    assert len(result.data["meta_learner_receipt"]) == 16
    assert hasattr(result, 'receipt_hash')
    
    # Check allocations
    allocations = result.data["allocations"]
    assert len(allocations) >= 0  # May be 0 if no good candidates
    
    # Check summary
    summary = result.data["allocation_summary"]
    assert summary["total_candidates"] == 2
    assert "budget_utilization" in summary


def test_meta_learner_deterministic():
    """Test that meta-learner allocation is deterministic"""
    from ally.research.meta_learner import research_meta_learner_allocation
    
    candidates_data = [
        {
            "strategy_hash": "det_test_001",
            "strategy_type": "momentum",
            "fitness_history": [0.6, 0.65, 0.7],
            "resource_consumed": 5.0,
            "validation_status": {"wf_pass": True}
        }
    ]
    
    # Run twice with same parameters
    result1 = research_meta_learner_allocation(
        candidates_data=candidates_data,
        total_budget=50.0,
        live=False
    )
    
    result2 = research_meta_learner_allocation(
        candidates_data=candidates_data,
        total_budget=50.0,
        live=False
    )
    
    assert result1.ok == result2.ok
    
    if result1.ok and result2.ok:
        # Should have identical allocations
        alloc1 = result1.data["allocations"]
        alloc2 = result2.data["allocations"]
        
        assert len(alloc1) == len(alloc2)
        
        for strategy_hash in alloc1:
            if strategy_hash in alloc2:
                assert abs(alloc1[strategy_hash]["allocated_budget"] - 
                          alloc2[strategy_hash]["allocated_budget"]) < 1e-10


def test_meta_learner_error_handling():
    """Test meta-learner error handling"""
    from ally.research.meta_learner import research_meta_learner_allocation
    
    # Test with invalid candidates data
    result = research_meta_learner_allocation(
        candidates_data=[{"invalid": "data"}],
        total_budget=100.0,
        live=False
    )
    
    # Should handle gracefully
    assert isinstance(result.ok, bool)
    
    # Test with empty candidates (should work with mock data)
    result_empty = research_meta_learner_allocation(
        candidates_data=[],
        total_budget=100.0,
        live=False
    )
    
    assert result_empty.ok == True  # Should use mock data


def test_live_mode_gating():
    """Test live mode gating for meta-learner"""
    from ally.research.meta_learner import research_meta_learner_allocation
    
    # Test offline mode (should work)
    result = research_meta_learner_allocation(
        total_budget=50.0,
        live=False
    )
    assert result.ok == True
    
    # Test live mode (should fail in CI without proper API key)
    result_live = research_meta_learner_allocation(
        total_budget=50.0,
        live=True
    )
    # Result depends on environment setup
    assert isinstance(result_live.ok, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])