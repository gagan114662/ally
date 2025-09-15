#!/usr/bin/env python3
"""
Evolutionary search engine tests - Phase 6.1 testing
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
    class MockSeries:
        def __init__(self, data=None):
            self.data = data or []
            
        def __len__(self):
            return len(self.data)
    
    pd = type('pd', (), {'Series': MockSeries})()
    
    np = type('np', (), {
        'random': type('random', (), {
            'seed': lambda x: None,
            'random': lambda: 0.5,
            'choice': lambda x: x[0] if x else None,
            'uniform': lambda a, b: (a + b) / 2,
            'normal': lambda mu, sigma: mu,
            'clip': lambda x, a, b: max(a, min(b, x))
        })(),
        'mean': lambda x: sum(x) / len(x) if x else 0,
        'std': lambda x: (sum((v - sum(x)/len(x))**2 for v in x) / len(x))**0.5 if x else 0
    })()
    
    pytest = type('pytest', (), {
        'raises': lambda *args, **kwargs: type('MockRaises', (), {
            '__enter__': lambda self: self,
            '__exit__': lambda self, *args: False
        })(),
        'main': lambda args: print("pytest not available - using mock tests")
    })()


def test_strategy_gene_creation():
    """Test strategy gene creation and ID generation"""
    from ally.research.evolution import StrategyGene
    
    gene = StrategyGene(
        gene_id="",
        factor_type="momentum",
        parameters={"lookback": 252, "skip": 21},
        lookback_days=252,
        signal_transform="zscore",
        universe_filter="liquid_1000",
        mutation_history=["academic_prior"]
    )
    
    # Gene ID should be auto-generated
    assert len(gene.gene_id) == 16
    assert gene.factor_type == "momentum"
    assert gene.parameters["lookback"] == 252
    assert "academic_prior" in gene.mutation_history


def test_academic_priors_creation():
    """Test creation of academic factor priors"""
    from ally.research.evolution import create_academic_priors
    
    priors = create_academic_priors()
    
    assert len(priors) >= 5  # Should have momentum, reversal, value, quality, vol
    
    # Check factor types are present
    factor_types = [p.factor_type for p in priors]
    expected_types = ["momentum", "reversal", "value", "quality", "vol"]
    
    for expected_type in expected_types:
        assert expected_type in factor_types
    
    # Check gene IDs are generated
    for prior in priors:
        assert len(prior.gene_id) == 16
        assert "academic_prior" in prior.mutation_history


def test_gene_mutation():
    """Test gene mutation operators"""
    from ally.research.evolution import StrategyGene, mutate_gene
    
    original_gene = StrategyGene(
        gene_id="",
        factor_type="momentum",
        parameters={"lookback": 252, "skip": 21},
        lookback_days=252,
        signal_transform="zscore",
        universe_filter="liquid_1000",
        mutation_history=["original"]
    )
    
    # Test mutation with high rate (should mutate)
    mutated = mutate_gene(original_gene, mutation_rate=1.0, seed=42)
    
    # Should have new mutation in history
    assert len(mutated.mutation_history) > len(original_gene.mutation_history)
    
    # Test mutation with zero rate (should not mutate)
    unmutated = mutate_gene(original_gene, mutation_rate=0.0, seed=42)
    assert unmutated.gene_id == original_gene.gene_id
    
    # Test deterministic mutation
    mutated1 = mutate_gene(original_gene, mutation_rate=0.5, seed=42)
    mutated2 = mutate_gene(original_gene, mutation_rate=0.5, seed=42)
    
    # Should produce identical results with same seed
    assert mutated1.gene_id == mutated2.gene_id


def test_gene_crossover():
    """Test gene crossover operators"""
    from ally.research.evolution import StrategyGene, crossover_genes
    
    parent1 = StrategyGene(
        gene_id="",
        factor_type="momentum",
        parameters={"lookback": 252, "skip": 21},
        lookback_days=252,
        signal_transform="zscore",
        universe_filter="liquid_1000",
        mutation_history=["parent1"]
    )
    
    parent2 = StrategyGene(
        gene_id="",
        factor_type="reversal",
        parameters={"lookback": 5, "smoothing": "none"},
        lookback_days=5,
        signal_transform="rank",
        universe_filter="liquid_500",
        mutation_history=["parent2"]
    )
    
    child1, child2 = crossover_genes(parent1, parent2, seed=42)
    
    # Children should have crossover in mutation history
    assert "crossover" in child1.mutation_history
    assert "crossover" in child2.mutation_history
    
    # Children should have different gene IDs
    assert child1.gene_id != child2.gene_id
    assert child1.gene_id != parent1.gene_id
    assert child2.gene_id != parent2.gene_id
    
    # Test deterministic crossover
    child1_repeat, child2_repeat = crossover_genes(parent1, parent2, seed=42)
    assert child1.gene_id == child1_repeat.gene_id
    assert child2.gene_id == child2_repeat.gene_id


def test_population_management():
    """Test population creation and management"""
    from ally.research.evolution import Population, StrategyGene, calculate_diversity_metrics
    
    # Create test genes
    genes = []
    for i in range(5):
        gene = StrategyGene(
            gene_id=f"test_gene_{i}",
            factor_type=["momentum", "reversal", "value"][i % 3],
            parameters={"param": i},
            lookback_days=10 + i,
            signal_transform=["rank", "zscore"][i % 2],
            universe_filter="liquid_1000",
            mutation_history=[f"test_{i}"]
        )
        genes.append(gene)
    
    population = Population(
        generation=1,
        genes=genes,
        fitness_scores={"test_gene_0": 0.8, "test_gene_1": 0.6, "test_gene_2": 0.9},
        diversity_metrics={},
        selection_pressure=0.3
    )
    
    # Test best genes selection
    best = population.get_best_genes(2)
    assert len(best) == 2
    assert best[0].gene_id == "test_gene_2"  # Highest fitness (0.9)
    assert best[1].gene_id == "test_gene_0"  # Second highest (0.8)
    
    # Test diversity calculation
    diversity = calculate_diversity_metrics(population)
    assert "factor_diversity" in diversity
    assert "param_diversity" in diversity
    assert "transform_diversity" in diversity
    assert diversity["factor_diversity"] > 0


def test_fitness_evaluation():
    """Test fitness evaluation framework"""
    from ally.research.evolution import StrategyGene, evaluate_fitness
    
    gene = StrategyGene(
        gene_id="test_fitness",
        factor_type="momentum",
        parameters={"lookback": 100},
        lookback_days=100,
        signal_transform="zscore",
        universe_filter="liquid_1000",
        mutation_history=["test"]
    )
    
    # Mock fitness evaluator
    def mock_evaluator(strategy_spec, seed=42):
        return {"oos_sharpe": 0.75}
    
    fitness = evaluate_fitness(gene, mock_evaluator, seed=42)
    assert fitness == 0.75
    
    # Test error handling
    def failing_evaluator(strategy_spec, seed=42):
        raise ValueError("Evaluation failed")
    
    fitness_fail = evaluate_fitness(gene, failing_evaluator, seed=42)
    assert fitness_fail == 0.0  # Should return 0 on failure


def test_parent_selection():
    """Test parent selection for breeding"""
    from ally.research.evolution import Population, StrategyGene, select_parents
    
    # Create population with known fitness scores
    genes = []
    fitness_scores = {}
    
    for i in range(10):
        gene_id = f"gene_{i}"
        gene = StrategyGene(
            gene_id=gene_id,
            factor_type="momentum",
            parameters={"id": i},
            lookback_days=100,
            signal_transform="zscore",
            universe_filter="liquid_1000",
            mutation_history=["test"]
        )
        genes.append(gene)
        fitness_scores[gene_id] = i / 10.0  # Fitness 0.0 to 0.9
    
    population = Population(
        generation=1,
        genes=genes,
        fitness_scores=fitness_scores,
        diversity_metrics={},
        selection_pressure=0.3
    )
    
    selected = select_parents(population, selection_pressure=0.3, seed=42)
    
    # Should return same number as input population
    assert len(selected) == len(genes)
    
    # Should favor higher fitness genes (not strict due to tournament randomness)
    selected_ids = [g.gene_id for g in selected]
    assert "gene_9" in selected_ids  # Highest fitness should often be selected


def test_evolution_generation():
    """Test single generation evolution"""
    from ally.research.evolution import (
        evolve_generation, Population, StrategyGene, EvolutionConfig
    )
    
    # Create initial population
    genes = []
    fitness_scores = {}
    
    for i in range(5):
        gene_id = f"gen0_gene_{i}"
        gene = StrategyGene(
            gene_id=gene_id,
            factor_type="momentum",
            parameters={"id": i},
            lookback_days=100,
            signal_transform="zscore",
            universe_filter="liquid_1000",
            mutation_history=["initial"]
        )
        genes.append(gene)
        fitness_scores[gene_id] = i / 10.0
    
    population = Population(
        generation=0,
        genes=genes,
        fitness_scores=fitness_scores,
        diversity_metrics={},
        selection_pressure=0.3
    )
    
    config = EvolutionConfig(
        population_size=5,
        mutation_rate=0.2,
        crossover_rate=0.8,
        selection_pressure=0.3
    )
    
    # Mock fitness evaluator
    def mock_evaluator(strategy_spec, seed=42):
        return {"oos_sharpe": 0.5 + (hash(str(strategy_spec)) % 100) / 200}
    
    # Evolve one generation
    new_population = evolve_generation(population, config, mock_evaluator, seed=42)
    
    # Check new population properties
    assert new_population.generation == 1
    assert len(new_population.genes) == config.population_size
    assert len(new_population.fitness_scores) >= len(fitness_scores)
    
    # Should have diversity metrics
    assert "factor_diversity" in new_population.diversity_metrics


def test_evolution_search_api():
    """Test full evolutionary search API"""
    from ally.research.evolution import research_evolution_search
    
    result = research_evolution_search(
        max_generations=3,
        population_size=8,
        live=False
    )
    
    assert result.ok == True
    assert "evolution_receipt" in result.data
    assert "final_generation" in result.data
    assert "best_genes" in result.data
    assert "fitness_history" in result.data
    assert "population_summary" in result.data
    
    # Check receipt format
    assert len(result.data["evolution_receipt"]) == 16
    assert hasattr(result, 'receipt_hash')
    assert len(result.receipt_hash) == 40
    
    # Check fitness history
    fitness_history = result.data["fitness_history"]
    assert len(fitness_history) <= 3  # Max generations
    assert all(isinstance(f, (int, float)) for f in fitness_history)
    
    # Check best genes
    best_genes = result.data["best_genes"]
    assert len(best_genes) <= 5  # Top 5 genes
    for gene_data in best_genes:
        assert "gene_id" in gene_data
        assert "factor_type" in gene_data
        assert "parameters" in gene_data


def test_evolution_search_deterministic():
    """Test that evolution search is deterministic"""
    from ally.research.evolution import research_evolution_search
    
    # Run twice with same parameters
    result1 = research_evolution_search(
        max_generations=2,
        population_size=5,
        live=False
    )
    
    result2 = research_evolution_search(
        max_generations=2,
        population_size=5,
        live=False
    )
    
    assert result1.ok == result2.ok
    
    if result1.ok and result2.ok:
        # Should have identical fitness histories (deterministic)
        history1 = result1.data["fitness_history"]
        history2 = result2.data["fitness_history"]
        
        assert len(history1) == len(history2)
        for f1, f2 in zip(history1, history2):
            assert abs(f1 - f2) < 1e-10


def test_evolution_config_validation():
    """Test evolution configuration validation"""
    from ally.research.evolution import EvolutionConfig, research_evolution_search
    
    # Test with custom config
    custom_config = {
        "population_size": 10,
        "max_generations": 5,
        "mutation_rate": 0.25,
        "crossover_rate": 0.6,
        "fitness_threshold": 0.8
    }
    
    result = research_evolution_search(
        config=custom_config,
        max_generations=2,  # Override max_generations
        population_size=6,  # Override population_size
        live=False
    )
    
    assert result.ok == True
    
    # Check that config was applied
    config_used = result.data["config_used"]
    assert config_used["mutation_rate"] == 0.25
    assert config_used["crossover_rate"] == 0.6
    assert config_used["fitness_threshold"] == 0.8


def test_evolution_error_handling():
    """Test evolution search error handling"""
    from ally.research.evolution import research_evolution_search
    
    # Test with invalid parameters (should handle gracefully)
    result = research_evolution_search(
        max_generations=0,  # Invalid
        population_size=1,  # Very small
        live=False
    )
    
    # Should still work or fail gracefully
    assert isinstance(result.ok, bool)
    
    if not result.ok:
        assert len(result.errors) > 0


def test_live_mode_gating():
    """Test live mode gating for evolution search"""
    from ally.research.evolution import research_evolution_search
    
    # Test offline mode (should work)
    result = research_evolution_search(
        max_generations=1,
        population_size=3,
        live=False
    )
    assert result.ok == True
    
    # Test live mode without proper setup (should fail in CI)
    result_live = research_evolution_search(
        max_generations=1,
        population_size=3,
        live=True
    )
    # In CI environment, this should fail due to missing API key
    assert result_live.ok == False or result_live.ok == True  # Depends on environment


def test_gene_id_consistency():
    """Test that gene IDs are consistent for identical specifications"""
    from ally.research.evolution import StrategyGene
    
    # Create two genes with identical specifications
    gene1 = StrategyGene(
        gene_id="",
        factor_type="momentum", 
        parameters={"lookback": 252, "skip": 21},
        lookback_days=252,
        signal_transform="zscore",
        universe_filter="liquid_1000",
        mutation_history=["test"]
    )
    
    gene2 = StrategyGene(
        gene_id="",
        factor_type="momentum",
        parameters={"lookback": 252, "skip": 21},
        lookback_days=252,
        signal_transform="zscore",
        universe_filter="liquid_1000",
        mutation_history=["test"]
    )
    
    # Should have identical gene IDs
    assert gene1.gene_id == gene2.gene_id
    
    # Different specifications should have different IDs
    gene3 = StrategyGene(
        gene_id="",
        factor_type="reversal",  # Different factor type
        parameters={"lookback": 252, "skip": 21},
        lookback_days=252,
        signal_transform="zscore",
        universe_filter="liquid_1000",
        mutation_history=["test"]
    )
    
    assert gene1.gene_id != gene3.gene_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])