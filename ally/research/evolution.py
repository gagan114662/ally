#!/usr/bin/env python3
"""
Evolutionary search engine for automated alpha discovery - Phase 6.1

Implements genetic programming with mutation/crossover operators to evolve
trading strategies. Seeds from academic factor priors and evolves through
fitness evaluation via Phase 5.x gates.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
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
            'normal': lambda mu, sigma: mu
        })(),
        'clip': lambda x, a, b: max(a, min(b, x))
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
class StrategyGene:
    """Represents a strategy gene with factor specifications"""
    gene_id: str
    factor_type: str  # 'momentum', 'reversal', 'value', 'quality', 'vol'
    parameters: Dict[str, Any]
    lookback_days: int
    signal_transform: str  # 'rank', 'zscore', 'raw', 'winsorize'
    universe_filter: str
    mutation_history: List[str]
    
    def __post_init__(self):
        if not self.gene_id:
            content = f"{self.factor_type}:{self.parameters}:{self.lookback_days}"
            self.gene_id = hashlib.sha1(content.encode()).hexdigest()[:16]


@dataclass
class Population:
    """Represents a population of strategy genes"""
    generation: int
    genes: List[StrategyGene]
    fitness_scores: Dict[str, float]
    diversity_metrics: Dict[str, Any]
    selection_pressure: float
    
    def get_best_genes(self, n: int) -> List[StrategyGene]:
        """Get top N genes by fitness"""
        if not self.fitness_scores:
            return self.genes[:n]
            
        sorted_genes = sorted(
            self.genes,
            key=lambda g: self.fitness_scores.get(g.gene_id, 0),
            reverse=True
        )
        return sorted_genes[:n]


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary search"""
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    selection_pressure: float = 0.3
    fitness_threshold: float = 0.5  # Min OOS Sharpe to survive
    diversity_threshold: float = 0.8  # Max correlation to avoid clones
    elite_preserve_pct: float = 0.1  # Top % to preserve unchanged
    random_inject_pct: float = 0.05  # Random injection rate


def create_academic_priors() -> List[StrategyGene]:
    """Create initial population seeded from academic factor literature"""
    priors = []
    
    # Momentum factors (Jegadeesh & Titman 1993)
    priors.append(StrategyGene(
        gene_id="",
        factor_type="momentum",
        parameters={"lookback": 252, "skip": 21, "decay": "linear"},
        lookback_days=252,
        signal_transform="zscore",
        universe_filter="liquid_1000",
        mutation_history=["academic_prior"]
    ))
    
    # Short-term reversal (Jegadeesh 1990)
    priors.append(StrategyGene(
        gene_id="", 
        factor_type="reversal",
        parameters={"lookback": 5, "smoothing": "none"},
        lookback_days=5,
        signal_transform="rank",
        universe_filter="liquid_1000", 
        mutation_history=["academic_prior"]
    ))
    
    # Value - Book-to-Market (Fama & French 1992)
    priors.append(StrategyGene(
        gene_id="",
        factor_type="value",
        parameters={"metric": "book_to_market", "rebalance": "quarterly"},
        lookback_days=252,
        signal_transform="zscore",
        universe_filter="liquid_1000",
        mutation_history=["academic_prior"]
    ))
    
    # Quality - ROE (Novy-Marx 2013)
    priors.append(StrategyGene(
        gene_id="",
        factor_type="quality", 
        parameters={"metric": "roe", "stability": "3y"},
        lookback_days=252,
        signal_transform="rank",
        universe_filter="liquid_1000",
        mutation_history=["academic_prior"]
    ))
    
    # Low volatility (Ang et al. 2006)
    priors.append(StrategyGene(
        gene_id="",
        factor_type="vol",
        parameters={"metric": "realized_vol", "window": 60},
        lookback_days=60,
        signal_transform="rank",
        universe_filter="liquid_1000",
        mutation_history=["academic_prior"]
    ))
    
    return priors


def mutate_gene(gene: StrategyGene, mutation_rate: float, seed: int = None) -> StrategyGene:
    """Apply mutation operators to a strategy gene"""
    if seed is not None:
        np.random.seed(seed)
        
    if np.random.random() > mutation_rate:
        return gene
        
    # Copy gene for mutation
    new_params = gene.parameters.copy()
    new_mutation_history = gene.mutation_history.copy()
    
    # Mutation operators
    mutation_type = np.random.choice([
        "param_tweak", "lookback_change", "transform_change", "universe_change"
    ])
    
    if mutation_type == "param_tweak":
        # Randomly adjust parameter values
        if gene.factor_type == "momentum" and "lookback" in new_params:
            # Momentum lookback: 60-504 days
            current = new_params["lookback"]
            delta = int(np.random.normal(0, 30))
            new_params["lookback"] = np.clip(current + delta, 60, 504)
            
        elif gene.factor_type == "reversal" and "lookback" in new_params:
            # Reversal lookback: 1-21 days
            current = new_params["lookback"]
            delta = int(np.random.normal(0, 3))
            new_params["lookback"] = np.clip(current + delta, 1, 21)
            
        new_mutation_history.append(f"param_tweak_{mutation_type}")
        
    elif mutation_type == "lookback_change":
        # Adjust gene-level lookback
        delta = int(np.random.normal(0, 21))
        new_lookback = np.clip(gene.lookback_days + delta, 5, 504)
        
        mutated_gene = StrategyGene(
            gene_id="",  # Will be regenerated
            factor_type=gene.factor_type,
            parameters=new_params,
            lookback_days=new_lookback,
            signal_transform=gene.signal_transform,
            universe_filter=gene.universe_filter,
            mutation_history=new_mutation_history + ["lookback_change"]
        )
        return mutated_gene
        
    elif mutation_type == "transform_change":
        # Change signal transformation
        transforms = ["rank", "zscore", "raw", "winsorize"]
        current_idx = transforms.index(gene.signal_transform) if gene.signal_transform in transforms else 0
        new_idx = (current_idx + 1) % len(transforms)
        
        mutated_gene = StrategyGene(
            gene_id="",
            factor_type=gene.factor_type,
            parameters=new_params,
            lookback_days=gene.lookback_days,
            signal_transform=transforms[new_idx],
            universe_filter=gene.universe_filter,
            mutation_history=new_mutation_history + ["transform_change"]
        )
        return mutated_gene
        
    elif mutation_type == "universe_change":
        # Change universe filter
        universes = ["liquid_500", "liquid_1000", "liquid_2000", "all"]
        current_idx = universes.index(gene.universe_filter) if gene.universe_filter in universes else 0
        new_idx = (current_idx + 1) % len(universes)
        
        mutated_gene = StrategyGene(
            gene_id="",
            factor_type=gene.factor_type,
            parameters=new_params,
            lookback_days=gene.lookback_days,
            signal_transform=gene.signal_transform,
            universe_filter=universes[new_idx],
            mutation_history=new_mutation_history + ["universe_change"]
        )
        return mutated_gene
    
    # Default: return gene with parameter mutations
    mutated_gene = StrategyGene(
        gene_id="",
        factor_type=gene.factor_type,
        parameters=new_params,
        lookback_days=gene.lookback_days,
        signal_transform=gene.signal_transform,
        universe_filter=gene.universe_filter,
        mutation_history=new_mutation_history
    )
    return mutated_gene


def crossover_genes(parent1: StrategyGene, parent2: StrategyGene, seed: int = None) -> Tuple[StrategyGene, StrategyGene]:
    """Apply crossover operator to create offspring genes"""
    if seed is not None:
        np.random.seed(seed)
    
    # Simple parameter crossover
    child1_params = {}
    child2_params = {}
    
    all_keys = set(parent1.parameters.keys()) | set(parent2.parameters.keys())
    
    for key in all_keys:
        if np.random.random() < 0.5:
            child1_params[key] = parent1.parameters.get(key, parent2.parameters.get(key))
            child2_params[key] = parent2.parameters.get(key, parent1.parameters.get(key))
        else:
            child1_params[key] = parent2.parameters.get(key, parent1.parameters.get(key))
            child2_params[key] = parent1.parameters.get(key, parent2.parameters.get(key))
    
    # Crossover other attributes
    child1 = StrategyGene(
        gene_id="",
        factor_type=parent1.factor_type if np.random.random() < 0.5 else parent2.factor_type,
        parameters=child1_params,
        lookback_days=parent1.lookback_days if np.random.random() < 0.5 else parent2.lookback_days,
        signal_transform=parent1.signal_transform if np.random.random() < 0.5 else parent2.signal_transform,
        universe_filter=parent1.universe_filter if np.random.random() < 0.5 else parent2.universe_filter,
        mutation_history=["crossover"]
    )
    
    child2 = StrategyGene(
        gene_id="",
        factor_type=parent2.factor_type if np.random.random() < 0.5 else parent1.factor_type,
        parameters=child2_params,
        lookback_days=parent2.lookback_days if np.random.random() < 0.5 else parent1.lookback_days,
        signal_transform=parent2.signal_transform if np.random.random() < 0.5 else parent1.signal_transform,
        universe_filter=parent2.universe_filter if np.random.random() < 0.5 else parent1.universe_filter,
        mutation_history=["crossover"]
    )
    
    return child1, child2


def evaluate_fitness(gene: StrategyGene, fitness_evaluator: Callable, seed: int = 42) -> float:
    """Evaluate gene fitness using Phase 5.x validation gates"""
    try:
        # Convert gene to strategy specification
        strategy_spec = {
            "gene_id": gene.gene_id,
            "factor_type": gene.factor_type,
            "parameters": gene.parameters,
            "lookback_days": gene.lookback_days,
            "signal_transform": gene.signal_transform,
            "universe_filter": gene.universe_filter
        }
        
        # Use fitness evaluator (should run through Phase 5.x gates)
        fitness_result = fitness_evaluator(strategy_spec, seed=seed)
        
        # Fitness is OOS Sharpe from walk-forward validation
        return fitness_result.get("oos_sharpe", 0.0)
        
    except Exception as e:
        # Failed evaluation = 0 fitness
        return 0.0


def select_parents(population: Population, selection_pressure: float, seed: int = None) -> List[StrategyGene]:
    """Select parents for next generation using tournament selection"""
    if seed is not None:
        np.random.seed(seed)
    
    selected = []
    pop_size = len(population.genes)
    tournament_size = max(2, int(pop_size * selection_pressure))
    
    for _ in range(pop_size):
        # Tournament selection
        tournament = np.random.choice(population.genes, size=tournament_size, replace=False)
        
        # Select winner by fitness
        winner = max(tournament, key=lambda g: population.fitness_scores.get(g.gene_id, 0))
        selected.append(winner)
    
    return selected


def calculate_diversity_metrics(population: Population) -> Dict[str, Any]:
    """Calculate population diversity metrics"""
    genes = population.genes
    
    if not genes:
        return {"factor_diversity": 0, "param_diversity": 0, "transform_diversity": 0}
    
    # Factor type diversity
    factor_types = [g.factor_type for g in genes]
    unique_factors = len(set(factor_types))
    factor_diversity = unique_factors / len(factor_types) if factor_types else 0
    
    # Parameter diversity (simplified)
    param_strings = [str(sorted(g.parameters.items())) for g in genes]
    unique_params = len(set(param_strings))
    param_diversity = unique_params / len(param_strings) if param_strings else 0
    
    # Transform diversity
    transforms = [g.signal_transform for g in genes]
    unique_transforms = len(set(transforms))
    transform_diversity = unique_transforms / len(transforms) if transforms else 0
    
    return {
        "factor_diversity": factor_diversity,
        "param_diversity": param_diversity,
        "transform_diversity": transform_diversity,
        "unique_factors": unique_factors,
        "unique_params": unique_params,
        "unique_transforms": unique_transforms
    }


def evolve_generation(
    population: Population,
    config: EvolutionConfig,
    fitness_evaluator: Callable,
    seed: int = 42
) -> Population:
    """Evolve population by one generation"""
    if seed is not None:
        np.random.seed(seed)
    
    # Evaluate fitness for all genes
    fitness_scores = {}
    for gene in population.genes:
        if gene.gene_id not in population.fitness_scores:
            fitness_scores[gene.gene_id] = evaluate_fitness(gene, fitness_evaluator, seed)
        else:
            fitness_scores[gene.gene_id] = population.fitness_scores[gene.gene_id]
    
    # Update population fitness
    population.fitness_scores.update(fitness_scores)
    
    # Select elite (preserve best genes unchanged)
    n_elite = max(1, int(config.population_size * config.elite_preserve_pct))
    elite_genes = population.get_best_genes(n_elite)
    
    # Select parents for breeding
    parents = select_parents(population, config.selection_pressure, seed)
    
    # Generate offspring through crossover and mutation
    offspring = []
    n_offspring_needed = config.population_size - n_elite
    
    for i in range(0, n_offspring_needed, 2):
        if i + 1 < len(parents):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            
            if np.random.random() < config.crossover_rate:
                child1, child2 = crossover_genes(parent1, parent2, seed + i)
            else:
                child1, child2 = parent1, parent2
            
            # Apply mutations
            child1 = mutate_gene(child1, config.mutation_rate, seed + i * 2)
            child2 = mutate_gene(child2, config.mutation_rate, seed + i * 2 + 1)
            
            offspring.extend([child1, child2])
    
    # Trim to exact size needed
    offspring = offspring[:n_offspring_needed]
    
    # Add random injection
    n_random = max(1, int(config.population_size * config.random_inject_pct))
    if n_random > 0:
        # Replace worst performers with random genes
        random_genes = create_academic_priors()[:n_random]
        if len(offspring) >= n_random:
            offspring = offspring[:-n_random] + random_genes
    
    # Combine elite + offspring
    new_genes = elite_genes + offspring
    
    # Create new population
    new_population = Population(
        generation=population.generation + 1,
        genes=new_genes[:config.population_size],
        fitness_scores=fitness_scores,
        diversity_metrics=calculate_diversity_metrics(
            Population(population.generation + 1, new_genes[:config.population_size], {}, {}, 0)
        ),
        selection_pressure=config.selection_pressure
    )
    
    return new_population


@register_tool("evo.search")
def research_evolution_search(
    config: Optional[Dict] = None,
    fitness_evaluator: Optional[Callable] = None,
    max_generations: int = 10,
    population_size: int = 20,
    live: bool = True
) -> Result:
    """
    Run evolutionary search for strategy discovery
    
    Args:
        config: Evolution configuration parameters
        fitness_evaluator: Function to evaluate gene fitness via Phase 5.x gates
        max_generations: Maximum number of generations to evolve
        population_size: Size of gene population
        live: Enable live mode (requires ALLY_LIVE=1)
    
    Returns:
        Result with evolution statistics and best genes
    """
    try:
        # Gate live mode
        if live:
            check_live_mode_allowed(
                live=live,
                api_key=os.getenv("EVOLUTION_API_KEY", "not_set"),
                service_name="Evolution Search"
            )
        
        # Default configuration
        evo_config = EvolutionConfig(
            population_size=population_size,
            max_generations=max_generations,
            mutation_rate=0.15,
            crossover_rate=0.7,
            selection_pressure=0.3,
            fitness_threshold=0.5,
            diversity_threshold=0.8,
            elite_preserve_pct=0.1,
            random_inject_pct=0.05
        )
        
        if config:
            for key, value in config.items():
                if hasattr(evo_config, key):
                    setattr(evo_config, key, value)
        
        # Default fitness evaluator (mock for CI)
        if fitness_evaluator is None:
            def default_evaluator(strategy_spec, seed=42):
                # Mock evaluator - returns random fitness
                np.random.seed(seed + hash(str(strategy_spec)) % 1000)
                return {"oos_sharpe": np.random.uniform(0, 1.5)}
            fitness_evaluator = default_evaluator
        
        # Initialize population with academic priors
        initial_genes = create_academic_priors()
        
        # Pad to desired population size
        while len(initial_genes) < population_size:
            # Create variations of existing genes through mutation
            base_gene = np.random.choice(initial_genes)
            mutated = mutate_gene(base_gene, 0.5, seed=len(initial_genes))
            initial_genes.append(mutated)
        
        initial_genes = initial_genes[:population_size]
        
        # Create initial population
        current_population = Population(
            generation=0,
            genes=initial_genes,
            fitness_scores={},
            diversity_metrics=calculate_diversity_metrics(
                Population(0, initial_genes, {}, {}, 0)
            ),
            selection_pressure=evo_config.selection_pressure
        )
        
        # Evolution loop
        best_fitness_history = []
        diversity_history = []
        
        for generation in range(max_generations):
            # Evolve population
            current_population = evolve_generation(
                current_population,
                evo_config,
                fitness_evaluator,
                seed=42 + generation
            )
            
            # Track metrics
            best_fitness = max(current_population.fitness_scores.values()) if current_population.fitness_scores else 0
            best_fitness_history.append(best_fitness)
            diversity_history.append(current_population.diversity_metrics)
            
            # Early stopping if fitness threshold met
            if best_fitness >= evo_config.fitness_threshold:
                break
        
        # Get final results
        best_genes = current_population.get_best_genes(5)
        
        # Generate receipt
        evolution_data = {
            "max_generations": max_generations,
            "final_generation": current_population.generation,
            "population_size": population_size,
            "best_fitness": max(best_fitness_history) if best_fitness_history else 0,
            "final_diversity": current_population.diversity_metrics,
            "config": asdict(evo_config)
        }
        
        receipt_hash = generate_receipt("evo.search", evolution_data)
        
        return Result(
            ok=True,
            data={
                "evolution_receipt": receipt_hash[:16],
                "final_generation": current_population.generation,
                "best_genes": [asdict(gene) for gene in best_genes],
                "fitness_history": best_fitness_history,
                "diversity_history": diversity_history,
                "population_summary": {
                    "total_genes": len(current_population.genes),
                    "fitness_scores": len(current_population.fitness_scores),
                    "best_fitness": max(current_population.fitness_scores.values()) if current_population.fitness_scores else 0,
                    "mean_fitness": sum(current_population.fitness_scores.values()) / len(current_population.fitness_scores) if current_population.fitness_scores else 0
                },
                "config_used": asdict(evo_config)
            },
            receipt_hash=receipt_hash
        )
        
    except LiveModeError as e:
        return Result(ok=False, errors=[str(e)])
    except Exception as e:
        return Result(ok=False, errors=[f"Evolution search failed: {str(e)}"])


if __name__ == "__main__":
    # Test evolution search
    result = research_evolution_search(
        max_generations=3,
        population_size=10,
        live=False
    )
    
    if result.ok:
        print("✅ Evolution search completed")
        print(f"Receipt: {result.data['evolution_receipt']}")
        print(f"Final generation: {result.data['final_generation']}")
        print(f"Best fitness: {result.data['population_summary']['best_fitness']:.3f}")
    else:
        print("❌ Evolution search failed")
        for error in result.errors:
            print(f"Error: {error}")