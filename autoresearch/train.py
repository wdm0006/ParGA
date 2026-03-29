#!/usr/bin/env python3
"""Autoresearch train.py — the ONLY file the agent modifies.

This file defines how ParGA runs on each benchmark problem. The benchmark
harness calls run() with a fitness function, dimensionality, bounds, and seed.

The agent should experiment with:
  - GA configuration (population size, generations, rates)
  - Operator choices (selection, crossover, mutation methods)
  - Island model configuration (topology, migration)
  - Algorithmic improvements to the ParGA framework itself
  - New operators or strategies implemented in Rust or Python

When improving the framework, edit the ParGA source code (src/*.rs,
python/parga/*.py) AND update this file to use the new features.
"""

import numpy as np
from parga import GA, GAResult, CrossoverMethod, MutationMethod


def run(
    fitness_fn,
    dims: int,
    bounds: tuple,
    seed: int,
) -> GAResult:
    """Run the GA on a single benchmark problem.

    Args:
        fitness_fn: Fitness function (takes np.ndarray, returns float, higher=better).
        dims: Number of dimensions.
        bounds: Tuple of (lower, upper) bounds for all genes.
        seed: Random seed for reproducibility.

    Returns:
        GAResult from the optimization.
    """
    result = GA(
        fitness_fn=fitness_fn,
        genome_length=dims,
        population_size=200,
        generations=500,
        bounds=bounds,
        mutation_rate=0.02,
        crossover_rate=0.9,
        elitism=4,
        tournament_size=5,
        parallel=False,
        seed=seed,
        crossover_method=CrossoverMethod.blend(alpha=0.5),
        mutation_method=MutationMethod.polynomial(eta=20.0),
        early_stopping=40,
    ).run()

    return result
