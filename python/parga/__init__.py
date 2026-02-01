"""ParGA - High-Performance Parallel Genetic Algorithm Library.

A Rust-powered genetic algorithm library with Python bindings,
featuring automatic strategy selection and island model parallelization.

Example:
    >>> import numpy as np
    >>> from parga import GA, minimize
    >>>
    >>> # Define a fitness function
    >>> def sphere(x):
    ...     return np.sum(x**2)
    >>>
    >>> # Simple usage - GA auto-selects the best strategy
    >>> result = minimize(sphere, genome_length=10, bounds=(-5, 5))
    >>> print(f"Minimum: {-result.best_fitness:.6f}")
    >>>
    >>> # Or use GA directly for more control
    >>> def fitness(genes):
    ...     return -np.sum(genes**2)  # Maximize negative = minimize
    >>>
    >>> result = GA(fitness, genome_length=10, bounds=(-5, 5)).run()
    >>>
    >>> # Force parallel execution for expensive fitness functions
    >>> result = GA(fitness, genome_length=10, parallel=True).run()
    >>>
    >>> # Use island model for complex optimization landscapes
    >>> result = GA(fitness, genome_length=10, islands=4).run()
"""

from parga._parga import (
    CrossoverMethod,
    # Result types
    GaResult,
    # Low-level classes (for advanced users)
    GeneticAlgorithm,
    IslandModel,
    IslandResult,
    MigrationTopology,
    MutationMethod,
    # Configuration classes
    SelectionMethod,
    ackley,
    griewank,
    rastrigin,
    rosenbrock,
    schwefel,
    # Benchmark functions
    sphere,
)
from parga.ga import GA, GAResult, maximize, minimize
from parga.parallel import ParallelGA, ParallelGAResult, ParallelIslandModel

__version__ = "0.1.0"
__all__ = [
    "GA",
    "CrossoverMethod",
    "GAResult",
    "GaResult",
    "GeneticAlgorithm",
    "IslandModel",
    "IslandResult",
    "MigrationTopology",
    "MutationMethod",
    "ParallelGA",
    "ParallelGAResult",
    "ParallelIslandModel",
    "SelectionMethod",
    "ackley",
    "griewank",
    "maximize",
    "minimize",
    "rastrigin",
    "rosenbrock",
    "schwefel",
    "sphere",
]
