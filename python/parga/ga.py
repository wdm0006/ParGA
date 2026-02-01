"""Unified Genetic Algorithm interface.

This module provides a single `GA` class that automatically selects
the optimal execution strategy based on fitness function cost and
available hardware.

Example:
    >>> from parga import GA
    >>> import numpy as np
    >>>
    >>> def fitness(genes):
    ...     return -np.sum(genes**2)
    >>>
    >>> # Just use GA - it figures out the best approach
    >>> result = GA(fitness, genome_length=10).run()
    >>> print(f"Best: {result.best_fitness}")
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Callable

import numpy as np

from parga._parga import (
    GeneticAlgorithm as RustGA,
)
from parga._parga import (
    IslandModel as RustIslandModel,
)
from parga.parallel import ParallelGA, ParallelIslandModel


class GAResult:
    """Unified result from genetic algorithm optimization."""

    def __init__(
        self,
        best_genes: np.ndarray,
        best_fitness: float,
        generations: int,
        fitness_history: list[float],
        strategy: str,
    ):
        self._best_genes = best_genes
        self.best_fitness = best_fitness
        self.generations = generations
        self.fitness_history = fitness_history
        self.strategy = strategy

    def best_genes(self) -> np.ndarray:
        """Return the best genome found."""
        if isinstance(self._best_genes, np.ndarray):
            return self._best_genes.copy()
        return np.array(self._best_genes)

    def __repr__(self) -> str:
        """Return string representation of result."""
        return (
            f"GAResult(best_fitness={self.best_fitness:.6f}, "
            f"generations={self.generations}, strategy='{self.strategy}')"
        )


class GA:
    """Unified Genetic Algorithm that auto-selects the optimal strategy.

    This class automatically determines whether to use:
    - Single-threaded Rust execution (for cheap fitness functions)
    - Parallel process pool (for expensive fitness functions)
    - Island model (for better exploration of complex landscapes)

    The decision is based on:
    1. Measured fitness function execution time
    2. Available CPU cores
    3. User hints (parallel, islands parameters)

    Args:
        fitness_fn: Function that takes a numpy array and returns a float.
                   Higher values are better (maximization).
        genome_length: Number of genes in each individual.
        population_size: Total population size (default: 100).
        generations: Number of generations to evolve (default: 100).
        bounds: Tuple of (lower, upper) bounds for all genes, or
               tuple of (lower_list, upper_list) for per-gene bounds.
        parallel: Force parallel execution. If None (default), auto-detect
                 based on fitness function cost.
        islands: Number of islands for island model. If > 1, uses island
                model with migration. Default is 1 (no islands).
        n_workers: Number of worker processes for parallel execution.
                  Default is number of CPU cores.
        mutation_rate: Probability of mutation per gene (default: 0.01).
        crossover_rate: Probability of crossover (default: 0.8).
        seed: Random seed for reproducibility.
        verbose: Print strategy selection info (default: False).

    Example:
        >>> # Simple usage - auto-selects best strategy
        >>> result = GA(fitness_fn, genome_length=10).run()
        >>>
        >>> # Force parallel for expensive fitness
        >>> result = GA(fitness_fn, genome_length=10, parallel=True).run()
        >>>
        >>> # Use island model for complex landscapes
        >>> result = GA(fitness_fn, genome_length=10, islands=4).run()
    """

    # Threshold in seconds - if fitness takes longer than this, use parallel
    # 0.5ms * 100 individuals * 100 generations = 5 seconds, worth parallelizing
    PARALLEL_THRESHOLD = 0.0005  # 0.5ms

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        genome_length: int,
        population_size: int = 100,
        generations: int = 100,
        bounds: tuple | None = None,
        parallel: bool | None = None,
        islands: int = 1,
        n_workers: int | None = None,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.8,
        elitism: int = 2,
        tournament_size: int = 3,
        migration_interval: int = 10,
        migration_count: int = 5,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.fitness_fn = fitness_fn
        self.genome_length = genome_length
        self.population_size = population_size
        self.generations = generations
        self.parallel = parallel
        self.islands = islands
        self.n_workers = n_workers or mp.cpu_count()
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self.seed = seed
        self.verbose = verbose

        # Parse bounds
        if bounds is None:
            self.lower_bounds = [-10.0] * genome_length
            self.upper_bounds = [10.0] * genome_length
        elif isinstance(bounds[0], (int, float)):
            # Single (lower, upper) for all genes
            self.lower_bounds = [float(bounds[0])] * genome_length
            self.upper_bounds = [float(bounds[1])] * genome_length
        else:
            # Per-gene bounds
            self.lower_bounds = list(bounds[0])
            self.upper_bounds = list(bounds[1])

        # Will be set after strategy selection
        self._strategy: str | None = None
        self._fitness_time: float | None = None

    def _measure_fitness_time(self) -> float:
        """Measure average fitness function execution time."""
        rng = np.random.default_rng(self.seed)
        lower = np.array(self.lower_bounds)
        upper = np.array(self.upper_bounds)

        # Run a few evaluations to get average time
        n_samples = 5
        times = []
        for _ in range(n_samples):
            genome = rng.uniform(lower, upper)
            start = time.perf_counter()
            _ = self.fitness_fn(genome)
            times.append(time.perf_counter() - start)

        return np.median(times)

    def _select_strategy(self) -> str:
        """Select the optimal execution strategy."""
        # If user explicitly set parallel preference, respect it
        if self.parallel is True:
            if self.islands > 1:
                return "parallel_island"
            return "parallel"
        elif self.parallel is False:
            if self.islands > 1:
                return "rust_island"
            return "rust"

        # Auto-detect based on fitness cost
        self._fitness_time = self._measure_fitness_time()

        if self.verbose:
            print(f"Fitness evaluation time: {self._fitness_time * 1000:.2f}ms")

        # Decision logic
        if self.islands > 1:
            # Island model requested
            if self._fitness_time > self.PARALLEL_THRESHOLD:
                return "parallel_island"
            return "rust_island"
        else:
            # Single population
            if self._fitness_time > self.PARALLEL_THRESHOLD:
                return "parallel"
            return "rust"

    def run(self) -> GAResult:
        """Run the genetic algorithm and return the result.

        Automatically selects the optimal execution strategy based on
        fitness function cost and configuration.

        Returns:
            GAResult with best solution, fitness history, and strategy used.
        """
        self._strategy = self._select_strategy()

        if self.verbose:
            print(f"Selected strategy: {self._strategy}")

        if self._strategy == "rust":
            return self._run_rust()
        elif self._strategy == "parallel":
            return self._run_parallel()
        elif self._strategy == "rust_island":
            return self._run_rust_island()
        elif self._strategy == "parallel_island":
            return self._run_parallel_island()
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

    def _run_rust(self) -> GAResult:
        """Run using Rust-based single-threaded GA."""
        ga = RustGA(
            fitness_fn=self.fitness_fn,
            genome_length=self.genome_length,
            population_size=self.population_size,
            generations=self.generations,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elitism=self.elitism,
            tournament_size=self.tournament_size,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            seed=self.seed,
        )
        result = ga.run()
        return GAResult(
            best_genes=np.array(result.best_genes()),
            best_fitness=result.best_fitness,
            generations=result.generations,
            fitness_history=list(result.fitness_history()),
            strategy="rust",
        )

    def _run_parallel(self) -> GAResult:
        """Run using parallel process pool."""
        ga = ParallelGA(
            fitness_fn=self.fitness_fn,
            genome_length=self.genome_length,
            population_size=self.population_size,
            generations=self.generations,
            n_workers=self.n_workers,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elitism=self.elitism,
            tournament_size=self.tournament_size,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            seed=self.seed,
        )
        result = ga.run()
        return GAResult(
            best_genes=result.best_genes(),
            best_fitness=result.best_fitness,
            generations=result.generations,
            fitness_history=result.fitness_history,
            strategy="parallel",
        )

    def _run_rust_island(self) -> GAResult:
        """Run using Rust-based island model."""
        island_pop = self.population_size // self.islands
        ga = RustIslandModel(
            fitness_fn=self.fitness_fn,
            genome_length=self.genome_length,
            num_islands=self.islands,
            island_population=island_pop,
            generations=self.generations,
            migration_interval=self.migration_interval,
            migration_count=self.migration_count,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elitism=self.elitism,
            tournament_size=self.tournament_size,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            seed=self.seed,
        )
        result = ga.run()
        return GAResult(
            best_genes=np.array(result.best_genes()),
            best_fitness=result.best_fitness,
            generations=result.generations,
            fitness_history=list(result.fitness_history()),
            strategy="rust_island",
        )

    def _run_parallel_island(self) -> GAResult:
        """Run using parallel island model."""
        island_pop = self.population_size // self.islands
        ga = ParallelIslandModel(
            fitness_fn=self.fitness_fn,
            genome_length=self.genome_length,
            num_islands=self.islands,
            island_population=island_pop,
            generations=self.generations,
            migration_interval=self.migration_interval,
            migration_count=self.migration_count,
            n_workers=self.n_workers,
            mutation_rate=self.mutation_rate,
            crossover_rate=self.crossover_rate,
            elitism=self.elitism,
            tournament_size=self.tournament_size,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            seed=self.seed,
        )
        result = ga.run()
        return GAResult(
            best_genes=result.best_genes(),
            best_fitness=result.best_fitness,
            generations=result.generations,
            fitness_history=result.fitness_history,
            strategy="parallel_island",
        )


def minimize(
    fitness_fn: Callable[[np.ndarray], float],
    genome_length: int,
    bounds: tuple | None = None,
    **kwargs,
) -> GAResult:
    """Minimize a function using genetic algorithm.

    Convenience function that wraps GA for minimization problems.
    Automatically negates the fitness function.

    Args:
        fitness_fn: Function to minimize. Takes numpy array, returns float.
        genome_length: Number of dimensions.
        bounds: Optional bounds as (lower, upper) or ([lowers], [uppers]).
        **kwargs: Additional arguments passed to GA.

    Returns:
        GAResult with best solution. Note: best_fitness is the negated
        (maximized) value; use -result.best_fitness for the actual minimum.

    Example:
        >>> def sphere(x):
        ...     return np.sum(x**2)
        >>> result = minimize(sphere, genome_length=10, bounds=(-5, 5))
        >>> print(f"Minimum value: {-result.best_fitness}")
    """

    # Wrap fitness to negate for minimization
    def neg_fitness(x):
        return -fitness_fn(x)

    ga = GA(neg_fitness, genome_length, bounds=bounds, **kwargs)
    return ga.run()


def maximize(
    fitness_fn: Callable[[np.ndarray], float],
    genome_length: int,
    bounds: tuple | None = None,
    **kwargs,
) -> GAResult:
    """Maximize a function using genetic algorithm.

    Convenience function that wraps GA for maximization problems.

    Args:
        fitness_fn: Function to maximize. Takes numpy array, returns float.
        genome_length: Number of dimensions.
        bounds: Optional bounds as (lower, upper) or ([lowers], [uppers]).
        **kwargs: Additional arguments passed to GA.

    Returns:
        GAResult with best solution.

    Example:
        >>> def neg_sphere(x):
        ...     return -np.sum(x**2)
        >>> result = maximize(neg_sphere, genome_length=10, bounds=(-5, 5))
    """
    ga = GA(fitness_fn, genome_length, bounds=bounds, **kwargs)
    return ga.run()
