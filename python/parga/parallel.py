"""Parallel fitness evaluation using process pools.

This module provides parallel fitness evaluation for Python fitness functions
by using multiprocessing to bypass the GIL limitation.

Example:
    >>> from parga.parallel import ParallelGA
    >>> import numpy as np
    >>>
    >>> def fitness(genes):
    ...     return -np.sum(genes**2)
    >>>
    >>> ga = ParallelGA(
    ...     fitness_fn=fitness,
    ...     genome_length=10,
    ...     population_size=100,
    ...     generations=100,
    ...     n_workers=4,
    ... )
    >>> result = ga.run()
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

# Use cloudpickle for better function serialization (handles lambdas, closures)
import cloudpickle
import numpy as np

from ._validation import validate_ga_config, validate_island_config


_worker_fitness_fn = None


def _dumps_fitness_fn(fitness_fn: Callable) -> bytes:
    """Serialize a fitness function for use in worker processes.

    Functions defined at module level are pickled *by reference* by default,
    which requires the defining module to be importable on the worker's
    ``sys.path``. That fails for fitness functions defined in test suites,
    scripts, or notebooks (e.g. a ``tests`` package or ``__main__``) that are
    not installed. To make workers robust regardless of where the function
    lives, force cloudpickle to serialize its defining module *by value*.
    """
    module_name = getattr(fitness_fn, "__module__", None)
    module = sys.modules.get(module_name) if module_name else None
    # ``__main__`` is already pickled by value by cloudpickle. Only real
    # modules with a filesystem source can be registered by value.
    if module is not None and module_name != "__main__" and hasattr(module, "__file__"):
        cloudpickle.register_pickle_by_value(module)
        try:
            return cloudpickle.dumps(fitness_fn)
        finally:
            cloudpickle.unregister_pickle_by_value(module)
    return cloudpickle.dumps(fitness_fn)


def _worker_init(fitness_fn_bytes: bytes) -> None:
    """Initialize worker process with deserialized fitness function.

    Called once per worker at pool startup via ProcessPoolExecutor initializer.
    """
    global _worker_fitness_fn  # noqa: PLW0603
    _worker_fitness_fn = cloudpickle.loads(fitness_fn_bytes)


def _evaluate_batch_worker(genomes: list) -> list[float]:
    """Worker function that evaluates a batch of genomes.

    This runs in a separate process. Uses the cached fitness function
    deserialized once at worker startup via _worker_init.
    """
    return [_worker_fitness_fn(genome) for genome in genomes]


class ParallelGA:
    """Genetic Algorithm with parallel fitness evaluation using process pools.

    This class provides true parallel execution of Python fitness functions
    by using multiprocessing. Each worker process has its own Python
    interpreter, bypassing the GIL.

    Requirements:
        - Fitness function must be a pure function (no side effects)
        - Fitness function must be picklable (use cloudpickle for lambdas)
        - Fitness function should not rely on global mutable state

    Args:
        fitness_fn: A callable that takes a numpy array and returns a float.
        genome_length: Length of each genome.
        population_size: Number of individuals in the population.
        generations: Number of generations to evolve.
        n_workers: Number of worker processes. Defaults to CPU count.
        mutation_rate: Probability of mutation per gene.
        crossover_rate: Probability of crossover.
        elitism: Number of elite individuals to preserve.
        lower_bounds: Lower bounds for each gene.
        upper_bounds: Upper bounds for each gene.
        seed: Random seed for reproducibility.
        chunk_size: Number of individuals per batch sent to workers.
    """

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        genome_length: int,
        population_size: int = 100,
        generations: int = 100,
        n_workers: int | None = None,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.8,
        elitism: int = 2,
        tournament_size: int = 3,
        lower_bounds: list[float] | None = None,
        upper_bounds: list[float] | None = None,
        seed: int | None = None,
        chunk_size: int | None = None,
    ):
        warnings.warn(
            "ParallelGA is deprecated. Use GA() instead — on free-threaded "
            "Python (3.13t+) it automatically uses Rust/rayon parallelism, "
            "and on GIL Python it falls back to ProcessPoolExecutor.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.fitness_fn = fitness_fn
        self.genome_length = genome_length
        self.population_size = population_size
        self.generations = generations
        self.n_workers = n_workers or mp.cpu_count()
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.lower_bounds = (
            lower_bounds if lower_bounds is not None else [-10.0] * genome_length
        )
        self.upper_bounds = (
            upper_bounds if upper_bounds is not None else [10.0] * genome_length
        )
        self.seed = seed
        self.chunk_size = chunk_size or max(1, population_size // (self.n_workers * 2))

        validate_ga_config(
            genome_length=genome_length,
            population_size=population_size,
            elitism=elitism,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )

    def _create_random_population(self, rng: np.random.Generator) -> list[np.ndarray]:
        """Create initial random population."""
        lower = np.array(self.lower_bounds)
        upper = np.array(self.upper_bounds)
        return [rng.uniform(lower, upper) for _ in range(self.population_size)]

    def _evaluate_parallel(
        self,
        population: list[np.ndarray],
        executor: ProcessPoolExecutor,
        existing_fitness: list[float | None] | None = None,
    ) -> list[float]:
        """Evaluate fitness for individuals in parallel.

        Args:
            population: List of genomes to evaluate.
            executor: Process pool executor with pre-initialized workers.
            existing_fitness: Optional list parallel to population. Entries
                that are not None are carried forward (e.g. elites); only
                None entries are sent to workers for evaluation.
        """
        if existing_fitness is not None:
            # Only evaluate individuals without fitness (offspring)
            indices_to_eval = [
                i for i, f in enumerate(existing_fitness) if f is None
            ]
            if not indices_to_eval:
                return existing_fitness  # All have fitness already

            genomes_to_eval = [population[i] for i in indices_to_eval]
        else:
            indices_to_eval = None
            genomes_to_eval = population

        # Create batches of genomes only (no function bytes)
        batches = []
        for i in range(0, len(genomes_to_eval), self.chunk_size):
            batches.append(genomes_to_eval[i : i + self.chunk_size])

        # Submit batches to workers
        futures = [executor.submit(_evaluate_batch_worker, batch) for batch in batches]

        # Collect results
        evaluated_fitness = []
        for future in futures:
            evaluated_fitness.extend(future.result())

        if existing_fitness is not None:
            # Merge evaluated results back into full fitness list
            result = list(existing_fitness)
            for idx, fit in zip(indices_to_eval, evaluated_fitness):
                result[idx] = fit
            return result

        return evaluated_fitness

    def _tournament_select(
        self,
        population: list[np.ndarray],
        fitness: list[float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Select an individual using tournament selection."""
        indices = rng.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = max(indices, key=lambda i: fitness[i])
        return population[best_idx].copy()

    def _crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Blend crossover (BLX-alpha)."""
        if rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        alpha = 0.5
        diff = np.abs(parent1 - parent2)
        lower = np.minimum(parent1, parent2) - alpha * diff
        upper = np.maximum(parent1, parent2) + alpha * diff

        child1 = rng.uniform(lower, upper)
        child2 = rng.uniform(lower, upper)

        # Clamp to bounds
        child1 = np.clip(child1, self.lower_bounds, self.upper_bounds)
        child2 = np.clip(child2, self.lower_bounds, self.upper_bounds)

        return child1, child2

    def _mutate(self, individual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Gaussian mutation."""
        mask = rng.random(len(individual)) < self.mutation_rate
        if np.any(mask):
            sigma = (np.array(self.upper_bounds) - np.array(self.lower_bounds)) * 0.1
            individual[mask] += rng.normal(0, sigma[mask])
            individual = np.clip(individual, self.lower_bounds, self.upper_bounds)
        return individual

    def run(self) -> ParallelGAResult:
        """Run the genetic algorithm with parallel fitness evaluation.

        Returns:
            ParallelGAResult with the best solution found.
        """
        rng = np.random.default_rng(self.seed)

        # Initialize population
        population = self._create_random_population(rng)
        fitness_history = []
        best_individual = None
        best_fitness = float("-inf")

        # Serialize fitness function once for the entire run
        fitness_fn_bytes = _dumps_fitness_fn(self.fitness_fn)

        # Use spawn to ensure clean worker processes
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(fitness_fn_bytes,),
        ) as executor:
            # Evaluate initial population
            fitness = self._evaluate_parallel(population, executor)

            # Track best
            for i, f in enumerate(fitness):
                if f > best_fitness:
                    best_fitness = f
                    best_individual = population[i].copy()
            fitness_history.append(best_fitness)

            # Evolution loop
            for _gen in range(self.generations):
                # Partial sort: only need top-k elites, not full sort
                if self.elitism > 0 and self.elitism < len(fitness):
                    fitness_arr = np.array(fitness)
                    # argpartition gives indices such that the top-k are in
                    # the first k positions (unordered among themselves)
                    elite_indices = np.argpartition(
                        -fitness_arr, self.elitism
                    )[: self.elitism]
                else:
                    elite_indices = list(range(self.elitism))

                # Create new population and track which have known fitness
                new_population = []
                new_fitness: list[float | None] = []

                # Elitism: keep best individuals with their fitness
                for i in elite_indices:
                    new_population.append(population[i].copy())
                    new_fitness.append(fitness[i])

                # Generate offspring (fitness unknown = None)
                while len(new_population) < self.population_size:
                    parent1 = self._tournament_select(population, fitness, rng)
                    parent2 = self._tournament_select(population, fitness, rng)

                    child1, child2 = self._crossover(parent1, parent2, rng)

                    child1 = self._mutate(child1, rng)
                    child2 = self._mutate(child2, rng)

                    new_population.append(child1)
                    new_fitness.append(None)
                    if len(new_population) < self.population_size:
                        new_population.append(child2)
                        new_fitness.append(None)

                population = new_population

                # Evaluate only offspring (elites keep their fitness)
                fitness = self._evaluate_parallel(
                    population, executor, existing_fitness=new_fitness
                )

                # Track best
                for i, f in enumerate(fitness):
                    if f > best_fitness:
                        best_fitness = f
                        best_individual = population[i].copy()
                fitness_history.append(best_fitness)

        return ParallelGAResult(
            best_genes=best_individual,
            best_fitness=best_fitness,
            generations=self.generations,
            fitness_history=fitness_history,
        )


class ParallelGAResult:
    """Result from parallel genetic algorithm run."""

    def __init__(
        self,
        best_genes: np.ndarray,
        best_fitness: float,
        generations: int,
        fitness_history: list[float],
    ):
        self._best_genes = best_genes
        self.best_fitness = best_fitness
        self.generations = generations
        self.fitness_history = fitness_history

    def best_genes(self) -> np.ndarray:
        """Return the best genome found."""
        return self._best_genes.copy()

    def __repr__(self) -> str:
        """Return string representation of result."""
        return (
            f"ParallelGAResult(best_fitness={self.best_fitness:.6f}, "
            f"generations={self.generations})"
        )


class ParallelIslandModel:
    """Island model with parallel fitness evaluation.

    Each island evolves independently with periodic migration between islands.
    Fitness evaluation uses process pools for true parallelism.

    Args:
        fitness_fn: A callable that takes a numpy array and returns a float.
        genome_length: Length of each genome.
        num_islands: Number of islands.
        island_population: Population size per island.
        generations: Total generations to evolve.
        migration_interval: Generations between migrations.
        migration_count: Number of individuals to migrate.
        n_workers: Number of worker processes for fitness evaluation.
        **kwargs: Additional arguments passed to each island's GA.
    """

    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        genome_length: int,
        num_islands: int = 4,
        island_population: int = 50,
        generations: int = 100,
        migration_interval: int = 10,
        migration_count: int = 5,
        n_workers: int | None = None,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.8,
        elitism: int = 2,
        tournament_size: int = 3,
        lower_bounds: list[float] | None = None,
        upper_bounds: list[float] | None = None,
        seed: int | None = None,
    ):
        warnings.warn(
            "ParallelIslandModel is deprecated. Use GA(islands=N) instead — "
            "on free-threaded Python (3.13t+) it automatically uses Rust/rayon "
            "parallelism, and on GIL Python it falls back to ProcessPoolExecutor.",
            DeprecationWarning,
            stacklevel=2,
        )

        self.fitness_fn = fitness_fn
        self.genome_length = genome_length
        self.num_islands = num_islands
        self.island_population = island_population
        self.generations = generations
        self.migration_interval = migration_interval
        self.migration_count = migration_count
        self.n_workers = n_workers or mp.cpu_count()
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.lower_bounds = (
            lower_bounds if lower_bounds is not None else [-10.0] * genome_length
        )
        self.upper_bounds = (
            upper_bounds if upper_bounds is not None else [10.0] * genome_length
        )
        self.seed = seed

        validate_island_config(
            genome_length=genome_length,
            num_islands=num_islands,
            island_population=island_population,
            migration_interval=migration_interval,
            migration_count=migration_count,
            elitism=elitism,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )

    def run(self) -> ParallelGAResult:
        """Run the island model with parallel fitness evaluation."""
        rng = np.random.default_rng(self.seed)

        # Initialize islands (each is a list of individuals)
        islands = []
        island_fitness = []
        lower = np.array(self.lower_bounds)
        upper = np.array(self.upper_bounds)

        for _ in range(self.num_islands):
            island = [rng.uniform(lower, upper) for _ in range(self.island_population)]
            islands.append(island)
            island_fitness.append([0.0] * self.island_population)

        best_individual = None
        best_fitness = float("-inf")
        fitness_history = []

        ctx = mp.get_context("spawn")

        # Serialize fitness function once for the entire run
        fitness_fn_bytes = _dumps_fitness_fn(self.fitness_fn)

        with ProcessPoolExecutor(
            max_workers=self.n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(fitness_fn_bytes,),
        ) as executor:
            # Evaluate initial populations - batch all islands together
            all_individuals = []
            island_sizes = []
            for island in islands:
                all_individuals.extend(island)
                island_sizes.append(len(island))

            batches = []
            for i in range(0, len(all_individuals), max(1, len(all_individuals) // (self.n_workers * 2))):
                batches.append(all_individuals[i : i + max(1, len(all_individuals) // (self.n_workers * 2))])

            futures = [executor.submit(_evaluate_batch_worker, batch) for batch in batches]
            all_fitness_values = []
            for future in futures:
                all_fitness_values.extend(future.result())

            offset = 0
            for i, size in enumerate(island_sizes):
                island_fitness[i] = all_fitness_values[offset : offset + size]
                offset += size

            # Track best
            for island, fitness in zip(islands, island_fitness):
                for j, f in enumerate(fitness):
                    if f > best_fitness:
                        best_fitness = f
                        best_individual = island[j].copy()
            fitness_history.append(best_fitness)

            # Evolution loop
            for gen in range(self.generations):
                # Track which individuals are new (need evaluation)
                all_new_fitness: list[float | None] = []

                # Evolve each island for one generation
                for island_idx in range(self.num_islands):
                    island = islands[island_idx]
                    fitness = island_fitness[island_idx]

                    # Partial sort: only need top-k elites
                    if self.elitism > 0 and self.elitism < len(fitness):
                        fitness_arr = np.array(fitness)
                        elite_indices = np.argpartition(
                            -fitness_arr, self.elitism
                        )[: self.elitism]
                    else:
                        elite_indices = list(range(self.elitism))

                    # Create new population
                    new_island = []
                    new_island_fitness: list[float | None] = []

                    # Elitism: keep best with their fitness
                    for i in elite_indices:
                        new_island.append(island[i].copy())
                        new_island_fitness.append(fitness[i])

                    # Generate offspring
                    while len(new_island) < self.island_population:
                        # Tournament selection
                        indices1 = rng.choice(len(island), size=self.tournament_size, replace=False)
                        indices2 = rng.choice(len(island), size=self.tournament_size, replace=False)
                        p1_idx = max(indices1, key=lambda i: fitness[i])
                        p2_idx = max(indices2, key=lambda i: fitness[i])

                        parent1 = island[p1_idx]
                        parent2 = island[p2_idx]

                        # Crossover (BLX-alpha)
                        if rng.random() < self.crossover_rate:
                            alpha = 0.5
                            diff = np.abs(parent1 - parent2)
                            lo = np.minimum(parent1, parent2) - alpha * diff
                            hi = np.maximum(parent1, parent2) + alpha * diff
                            child1 = rng.uniform(lo, hi)
                            child2 = rng.uniform(lo, hi)
                            child1 = np.clip(child1, self.lower_bounds, self.upper_bounds)
                            child2 = np.clip(child2, self.lower_bounds, self.upper_bounds)
                        else:
                            child1 = parent1.copy()
                            child2 = parent2.copy()

                        # Mutation
                        for child in [child1, child2]:
                            mask = rng.random(len(child)) < self.mutation_rate
                            if np.any(mask):
                                ub = np.array(self.upper_bounds)
                                lb = np.array(self.lower_bounds)
                                sigma = (ub - lb) * 0.1
                                child[mask] += rng.normal(0, sigma[mask])
                                child[:] = np.clip(child, lb, ub)

                        new_island.append(child1)
                        new_island_fitness.append(None)
                        if len(new_island) < self.island_population:
                            new_island.append(child2)
                            new_island_fitness.append(None)

                    islands[island_idx] = new_island
                    all_new_fitness.extend(new_island_fitness)

                # Collect all individuals for batch evaluation
                all_individuals = []
                for island in islands:
                    all_individuals.extend(island)

                # Only evaluate individuals without fitness (offspring)
                indices_to_eval = [
                    i for i, f in enumerate(all_new_fitness) if f is None
                ]
                if indices_to_eval:
                    genomes_to_eval = [all_individuals[i] for i in indices_to_eval]
                    chunk_size = max(1, len(genomes_to_eval) // (self.n_workers * 2))
                    batches = []
                    for i in range(0, len(genomes_to_eval), chunk_size):
                        batches.append(genomes_to_eval[i : i + chunk_size])

                    futures = [executor.submit(_evaluate_batch_worker, batch) for batch in batches]
                    evaluated = []
                    for future in futures:
                        evaluated.extend(future.result())

                    for idx, fit in zip(indices_to_eval, evaluated):
                        all_new_fitness[idx] = fit

                # Distribute fitness back to islands
                offset = 0
                for i, island in enumerate(islands):
                    size = len(island)
                    island_fitness[i] = all_new_fitness[offset : offset + size]
                    offset += size

                # Migration (ring topology)
                if (gen + 1) % self.migration_interval == 0:
                    migrants = []
                    for i in range(self.num_islands):
                        # Get best individuals to migrate
                        fitness_arr = np.array(island_fitness[i])
                        mc = min(self.migration_count, len(fitness_arr))
                        if mc < len(fitness_arr):
                            best_indices = np.argpartition(-fitness_arr, mc)[: mc]
                        else:
                            best_indices = np.argsort(-fitness_arr)[: mc]
                        migrants.append(
                            [(islands[i][j].copy(), island_fitness[i][j]) for j in best_indices]
                        )

                    # Send migrants to next island
                    for i in range(self.num_islands):
                        dest = (i + 1) % self.num_islands
                        for migrant, migrant_fitness in migrants[i]:
                            # Replace worst individuals, carrying the migrant's
                            # real fitness across (fitness is genome-intrinsic).
                            worst_idx = np.argmin(island_fitness[dest])
                            islands[dest][worst_idx] = migrant
                            island_fitness[dest][worst_idx] = migrant_fitness

                # Track best
                for island, fitness in zip(islands, island_fitness):
                    for j, f in enumerate(fitness):
                        if f > best_fitness:
                            best_fitness = f
                            best_individual = island[j].copy()
                fitness_history.append(best_fitness)

        return ParallelGAResult(
            best_genes=best_individual,
            best_fitness=best_fitness,
            generations=self.generations,
            fitness_history=fitness_history,
        )
