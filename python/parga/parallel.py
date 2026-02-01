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
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

# Use cloudpickle for better function serialization (handles lambdas, closures)
import cloudpickle
import numpy as np


def _evaluate_batch_worker(args: tuple) -> list[float]:
    """Worker function that evaluates a batch of genomes.

    This runs in a separate process with its own Python interpreter.
    """
    fitness_fn_bytes, genomes = args
    fitness_fn = cloudpickle.loads(fitness_fn_bytes)
    return [fitness_fn(genome) for genome in genomes]


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
        self.fitness_fn = fitness_fn
        self.genome_length = genome_length
        self.population_size = population_size
        self.generations = generations
        self.n_workers = n_workers or mp.cpu_count()
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.lower_bounds = lower_bounds or [-10.0] * genome_length
        self.upper_bounds = upper_bounds or [10.0] * genome_length
        self.seed = seed
        self.chunk_size = chunk_size or max(1, population_size // (self.n_workers * 2))

        # Validate
        if len(self.lower_bounds) != genome_length:
            raise ValueError("lower_bounds length must match genome_length")
        if len(self.upper_bounds) != genome_length:
            raise ValueError("upper_bounds length must match genome_length")

    def _create_random_population(self, rng: np.random.Generator) -> list[np.ndarray]:
        """Create initial random population."""
        lower = np.array(self.lower_bounds)
        upper = np.array(self.upper_bounds)
        return [rng.uniform(lower, upper) for _ in range(self.population_size)]

    def _evaluate_parallel(
        self,
        population: list[np.ndarray],
        executor: ProcessPoolExecutor,
    ) -> list[float]:
        """Evaluate fitness for all individuals in parallel."""
        # Serialize the fitness function once with cloudpickle
        fitness_fn_bytes = cloudpickle.dumps(self.fitness_fn)

        # Create batches
        batches = []
        for i in range(0, len(population), self.chunk_size):
            batch = population[i : i + self.chunk_size]
            batches.append((fitness_fn_bytes, batch))

        # Submit batches to workers
        futures = [executor.submit(_evaluate_batch_worker, batch) for batch in batches]

        # Collect results
        all_fitness = []
        for future in futures:
            all_fitness.extend(future.result())

        return all_fitness

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

        # Use spawn to ensure clean worker processes
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as executor:
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
                # Sort by fitness (descending)
                sorted_indices = np.argsort(fitness)[::-1]
                population = [population[i] for i in sorted_indices]
                fitness = [fitness[i] for i in sorted_indices]

                # Create new population
                new_population = []

                # Elitism: keep best individuals
                for i in range(self.elitism):
                    new_population.append(population[i].copy())

                # Generate offspring
                while len(new_population) < self.population_size:
                    parent1 = self._tournament_select(population, fitness, rng)
                    parent2 = self._tournament_select(population, fitness, rng)

                    child1, child2 = self._crossover(parent1, parent2, rng)

                    child1 = self._mutate(child1, rng)
                    child2 = self._mutate(child2, rng)

                    new_population.append(child1)
                    if len(new_population) < self.population_size:
                        new_population.append(child2)

                population = new_population

                # Evaluate new population
                fitness = self._evaluate_parallel(population, executor)

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
        self.lower_bounds = lower_bounds or [-10.0] * genome_length
        self.upper_bounds = upper_bounds or [10.0] * genome_length
        self.seed = seed

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

        # Serialize fitness function once
        fitness_fn_bytes = cloudpickle.dumps(self.fitness_fn)

        with ProcessPoolExecutor(max_workers=self.n_workers, mp_context=ctx) as executor:
            # Evaluate initial populations
            for i, island in enumerate(islands):
                batch = (fitness_fn_bytes, island)
                future = executor.submit(_evaluate_batch_worker, batch)
                island_fitness[i] = future.result()

            # Track best
            for island, fitness in zip(islands, island_fitness):
                for j, f in enumerate(fitness):
                    if f > best_fitness:
                        best_fitness = f
                        best_individual = island[j].copy()
            fitness_history.append(best_fitness)

            # Evolution loop
            for gen in range(self.generations):
                # Evolve each island for one generation
                for island_idx in range(self.num_islands):
                    island = islands[island_idx]
                    fitness = island_fitness[island_idx]

                    # Sort by fitness
                    sorted_indices = np.argsort(fitness)[::-1]
                    island = [island[i] for i in sorted_indices]
                    fitness = [fitness[i] for i in sorted_indices]

                    # Create new population
                    new_island = []

                    # Elitism
                    for i in range(self.elitism):
                        new_island.append(island[i].copy())

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
                                upper = np.array(self.upper_bounds)
                                lower = np.array(self.lower_bounds)
                                sigma = (upper - lower) * 0.1
                                child[mask] += rng.normal(0, sigma[mask])
                                child[:] = np.clip(child, lower, upper)

                        new_island.append(child1)
                        if len(new_island) < self.island_population:
                            new_island.append(child2)

                    islands[island_idx] = new_island

                # Evaluate all islands in parallel
                all_individuals = []
                island_sizes = []
                for island in islands:
                    all_individuals.extend(island)
                    island_sizes.append(len(island))

                # Batch evaluate
                batch = (fitness_fn_bytes, all_individuals)
                future = executor.submit(_evaluate_batch_worker, batch)
                all_fitness = future.result()

                # Distribute fitness back to islands
                offset = 0
                for i, size in enumerate(island_sizes):
                    island_fitness[i] = all_fitness[offset : offset + size]
                    offset += size

                # Migration (ring topology)
                if (gen + 1) % self.migration_interval == 0:
                    migrants = []
                    for i in range(self.num_islands):
                        # Get best individuals to migrate
                        sorted_indices = np.argsort(island_fitness[i])[::-1]
                        best_indices = sorted_indices[: self.migration_count]
                        migrants.append([islands[i][j].copy() for j in best_indices])

                    # Send migrants to next island
                    for i in range(self.num_islands):
                        dest = (i + 1) % self.num_islands
                        for migrant in migrants[i]:
                            # Replace worst individuals
                            worst_idx = np.argmin(island_fitness[dest])
                            islands[dest][worst_idx] = migrant
                            island_fitness[dest][worst_idx] = 0.0  # Will be re-evaluated

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
