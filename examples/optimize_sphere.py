#!/usr/bin/env python3
"""
Example: Optimizing the Sphere function using ParGA.

The sphere function is a simple convex function with a global minimum at the origin.
This is a good first test for any optimization algorithm.
"""

import time

import numpy as np

from parga import (
    CrossoverMethod,
    GeneticAlgorithm,
    IslandModel,
    MigrationTopology,
    MutationMethod,
    SelectionMethod,
)


def sphere(genes: np.ndarray) -> float:
    """Sphere function: minimize sum of squares."""
    return -np.sum(genes**2)


def main():
    print("=" * 60)
    print("ParGA Example: Sphere Function Optimization")
    print("=" * 60)

    dimensions = 20
    print(f"\nOptimizing {dimensions}-dimensional sphere function")
    print("Global optimum: f(0, 0, ..., 0) = 0")

    # Simple GA
    print("\n--- Simple Genetic Algorithm ---")
    start = time.time()

    ga = GeneticAlgorithm(
        fitness_fn=sphere,
        genome_length=dimensions,
        population_size=100,
        generations=200,
        mutation_rate=0.02,
        crossover_rate=0.8,
        elitism=2,
        lower_bounds=[-5.0] * dimensions,
        upper_bounds=[5.0] * dimensions,
        seed=42,
    )

    result = ga.run()
    elapsed = time.time() - start

    print(f"Best fitness: {result.best_fitness:.6f}")
    print(f"Converged: {result.converged}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Solution (first 5 dims): {result.best_genes()[:5]}")

    # Island Model
    print("\n--- Island Model ---")
    start = time.time()

    island_ga = IslandModel(
        fitness_fn=sphere,
        genome_length=dimensions,
        num_islands=4,
        island_population=50,
        generations=200,
        migration_interval=20,
        migration_count=5,
        topology=MigrationTopology.ring(),
        mutation_rate=0.02,
        crossover_rate=0.8,
        lower_bounds=[-5.0] * dimensions,
        upper_bounds=[5.0] * dimensions,
        seed=42,
    )

    # Customize operators
    island_ga.set_selection(SelectionMethod.tournament(5))
    island_ga.set_crossover(CrossoverMethod.blend(0.5))
    island_ga.set_mutation(MutationMethod.gaussian(0.1))

    result = island_ga.run()
    elapsed = time.time() - start

    print(f"Best fitness: {result.best_fitness:.6f}")
    print(f"Converged: {result.converged}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Island best fitnesses: {result.island_best_fitness()}")
    print(f"Solution (first 5 dims): {result.best_genes()[:5]}")

    # Compare topologies
    print("\n--- Comparing Migration Topologies ---")
    topologies = [
        ("Ring", MigrationTopology.ring()),
        ("Star", MigrationTopology.star()),
        ("Ladder", MigrationTopology.ladder()),
        ("Fully Connected", MigrationTopology.fully_connected()),
    ]

    for name, topology in topologies:
        start = time.time()
        island_ga = IslandModel(
            fitness_fn=sphere,
            genome_length=dimensions,
            num_islands=4,
            island_population=50,
            generations=100,
            migration_interval=10,
            topology=topology,
            seed=42,
        )
        result = island_ga.run()
        elapsed = time.time() - start
        print(f"{name:20s}: fitness = {result.best_fitness:10.6f}, time = {elapsed:.3f}s")


if __name__ == "__main__":
    main()
