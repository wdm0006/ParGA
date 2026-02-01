#!/usr/bin/env python3
"""
Example: Optimizing the Rastrigin function using ParGA.

The Rastrigin function is a highly multimodal function with many local minima,
making it a challenging test for genetic algorithms.
"""

import time

import numpy as np

from parga import (
    IslandModel,
    MigrationTopology,
    rastrigin,
)


def main():
    print("=" * 60)
    print("ParGA Example: Rastrigin Function Optimization")
    print("=" * 60)

    dimensions = 10
    print(f"\nOptimizing {dimensions}-dimensional Rastrigin function")
    print("Global optimum: f(0, 0, ..., 0) = 0")
    print("This is a highly multimodal function with many local minima.\n")

    # Island model is better for multimodal functions
    print("--- Island Model with Ring Migration ---")
    start = time.time()

    island_ga = IslandModel(
        fitness_fn=rastrigin,  # Using built-in Rastrigin
        genome_length=dimensions,
        num_islands=8,
        island_population=100,
        generations=300,
        migration_interval=25,
        migration_count=10,
        topology=MigrationTopology.ring(),
        mutation_rate=0.05,
        crossover_rate=0.9,
        elitism=2,
        tournament_size=5,
        lower_bounds=[-5.12] * dimensions,
        upper_bounds=[5.12] * dimensions,
        seed=42,
    )

    result = island_ga.run()
    elapsed = time.time() - start

    print(f"Best fitness: {result.best_fitness:.6f}")
    print("(Optimal is 0.0, closer to 0 is better)")
    print(f"Time: {elapsed:.3f}s")
    print(f"Best solution: {result.best_genes()}")

    # The solution should be close to zero for all dimensions
    solution = result.best_genes()
    max_deviation = np.max(np.abs(solution))
    print(f"\nMax deviation from origin: {max_deviation:.4f}")

    # Try with more islands
    print("\n--- Scaling with number of islands ---")
    for num_islands in [2, 4, 8, 16]:
        start = time.time()
        island_ga = IslandModel(
            fitness_fn=rastrigin,
            genome_length=dimensions,
            num_islands=num_islands,
            island_population=50,
            generations=200,
            migration_interval=20,
            topology=MigrationTopology.ring(),
            lower_bounds=[-5.12] * dimensions,
            upper_bounds=[5.12] * dimensions,
            seed=42,
        )
        result = island_ga.run()
        elapsed = time.time() - start
        print(
            f"{num_islands:2d} islands: fitness = {result.best_fitness:10.4f}, "
            f"time = {elapsed:.3f}s"
        )


if __name__ == "__main__":
    main()
