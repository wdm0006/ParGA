"""Travelling salesman problem with PermutationGA.

Ten cities sit on a circle but are labelled in shuffled order, so the optimal
tour is the circle's perimeter (a regular decagon).

    python examples/tsp_permutation.py
"""

import numpy as np

from parga import PermutationCrossover, PermutationGA, PermutationMutation

N_CITIES = 10


def make_cities(n: int = N_CITIES, seed: int = 0) -> np.ndarray:
    angles = 2 * np.pi * np.arange(n) / n
    ring = np.column_stack([np.cos(angles), np.sin(angles)])
    return ring[np.random.default_rng(seed).permutation(n)]


def tour_length(cities: np.ndarray, order: np.ndarray) -> float:
    path = cities[order]
    return float(np.linalg.norm(path - np.roll(path, -1, axis=0), axis=1).sum())


def main() -> None:
    cities = make_cities()

    def fitness(order: np.ndarray) -> float:
        return -tour_length(cities, order)

    result = PermutationGA(
        fitness,
        genome_length=N_CITIES,
        population_size=100,
        generations=100,
        seed=42,
        crossover_method=PermutationCrossover.order(),
        mutation_method=PermutationMutation.inversion(),
    ).run()

    optimal = N_CITIES * 2 * np.sin(np.pi / N_CITIES)
    print(f"Best tour: {result.best_order().tolist()}")
    print(f"Length:    {-result.best_fitness:.6f} (optimal {optimal:.6f})")


if __name__ == "__main__":
    main()
