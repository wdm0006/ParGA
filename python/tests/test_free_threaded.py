"""Tests for free-threaded Python (3.13t+) rayon parallelism.

These tests are skipped on GIL Python since rayon parallelism
requires free-threaded builds.
"""

import numpy as np
import pytest

from parga import GA, GeneticAlgorithm, IslandModel, is_free_threaded

pytestmark = pytest.mark.skipif(
    not is_free_threaded(),
    reason="Requires free-threaded Python (3.13t+)",
)


def simple_fitness(genes: np.ndarray) -> float:
    """Simple fitness function: negative sum of squares."""
    return -np.sum(genes**2)


class TestFreeThreadedGA:
    """Tests for GA running with rayon parallelism on free-threaded Python."""

    def test_basic_optimization(self):
        """Rust GA with rayon should optimize sphere correctly."""
        ga = GeneticAlgorithm(
            fitness_fn=simple_fitness,
            genome_length=5,
            population_size=100,
            generations=50,
            lower_bounds=[-5.0] * 5,
            upper_bounds=[5.0] * 5,
            seed=42,
        )
        result = ga.run()
        assert result.best_fitness > -1.0

    def test_ga_auto_selects_rust(self):
        """GA class should select 'rust' strategy on free-threaded Python."""
        ga = GA(
            simple_fitness,
            genome_length=5,
            population_size=50,
            generations=10,
            verbose=False,
        )
        result = ga.run()
        assert ga._strategy == "rust"
        assert result.best_fitness is not None

    def test_ga_island_auto_selects_rust_island(self):
        """GA class with islands should select 'rust_island' on free-threaded."""
        ga = GA(
            simple_fitness,
            genome_length=5,
            population_size=100,
            generations=10,
            islands=4,
            verbose=False,
        )
        result = ga.run()
        assert ga._strategy == "rust_island"
        assert result.best_fitness is not None


class TestFreeThreadedIslandModel:
    """Tests for island model with rayon on free-threaded Python."""

    def test_island_model_basic(self):
        """Island model should work with rayon parallelism."""
        island_ga = IslandModel(
            fitness_fn=simple_fitness,
            genome_length=5,
            num_islands=4,
            island_population=30,
            generations=20,
            migration_interval=10,
            seed=42,
        )
        result = island_ga.run()
        assert result.best_fitness is not None
        assert len(result.island_best_fitness()) == 4


class TestConcurrentCallbacks:
    """Tests verifying concurrent Python callback execution."""

    def test_concurrent_evaluation(self):
        """Verify that fitness evaluations produce correct results under concurrency."""
        import threading

        thread_ids = set()
        lock = threading.Lock()

        def tracking_fitness(genes: np.ndarray) -> float:
            tid = threading.get_ident()
            with lock:
                thread_ids.add(tid)
            # Do enough work to overlap with other threads
            return -np.sum(genes**2)

        ga = GeneticAlgorithm(
            fitness_fn=tracking_fitness,
            genome_length=10,
            population_size=200,
            generations=5,
            lower_bounds=[-5.0] * 10,
            upper_bounds=[5.0] * 10,
            seed=42,
        )
        result = ga.run()

        # Should have used multiple threads
        assert len(thread_ids) > 1, (
            f"Expected multiple threads, got {len(thread_ids)}"
        )
        # Results should still be correct
        assert result.best_fitness > -50.0
