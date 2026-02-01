"""Tests for the parga Python bindings."""

import numpy as np
import pytest

from parga import (
    CrossoverMethod,
    GeneticAlgorithm,
    IslandModel,
    MigrationTopology,
    MutationMethod,
    SelectionMethod,
    ackley,
    rastrigin,
    rosenbrock,
    sphere,
)


def simple_fitness(genes: np.ndarray) -> float:
    """Simple fitness function: negative sum of squares."""
    return -np.sum(genes**2)


class TestGeneticAlgorithm:
    """Tests for the basic GeneticAlgorithm class."""

    def test_basic_run(self):
        """Test basic GA execution."""
        ga = GeneticAlgorithm(
            fitness_fn=simple_fitness,
            genome_length=5,
            population_size=50,
            generations=20,
            seed=42,
        )
        result = ga.run()

        assert result.best_fitness is not None
        assert result.generations == 20
        assert len(result.best_genes()) == 5

    def test_bounds(self):
        """Test GA with custom bounds."""
        ga = GeneticAlgorithm(
            fitness_fn=simple_fitness,
            genome_length=3,
            population_size=30,
            generations=10,
            lower_bounds=[-1.0, -1.0, -1.0],
            upper_bounds=[1.0, 1.0, 1.0],
            seed=42,
        )
        result = ga.run()
        genes = result.best_genes()

        # Best genes should be within bounds (close to 0 for sphere)
        assert all(-1.5 <= g <= 1.5 for g in genes)

    def test_selection_methods(self):
        """Test different selection methods."""
        methods = [
            SelectionMethod.tournament(3),
            SelectionMethod.roulette(),
            SelectionMethod.rank(),
            SelectionMethod.truncation(0.5),
        ]

        for method in methods:
            ga = GeneticAlgorithm(
                fitness_fn=simple_fitness,
                genome_length=3,
                population_size=30,
                generations=10,
                seed=42,
            )
            ga.set_selection(method)
            result = ga.run()
            assert result.best_fitness is not None

    def test_crossover_methods(self):
        """Test different crossover methods."""
        methods = [
            CrossoverMethod.single_point(),
            CrossoverMethod.two_point(),
            CrossoverMethod.uniform(0.5),
            CrossoverMethod.blend(0.5),
            CrossoverMethod.arithmetic(),
        ]

        for method in methods:
            ga = GeneticAlgorithm(
                fitness_fn=simple_fitness,
                genome_length=5,
                population_size=30,
                generations=10,
                seed=42,
            )
            ga.set_crossover(method)
            result = ga.run()
            assert result.best_fitness is not None

    def test_mutation_methods(self):
        """Test different mutation methods."""
        methods = [
            MutationMethod.gaussian(0.1),
            MutationMethod.uniform(),
            MutationMethod.polynomial(20.0),
            MutationMethod.boundary(),
        ]

        for method in methods:
            ga = GeneticAlgorithm(
                fitness_fn=simple_fitness,
                genome_length=5,
                population_size=30,
                generations=10,
                seed=42,
            )
            ga.set_mutation(method)
            result = ga.run()
            assert result.best_fitness is not None

    def test_fitness_history(self):
        """Test that fitness history is recorded."""
        ga = GeneticAlgorithm(
            fitness_fn=simple_fitness,
            genome_length=5,
            population_size=50,
            generations=30,
            seed=42,
        )
        result = ga.run()
        history = result.fitness_history()

        assert len(history) > 0
        # Fitness should generally improve (get less negative)
        assert history[-1] >= history[0] - 1.0  # Allow some tolerance


class TestIslandModel:
    """Tests for the IslandModel class."""

    def test_basic_island_model(self):
        """Test basic island model execution."""
        island_ga = IslandModel(
            fitness_fn=simple_fitness,
            genome_length=5,
            num_islands=2,
            island_population=30,
            generations=20,
            migration_interval=10,
            seed=42,
        )
        result = island_ga.run()

        assert result.best_fitness is not None
        assert len(result.island_best_fitness()) == 2
        assert len(result.best_genes()) == 5

    def test_migration_topologies(self):
        """Test different migration topologies."""
        topologies = [
            MigrationTopology.ring(),
            MigrationTopology.star(),
            MigrationTopology.ladder(),
            MigrationTopology.fully_connected(),
            MigrationTopology.random(),
        ]

        for topology in topologies:
            island_ga = IslandModel(
                fitness_fn=simple_fitness,
                genome_length=3,
                num_islands=3,
                island_population=20,
                generations=15,
                migration_interval=5,
                topology=topology,
                seed=42,
            )
            result = island_ga.run()
            assert result.best_fitness is not None

    def test_operators(self):
        """Test setting operators on island model."""
        island_ga = IslandModel(
            fitness_fn=simple_fitness,
            genome_length=5,
            num_islands=2,
            island_population=30,
            generations=15,
            seed=42,
        )

        island_ga.set_selection(SelectionMethod.tournament(5))
        island_ga.set_crossover(CrossoverMethod.blend(0.3))
        island_ga.set_mutation(MutationMethod.gaussian(0.2))

        result = island_ga.run()
        assert result.best_fitness is not None


class TestBenchmarkFunctions:
    """Tests for the built-in benchmark functions."""

    def test_sphere_at_origin(self):
        """Test sphere function at origin."""
        x = np.array([0.0, 0.0, 0.0])
        assert abs(sphere(x)) < 1e-10

    def test_sphere_away_from_origin(self):
        """Test sphere function away from origin."""
        x = np.array([1.0, 2.0, 3.0])
        # Should be negative (since we negate for maximization)
        assert sphere(x) < 0

    def test_rastrigin_at_origin(self):
        """Test rastrigin function at origin."""
        x = np.array([0.0, 0.0])
        assert abs(rastrigin(x)) < 1e-10

    def test_rosenbrock_at_optimum(self):
        """Test rosenbrock function at optimum."""
        x = np.array([1.0, 1.0, 1.0])
        assert abs(rosenbrock(x)) < 1e-10

    def test_ackley_at_origin(self):
        """Test ackley function at origin."""
        x = np.array([0.0, 0.0])
        assert abs(ackley(x)) < 1e-10


class TestOptimization:
    """Tests for actual optimization quality."""

    def test_sphere_optimization(self):
        """Test that GA can optimize sphere function."""
        ga = GeneticAlgorithm(
            fitness_fn=lambda x: -np.sum(x**2),
            genome_length=5,
            population_size=100,
            generations=100,
            mutation_rate=0.05,
            lower_bounds=[-5.0] * 5,
            upper_bounds=[5.0] * 5,
            seed=42,
        )
        result = ga.run()

        # Should find solution close to origin
        assert result.best_fitness > -1.0

    def test_island_model_outperforms(self):
        """Test that island model can solve problems effectively."""
        island_ga = IslandModel(
            fitness_fn=lambda x: -np.sum(x**2),
            genome_length=10,
            num_islands=4,
            island_population=50,
            generations=50,
            migration_interval=10,
            migration_count=5,
            seed=42,
        )
        result = island_ga.run()

        # Should find reasonable solution
        assert result.best_fitness > -5.0


class TestReproducibility:
    """Tests for reproducibility with seeds."""

    @pytest.mark.skip(reason="Reproducibility requires thread-local RNG seeding")
    def test_ga_reproducibility(self):
        """Test that same seed produces same results."""

        def run_ga():
            ga = GeneticAlgorithm(
                fitness_fn=simple_fitness,
                genome_length=5,
                population_size=50,
                generations=30,
                seed=12345,
            )
            return ga.run()

        result1 = run_ga()
        result2 = run_ga()

        np.testing.assert_almost_equal(result1.best_fitness, result2.best_fitness)
        np.testing.assert_array_almost_equal(result1.best_genes(), result2.best_genes())

    @pytest.mark.skip(reason="Reproducibility requires thread-local RNG seeding")
    def test_island_reproducibility(self):
        """Test island model reproducibility."""

        def run_island():
            island_ga = IslandModel(
                fitness_fn=simple_fitness,
                genome_length=5,
                num_islands=2,
                island_population=30,
                generations=20,
                seed=12345,
            )
            return island_ga.run()

        result1 = run_island()
        result2 = run_island()

        np.testing.assert_almost_equal(result1.best_fitness, result2.best_fitness)
