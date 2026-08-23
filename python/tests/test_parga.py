"""Tests for the parga Python bindings."""

import warnings

import numpy as np
import pytest

from parga import (
    GA,
    CrossoverMethod,
    GeneticAlgorithm,
    IslandModel,
    MigrationTopology,
    MutationMethod,
    ParallelGA,
    ParallelIslandModel,
    SelectionMethod,
    ackley,
    griewank,
    is_free_threaded,
    maximize,
    minimize,
    rastrigin,
    rosenbrock,
    schwefel,
    set_num_threads,
    sphere,
)


def simple_fitness(genes: np.ndarray) -> float:
    """Simple fitness function: negative sum of squares."""
    return -np.sum(genes**2)


def strictly_negative_fitness(genes: np.ndarray) -> float:
    """Fitness that is always strictly negative (bounded above by -1.0)."""
    return -(np.sum(genes**2) + 1.0)


def sphere_objective(genes: np.ndarray) -> float:
    """Sphere function to MINIMIZE: sum of squares, optimum 0 at the origin."""
    return float(np.sum(genes**2))


def neg_sphere_objective(genes: np.ndarray) -> float:
    """Negated sphere to MAXIMIZE: optimum 0 at the origin."""
    return float(-np.sum(genes**2))


def nan_fitness(genes: np.ndarray) -> float:
    """Fitness callback that returns NaN."""
    return float("nan")


def inf_fitness(genes: np.ndarray) -> float:
    """Fitness callback that returns positive infinity."""
    return float("inf")


def neg_inf_fitness(genes: np.ndarray) -> float:
    """Fitness callback that returns negative infinity."""
    return float("-inf")


def plateau_fitness(genes: np.ndarray) -> float:
    """Constant objective, so the best fitness can never improve after gen 0."""
    return -1.0


# Module-level so the process-pool engines can serialize them to workers.
NON_FINITE_FITNESS_FNS = [nan_fitness, inf_fitness, neg_inf_fitness]


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

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"population_size": 0}, "population size"),
            ({"genome_length": 0}, "genome_length"),
            ({"population_size": 2, "elitism": 3}, "elitism"),
            ({"population_size": 2, "tournament_size": 3}, "tournament_size"),
            ({"mutation_rate": 1.1}, "mutation_rate"),
            ({"lower_bounds": [0.0], "upper_bounds": [1.0, 1.0]}, "upper_bounds"),
            ({"lower_bounds": [2.0], "upper_bounds": [1.0]}, "lower bound"),
        ],
    )
    def test_invalid_config_is_value_error(self, kwargs, message):
        base = {"fitness_fn": simple_fitness, "genome_length": 1}
        base.update(kwargs)
        with pytest.raises(ValueError, match=message):
            GeneticAlgorithm(**base)

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

    @pytest.mark.parametrize(
        "factory, parameter",
        [
            (lambda: SelectionMethod.truncation(0.0), "ratio"),
            (lambda: SelectionMethod.truncation(1.1), "ratio"),
            (lambda: SelectionMethod.truncation(np.nan), "ratio"),
            (lambda: SelectionMethod.truncation(np.inf), "ratio"),
            (lambda: SelectionMethod.truncation(-np.inf), "ratio"),
            (lambda: CrossoverMethod.uniform(-0.1), "probability"),
            (lambda: CrossoverMethod.uniform(1.1), "probability"),
            (lambda: CrossoverMethod.uniform(np.nan), "probability"),
            (lambda: CrossoverMethod.uniform(np.inf), "probability"),
            (lambda: CrossoverMethod.uniform(-np.inf), "probability"),
            (lambda: CrossoverMethod.blend(-0.1), "alpha"),
            (lambda: CrossoverMethod.blend(np.nan), "alpha"),
            (lambda: CrossoverMethod.blend(np.inf), "alpha"),
            (lambda: CrossoverMethod.blend(-np.inf), "alpha"),
            (lambda: CrossoverMethod.simulated_binary(-0.1), "eta"),
            (lambda: CrossoverMethod.simulated_binary(np.nan), "eta"),
            (lambda: CrossoverMethod.simulated_binary(np.inf), "eta"),
            (lambda: CrossoverMethod.simulated_binary(-np.inf), "eta"),
            (lambda: MutationMethod.gaussian(0.0), "sigma"),
            (lambda: MutationMethod.gaussian(-0.1), "sigma"),
            (lambda: MutationMethod.gaussian(np.nan), "sigma"),
            (lambda: MutationMethod.gaussian(np.inf), "sigma"),
            (lambda: MutationMethod.gaussian(-np.inf), "sigma"),
            (lambda: MutationMethod.polynomial(-0.1), "eta"),
            (lambda: MutationMethod.polynomial(np.nan), "eta"),
            (lambda: MutationMethod.polynomial(np.inf), "eta"),
            (lambda: MutationMethod.polynomial(-np.inf), "eta"),
        ],
    )
    def test_parameterized_operator_methods_reject_invalid_values(
        self, factory, parameter
    ):
        """Parameterized operator methods reject invalid domains."""
        with pytest.raises(ValueError, match=parameter):
            factory()

    @pytest.mark.parametrize(
        "factory",
        [
            SelectionMethod.truncation,
            CrossoverMethod.uniform,
            CrossoverMethod.blend,
            CrossoverMethod.simulated_binary,
            MutationMethod.gaussian,
            MutationMethod.polynomial,
            lambda: SelectionMethod.truncation(1.0),
            lambda: CrossoverMethod.uniform(0.0),
            lambda: CrossoverMethod.uniform(1.0),
            lambda: CrossoverMethod.blend(0.0),
            lambda: CrossoverMethod.simulated_binary(0.0),
            lambda: MutationMethod.polynomial(0.0),
        ],
    )
    def test_parameterized_operator_methods_accept_valid_boundaries(self, factory):
        """Parameterized operator methods accept defaults and valid boundaries."""
        assert factory() is not None

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

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"migration_interval": 0}, "migration_interval"),
            (
                {"num_islands": 1, "topology": MigrationTopology.random()},
                "num_islands",
            ),
            (
                {
                    "island_population": 2,
                    "tournament_size": 2,
                    "migration_count": 3,
                },
                "migration_count",
            ),
            ({"island_population": 2, "elitism": 3}, "elitism"),
        ],
    )
    def test_invalid_config_is_value_error(self, kwargs, message):
        base = {
            "fitness_fn": simple_fitness,
            "genome_length": 1,
            "num_islands": 2,
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=message):
            IslandModel(**base)

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

    def test_parallel_island_migration_preserves_fitness(self):
        """Migration must carry each migrant's real fitness, not a 0.0 placeholder.

        ParGA maximizes and this objective is strictly negative, so a spurious
        0.0 placeholder would beat every real individual and corrupt both the
        reported best_fitness and the next generation's selection/elitism.
        Regression test for the pre-fix behavior where migrants were assigned
        island_fitness = 0.0.
        """
        with pytest.warns(DeprecationWarning):
            model = ParallelIslandModel(
                fitness_fn=strictly_negative_fitness,
                genome_length=4,
                num_islands=3,
                island_population=20,
                generations=12,
                migration_interval=5,
                migration_count=3,
                n_workers=2,
                seed=42,
                lower_bounds=[-5.0] * 4,
                upper_bounds=[5.0] * 4,
            )
        result = model.run()

        # The true optimum of this objective is strictly negative, so the
        # reported best must never be the fabricated 0.0 placeholder.
        assert result.best_fitness is not None
        assert result.best_fitness < 0.0
        # The reported genome must actually evaluate to the reported fitness.
        assert result.best_fitness == pytest.approx(
            strictly_negative_fitness(np.asarray(result.best_genes()))
        )

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

    @pytest.mark.parametrize(
        "benchmark_fn", [sphere, rastrigin, rosenbrock, ackley, griewank, schwefel]
    )
    def test_strided_view_matches_contiguous_copy(self, benchmark_fn):
        """Benchmark functions accept strided arrays without changing results."""
        strided = np.arange(8.0)[::2]
        assert not strided.flags.c_contiguous

        assert benchmark_fn(strided) == pytest.approx(benchmark_fn(strided.copy()))


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
        np.testing.assert_array_almost_equal(result1.best_genes(), result2.best_genes())
        np.testing.assert_array_almost_equal(
            result1.fitness_history(), result2.fitness_history()
        )


class TestFitnessCallbackErrors:
    """Tests that buggy fitness callbacks surface errors instead of being swallowed."""

    def test_ga_raising_callback(self):
        """A callback that raises makes GeneticAlgorithm.run() raise."""

        def raising_fitness(genes: np.ndarray) -> float:
            raise ValueError("boom from fitness")

        ga = GeneticAlgorithm(
            fitness_fn=raising_fitness,
            genome_length=3,
            population_size=20,
            generations=5,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="boom from fitness"):
            ga.run()

    def test_ga_non_float_callback(self):
        """A callback returning a non-float makes GeneticAlgorithm.run() raise."""

        def non_float_fitness(genes: np.ndarray):
            return None

        ga = GeneticAlgorithm(
            fitness_fn=non_float_fitness,
            genome_length=3,
            population_size=20,
            generations=5,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="did not return a float"):
            ga.run()

    def test_ga_string_return_callback(self):
        """A callback returning a string makes GeneticAlgorithm.run() raise."""

        def string_fitness(genes: np.ndarray):
            return "not a number"

        ga = GeneticAlgorithm(
            fitness_fn=string_fitness,
            genome_length=3,
            population_size=20,
            generations=5,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="did not return a float"):
            ga.run()

    def test_island_raising_callback(self):
        """A callback that raises makes IslandModel.run() raise."""

        def raising_fitness(genes: np.ndarray) -> float:
            raise KeyError("missing key in fitness")

        island_ga = IslandModel(
            fitness_fn=raising_fitness,
            genome_length=3,
            num_islands=2,
            island_population=20,
            generations=5,
            migration_interval=5,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="missing key in fitness"):
            island_ga.run()

    def test_island_non_float_callback(self):
        """A callback returning a non-float makes IslandModel.run() raise."""

        def non_float_fitness(genes: np.ndarray):
            return None

        island_ga = IslandModel(
            fitness_fn=non_float_fitness,
            genome_length=3,
            num_islands=2,
            island_population=20,
            generations=5,
            migration_interval=5,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="did not return a float"):
            island_ga.run()

    @pytest.mark.parametrize("fitness_fn", NON_FINITE_FITNESS_FNS)
    def test_ga_non_finite_callback(self, fitness_fn):
        """NaN/±inf from a callback makes GeneticAlgorithm.run() raise."""
        ga = GeneticAlgorithm(
            fitness_fn=fitness_fn,
            genome_length=3,
            population_size=20,
            generations=5,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="non-finite value"):
            ga.run()

    @pytest.mark.parametrize("fitness_fn", NON_FINITE_FITNESS_FNS)
    def test_island_non_finite_callback(self, fitness_fn):
        """NaN/±inf from a callback makes IslandModel.run() raise."""
        island_ga = IslandModel(
            fitness_fn=fitness_fn,
            genome_length=3,
            num_islands=2,
            island_population=20,
            generations=5,
            migration_interval=5,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="non-finite value"):
            island_ga.run()

    @pytest.mark.parametrize("fitness_fn", NON_FINITE_FITNESS_FNS)
    def test_facade_non_finite_callback(self, fitness_fn):
        """The GA facade surfaces a non-finite callback result on the Rust path."""
        ga = GA(
            fitness_fn=fitness_fn,
            genome_length=3,
            population_size=20,
            generations=5,
            parallel=False,
            seed=42,
        )
        with pytest.raises(RuntimeError, match="non-finite value"):
            ga.run()

    @pytest.mark.parametrize("fitness_fn", NON_FINITE_FITNESS_FNS)
    def test_parallel_ga_non_finite_callback(self, fitness_fn):
        """NaN/±inf from a callback makes ParallelGA.run() raise."""
        with pytest.warns(DeprecationWarning):
            ga = ParallelGA(
                fitness_fn=fitness_fn,
                genome_length=3,
                population_size=8,
                generations=2,
                n_workers=1,
                seed=42,
            )
        with pytest.raises(RuntimeError, match="non-finite value"):
            ga.run()

    @pytest.mark.parametrize("fitness_fn", NON_FINITE_FITNESS_FNS)
    def test_parallel_island_non_finite_callback(self, fitness_fn):
        """NaN/±inf from a callback makes ParallelIslandModel.run() raise."""
        with pytest.warns(DeprecationWarning):
            model = ParallelIslandModel(
                fitness_fn=fitness_fn,
                genome_length=3,
                num_islands=2,
                island_population=8,
                generations=2,
                migration_interval=1,
                migration_count=1,
                n_workers=1,
                seed=42,
            )
        with pytest.raises(RuntimeError, match="non-finite value"):
            model.run()

    def test_correct_callback_unaffected(self):
        """A correct float-returning callback still succeeds."""
        ga = GeneticAlgorithm(
            fitness_fn=simple_fitness,
            genome_length=3,
            population_size=20,
            generations=5,
            seed=42,
        )
        result = ga.run()
        assert result.best_fitness is not None

    def test_correct_callback_unaffected_parallel(self):
        """A correct float-returning callback still succeeds on the process pool."""
        with pytest.warns(DeprecationWarning):
            ga = ParallelGA(
                fitness_fn=simple_fitness,
                genome_length=3,
                population_size=8,
                generations=2,
                n_workers=1,
                seed=42,
            )
        result = ga.run()
        assert result.best_fitness is not None
        assert np.isfinite(result.best_fitness)


class TestFreeThreadedUtils:
    """Tests for free-threaded Python utility functions."""

    def test_is_free_threaded_returns_bool(self):
        """is_free_threaded() returns a boolean."""
        result = is_free_threaded()
        assert isinstance(result, bool)

    def test_set_num_threads_invalid(self):
        """set_num_threads(0) raises ValueError."""
        with pytest.raises(ValueError):
            set_num_threads(0)


class TestFacade:
    """Tests for the high-level GA / minimize / maximize facade.

    These exercise the deterministic ``parallel=False`` Rust paths so they
    never depend on fitness-timing measurement and never flake.
    """

    @pytest.mark.parametrize("parallel", [False, True])
    def test_invalid_serial_config_is_backend_independent(self, parallel):
        with pytest.raises(ValueError, match="elitism"):
            GA(
                simple_fitness,
                genome_length=2,
                population_size=2,
                elitism=3,
                parallel=parallel,
            )

    @pytest.mark.parametrize("parallel", [False, True])
    def test_invalid_island_config_is_backend_independent(self, parallel):
        with pytest.raises(ValueError, match="migration_interval"):
            GA(
                simple_fitness,
                genome_length=2,
                population_size=8,
                islands=2,
                migration_interval=0,
                parallel=parallel,
            )

    def test_scalar_bounds_parse_and_run(self):
        """Scalar (lower, upper) bounds parse and produce genes within range."""
        result = GA(
            simple_fitness,
            genome_length=4,
            population_size=30,
            generations=20,
            bounds=(-5, 5),
            parallel=False,
            seed=42,
        ).run()
        genes = result.best_genes()
        assert len(genes) == 4
        assert np.all(genes >= -5.0)
        assert np.all(genes <= 5.0)

    def test_per_gene_bounds_parse_and_run(self):
        """Per-gene ([lowers], [uppers]) bounds parse and are respected."""
        lowers = [-1.0, -2.0, -3.0]
        uppers = [1.0, 2.0, 3.0]
        result = GA(
            simple_fitness,
            genome_length=3,
            population_size=30,
            generations=20,
            bounds=(lowers, uppers),
            parallel=False,
            seed=42,
        ).run()
        genes = result.best_genes()
        assert len(genes) == 3
        assert np.all(genes >= np.array(lowers))
        assert np.all(genes <= np.array(uppers))

    def test_minimize_converges_near_zero(self):
        """minimize(sphere) converges near the origin; -best_fitness ~ true min."""
        result = minimize(
            sphere_objective,
            genome_length=3,
            bounds=(-5, 5),
            population_size=60,
            generations=80,
            parallel=False,
            seed=7,
        )
        # minimize negates internally, so best_fitness is the negated objective;
        # the true minimum value is -result.best_fitness and should be ~0.
        assert -result.best_fitness == pytest.approx(0.0, abs=0.1)
        assert np.allclose(result.best_genes(), 0.0, atol=0.5)

    def test_maximize_converges_near_zero(self):
        """maximize(neg_sphere) mirrors minimize: best_fitness ~ 0 at the origin."""
        result = maximize(
            neg_sphere_objective,
            genome_length=3,
            bounds=(-5, 5),
            population_size=60,
            generations=80,
            parallel=False,
            seed=7,
        )
        assert result.best_fitness == pytest.approx(0.0, abs=0.1)
        assert np.allclose(result.best_genes(), 0.0, atol=0.5)

    def test_strategy_is_rust(self):
        """parallel=False single-population selects the deterministic rust path."""
        result = GA(
            simple_fitness,
            genome_length=4,
            population_size=30,
            generations=10,
            parallel=False,
            seed=42,
        ).run()
        assert result.strategy == "rust"

    def test_strategy_is_rust_island(self):
        """parallel=False with islands>1 selects the rust_island path."""
        result = GA(
            simple_fitness,
            genome_length=4,
            population_size=40,
            generations=10,
            islands=2,
            parallel=False,
            seed=42,
        ).run()
        assert result.strategy == "rust_island"

    def test_reproducibility_rust_path(self):
        """Two rust-path runs with the same seed give bit-identical results."""

        def run():
            return GA(
                simple_fitness,
                genome_length=5,
                population_size=40,
                generations=25,
                parallel=False,
                seed=98765,
            ).run()

        r1 = run()
        r2 = run()
        assert r1.best_fitness == r2.best_fitness
        np.testing.assert_array_equal(r1.best_genes(), r2.best_genes())

    @pytest.mark.skipif(
        is_free_threaded(),
        reason="free-threaded Python always selects a Rust path, which honors the options",
    )
    @pytest.mark.parametrize("islands,strategy", [(1, "parallel"), (2, "parallel_island")])
    def test_parallel_path_warns_about_ignored_options(self, islands, strategy):
        """The process-pool paths warn, naming each Rust-only option they drop."""
        ga = GA(
            simple_fitness,
            genome_length=2,
            population_size=8,
            generations=2,
            islands=islands,
            migration_interval=1,
            migration_count=1,
            n_workers=1,
            parallel=True,
            seed=42,
            selection_method=SelectionMethod.rank(),
            restart_on_stagnation=5,
            early_stopping=5,
            mutation_rate_end=0.0,
        )
        with pytest.warns(UserWarning) as record:
            result = ga.run()

        assert result.strategy == strategy
        messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
        assert len(messages) == 1
        message = messages[0]
        assert "selection_method" in message
        assert "restart_on_stagnation" in message
        assert strategy in message
        # Options that ARE forwarded must not be named as ignored.
        assert "mutation_rate" not in message
        assert "early_stopping" not in message
        # Unset advanced options must not be named either.
        assert "local_search_iters" not in message

    @pytest.mark.skipif(
        is_free_threaded(),
        reason="free-threaded Python always selects a Rust path, which honors the options",
    )
    def test_parallel_path_without_advanced_options_does_not_warn(self):
        """No UserWarning when the parallel path has nothing to ignore."""
        ga = GA(
            simple_fitness,
            genome_length=2,
            population_size=8,
            generations=2,
            n_workers=1,
            parallel=True,
            seed=42,
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = ga.run()

        assert result.strategy == "parallel"
        assert [w for w in record if issubclass(w.category, UserWarning)] == []

    @pytest.mark.parametrize("islands,strategy", [(1, "rust"), (2, "rust_island")])
    def test_rust_path_does_not_warn_about_options(self, islands, strategy):
        """Both Rust paths apply the operator overrides, so they must not warn."""
        ga = GA(
            simple_fitness,
            genome_length=2,
            population_size=8,
            generations=2,
            islands=islands,
            migration_interval=1,
            migration_count=1,
            parallel=False,
            seed=42,
            selection_method=SelectionMethod.rank(),
            early_stopping=5,
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = ga.run()

        assert result.strategy == strategy
        assert [w for w in record if issubclass(w.category, UserWarning)] == []

    def test_rust_island_does_not_warn_about_operator_overrides(self):
        """`rust_island` applies all three operator overrides, so it must not warn."""
        ga = GA(
            simple_fitness,
            genome_length=2,
            population_size=8,
            generations=2,
            islands=2,
            migration_interval=1,
            migration_count=1,
            parallel=False,
            seed=42,
            crossover_method=CrossoverMethod.arithmetic(),
            mutation_method=MutationMethod.uniform(),
            selection_method=SelectionMethod.rank(),
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = ga.run()

        assert result.strategy == "rust_island"
        assert [w for w in record if issubclass(w.category, UserWarning)] == []

    def test_rust_island_warns_about_single_population_options(self):
        """`rust_island` cannot apply the single-population settings, so it warns."""
        ga = GA(
            simple_fitness,
            genome_length=2,
            population_size=8,
            generations=2,
            islands=2,
            migration_interval=1,
            migration_count=1,
            parallel=False,
            seed=42,
            selection_method=SelectionMethod.rank(),
            restart_on_stagnation=3,
            local_search_iters=2,
            random_immigrants=1,
        )
        with pytest.warns(UserWarning) as record:
            result = ga.run()

        assert result.strategy == "rust_island"
        messages = [str(w.message) for w in record if issubclass(w.category, UserWarning)]
        assert len(messages) == 1
        message = messages[0]
        assert "rust_island" in message
        for name in ("restart_on_stagnation", "local_search_iters", "random_immigrants"):
            assert name in message
        # The operator overrides ARE applied on this path, so they must not be named.
        assert "selection_method" not in message

    @pytest.mark.parametrize(
        "option,value",
        [
            ("restart_on_stagnation", 3),
            ("local_search_iters", 2),
            ("random_immigrants", 1),
        ],
    )
    def test_rust_single_population_applies_advanced_options(self, option, value):
        """The `rust` strategy applies each advanced setting, so it must not warn."""
        ga = GA(
            simple_fitness,
            genome_length=2,
            population_size=8,
            generations=2,
            parallel=False,
            seed=42,
            **{option: value},
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = ga.run()

        assert result.strategy == "rust"
        assert [w for w in record if issubclass(w.category, UserWarning)] == []

    def test_best_genes_returns_copy(self):
        """GAResult.best_genes() returns a copy; mutating it is harmless."""
        result = GA(
            simple_fitness,
            genome_length=4,
            population_size=30,
            generations=10,
            parallel=False,
            seed=42,
        ).run()
        genes = result.best_genes()
        original = genes.copy()
        genes[:] = 999.0
        # A fresh call must still return the untouched best genome.
        np.testing.assert_array_equal(result.best_genes(), original)


def _parallel_ga(**kwargs) -> ParallelGA:
    """Build a small ParallelGA, swallowing the deprecation warning."""
    settings = {
        "fitness_fn": plateau_fitness,
        "genome_length": 2,
        "population_size": 8,
        "generations": 6,
        "n_workers": 1,
        "seed": 7,
    }
    settings.update(kwargs)
    with pytest.warns(DeprecationWarning):
        return ParallelGA(**settings)


def _parallel_island(**kwargs) -> ParallelIslandModel:
    """Build a small ParallelIslandModel, swallowing the deprecation warning."""
    settings = {
        "fitness_fn": plateau_fitness,
        "genome_length": 2,
        "num_islands": 2,
        "island_population": 8,
        "generations": 6,
        "migration_interval": 1,
        "migration_count": 1,
        "n_workers": 1,
        "seed": 7,
    }
    settings.update(kwargs)
    with pytest.warns(DeprecationWarning):
        return ParallelIslandModel(**settings)


class TestParallelEngineOptions:
    """early_stopping and mutation_rate_end on the process-pool engines."""

    def test_ga_early_stopping_stops_early(self):
        """A plateaued objective stops before the configured generation count."""
        result = _parallel_ga(early_stopping=1).run()

        assert result.generations < 6
        # gen 0 improves on -inf, gen 1 stagnates and trips patience=1.
        assert result.generations == 2
        # One history entry for the initial evaluation plus one per generation.
        assert len(result.fitness_history) == result.generations + 1

    def test_ga_without_early_stopping_runs_every_generation(self):
        """Default (None) keeps today's behavior: no early exit."""
        result = _parallel_ga().run()

        assert result.generations == 6
        assert len(result.fitness_history) == 7

    def test_island_early_stopping_stops_early(self):
        """The island engine honors early_stopping equivalently."""
        result = _parallel_island(early_stopping=1).run()

        assert result.generations == 2
        assert len(result.fitness_history) == result.generations + 1

    def test_island_without_early_stopping_runs_every_generation(self):
        """Default (None) keeps today's behavior on the island engine too."""
        result = _parallel_island().run()

        assert result.generations == 6
        assert len(result.fitness_history) == 7

    def test_ga_mutation_rate_end_changes_the_search(self):
        """Decaying the mutation rate to 0 reaches a different best fitness."""
        settings = {
            "fitness_fn": neg_sphere_objective,
            "genome_length": 4,
            "generations": 5,
            "seed": 1234,
            "mutation_rate": 0.5,
            "lower_bounds": [-5.0] * 4,
            "upper_bounds": [5.0] * 4,
        }
        baseline = _parallel_ga(**settings).run()
        decayed = _parallel_ga(mutation_rate_end=0.0, **settings).run()
        # A final rate equal to the starting rate is the identity case: it must
        # reproduce the baseline exactly, pinning "no behavior change by default".
        flat = _parallel_ga(mutation_rate_end=0.5, **settings).run()

        assert baseline.best_fitness != decayed.best_fitness
        assert flat.best_fitness == baseline.best_fitness
        np.testing.assert_array_equal(flat.best_genes(), baseline.best_genes())

    def test_island_mutation_rate_end_changes_the_search(self):
        """The island engine applies the decayed rate too."""
        settings = {
            "fitness_fn": neg_sphere_objective,
            "genome_length": 4,
            "generations": 5,
            "migration_interval": 2,
            "seed": 99,
            "mutation_rate": 0.5,
            "lower_bounds": [-5.0] * 4,
            "upper_bounds": [5.0] * 4,
        }
        baseline = _parallel_island(**settings).run()
        decayed = _parallel_island(mutation_rate_end=0.0, **settings).run()
        flat = _parallel_island(mutation_rate_end=0.5, **settings).run()

        assert baseline.best_fitness != decayed.best_fitness
        assert flat.best_fitness == baseline.best_fitness
        np.testing.assert_array_equal(flat.best_genes(), baseline.best_genes())

    @pytest.mark.parametrize("bad_rate", [-0.1, 1.5, float("nan")])
    def test_ga_rejects_invalid_mutation_rate_end(self, bad_rate):
        """mutation_rate_end is validated like every other rate."""
        with pytest.raises(ValueError, match="mutation_rate_end"):
            _parallel_ga(mutation_rate_end=bad_rate)

    @pytest.mark.parametrize("bad_rate", [-0.1, 1.5, float("nan")])
    def test_island_rejects_invalid_mutation_rate_end(self, bad_rate):
        """The island engine validates mutation_rate_end too."""
        with pytest.raises(ValueError, match="mutation_rate_end"):
            _parallel_island(mutation_rate_end=bad_rate)

    @pytest.mark.skipif(
        is_free_threaded(),
        reason="free-threaded Python always selects a Rust path",
    )
    @pytest.mark.parametrize(
        "islands,strategy", [(1, "parallel"), (2, "parallel_island")]
    )
    def test_facade_forwards_early_stopping(self, islands, strategy):
        """GA forwards early_stopping to the process-pool engines."""
        ga = GA(
            plateau_fitness,
            genome_length=2,
            population_size=8,
            generations=6,
            islands=islands,
            migration_interval=1,
            migration_count=1,
            n_workers=1,
            parallel=True,
            seed=7,
            early_stopping=1,
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = ga.run()

        assert result.strategy == strategy
        assert result.generations == 2
        assert len(result.fitness_history) == result.generations + 1
        # early_stopping is honored now, so nothing may warn about ignoring it.
        assert [w for w in record if issubclass(w.category, UserWarning)] == []

    @pytest.mark.skipif(
        is_free_threaded(),
        reason="free-threaded Python always selects a Rust path",
    )
    def test_facade_forwards_mutation_rate_end(self):
        """GA forwards mutation_rate_end, and it changes the search."""

        def run(**kwargs):
            return GA(
                neg_sphere_objective,
                genome_length=4,
                population_size=8,
                generations=5,
                bounds=(-5.0, 5.0),
                n_workers=1,
                parallel=True,
                seed=1234,
                mutation_rate=0.5,
                **kwargs,
            ).run()

        baseline = run()
        decayed = run(mutation_rate_end=0.0)

        assert baseline.strategy == "parallel"
        assert decayed.strategy == "parallel"
        assert baseline.best_fitness != decayed.best_fitness


class TestFacadeMigrationTopology:
    """`GA(islands=N)` is the recommended island API, so it must be able to pick a topology.

    Topology is not cosmetic: on the standard multimodal benchmarks a fully connected
    topology reaches a better optimum than a ring at the same budget, because a ring slows
    how fast a good solution reaches the islands that have not found it.
    """

    @staticmethod
    def _sphere(genes):
        return -float(np.sum(np.asarray(genes, dtype=np.float64) ** 2))

    def _run(self, topology, seed=0):
        kwargs = dict(
            genome_length=6,
            bounds=(-5.0, 5.0),
            population_size=80,
            generations=20,
            islands=4,
            seed=seed,
            parallel=False,
        )
        if topology is not None:
            kwargs["migration_topology"] = topology
        return GA(self._sphere, **kwargs).run()

    def test_topology_is_accepted_and_reaches_the_engine(self):
        """Two different topologies on the same seed must not produce the same run."""
        ring = self._run(MigrationTopology.ring())
        full = self._run(MigrationTopology.fully_connected())

        assert ring.best_fitness is not None
        assert full.best_fitness is not None
        # Same seed, same budget: if topology were dropped these would be identical.
        assert ring.best_fitness != full.best_fitness

    def test_default_is_unchanged_when_not_supplied(self):
        """Omitting the argument must behave exactly as before it existed."""
        explicit_ring = self._run(MigrationTopology.ring())
        default = self._run(None)
        assert default.best_fitness == explicit_ring.best_fitness

    def test_single_population_warns_that_topology_is_ignored(self):
        """A topology means nothing without islands, and silence would hide that."""
        with pytest.warns(UserWarning, match="migration_topology"):
            GA(
                self._sphere,
                genome_length=6,
                bounds=(-5.0, 5.0),
                population_size=40,
                generations=5,
                islands=1,
                parallel=False,
                migration_topology=MigrationTopology.ring(),
            ).run()

