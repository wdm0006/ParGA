"""Tests for the parga Python bindings."""

import warnings

import numpy as np
import pytest

from parga import parallel as parallel_module

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

    def test_zero_restart_on_stagnation_is_rejected_at_construction(self):
        """Patience 0 used to panic inside run() with a PanicException, a
        BaseException subclass that ``except Exception`` does not catch."""
        with pytest.raises(ValueError, match="restart_on_stagnation"):
            GA(
                simple_fitness,
                genome_length=3,
                population_size=20,
                generations=5,
                bounds=(-5, 5),
                parallel=False,
                seed=7,
                restart_on_stagnation=0,
            )

    @pytest.mark.parametrize(
        ("parallel", "islands"), [(False, 1), (True, 1), (False, 2), (True, 2)]
    )
    def test_zero_early_stopping_is_rejected_at_construction(
        self, parallel, islands
    ):
        """Patience 0 used to truncate every run to a single generation."""
        with pytest.raises(ValueError, match="early_stopping"):
            GA(
                simple_fitness,
                genome_length=3,
                population_size=20,
                generations=10,
                bounds=(-5, 5),
                parallel=parallel,
                islands=islands,
                migration_interval=1,
                migration_count=1,
                seed=7,
                early_stopping=0,
            )

    def test_patience_of_one_still_runs(self):
        settings = {
            "genome_length": 3,
            "population_size": 20,
            "generations": 10,
            "bounds": (-5, 5),
            "parallel": False,
            "seed": 7,
        }
        early = GA(simple_fitness, early_stopping=1, **settings).run()
        assert early.strategy == "rust"
        assert early.generations == 3

        restart = GA(simple_fitness, restart_on_stagnation=1, **settings).run()
        assert restart.generations == 10

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
        """The process-pool paths warn, naming each option they still drop."""
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
            crossover_method=CrossoverMethod.arithmetic(),
            mutation_method=MutationMethod.uniform(),
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
        assert "restart_on_stagnation" in message
        assert strategy in message
        # Operator overrides are applied by the process-pool engines now.
        for name in ("selection_method", "crossover_method", "mutation_method"):
            assert name not in message
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

    def test_engines_reject_zero_early_stopping(self):
        with pytest.raises(ValueError, match="early_stopping"):
            _parallel_ga(early_stopping=0)
        with pytest.raises(ValueError, match="early_stopping"):
            _parallel_island(early_stopping=0)

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


class TestRustIslandEngineOptions:
    """early_stopping and mutation_rate_end on the Rust island strategy."""

    @staticmethod
    def _island_ga(**kwargs) -> GA:
        settings = {
            "fitness_fn": plateau_fitness,
            "genome_length": 2,
            "population_size": 8,
            "generations": 10,
            "islands": 2,
            "migration_interval": 1,
            "migration_count": 1,
            "parallel": False,
            "seed": 42,
        }
        settings.update(kwargs)
        return GA(**settings)

    def test_early_stopping_stops_early(self):
        """A plateaued objective stops before the configured generation count."""
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = self._island_ga(early_stopping=1).run()

        assert result.strategy == "rust_island"
        assert result.generations < 10
        # gen 0 improves on -inf, gen 1 stagnates and trips patience=1.
        assert result.generations == 2
        assert len(result.fitness_history) == result.generations + 1
        # Both options are honored now, so nothing may warn about ignoring them.
        assert [w for w in record if issubclass(w.category, UserWarning)] == []

    def test_without_early_stopping_runs_every_generation(self):
        """Default (None) keeps today's behavior: no early exit."""
        result = self._island_ga().run()

        assert result.strategy == "rust_island"
        assert result.generations == 10
        assert len(result.fitness_history) == 11

    def test_mutation_rate_end_changes_the_search(self):
        """GA forwards mutation_rate_end, and it changes the search."""

        def run(**kwargs):
            return self._island_ga(
                fitness_fn=neg_sphere_objective,
                genome_length=4,
                generations=5,
                bounds=(-5.0, 5.0),
                mutation_rate=0.5,
                **kwargs,
            ).run()

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            baseline = run()
            decayed = run(mutation_rate_end=0.0)
            # The identity case must reproduce the un-decayed run exactly.
            flat = run(mutation_rate_end=0.5)

        assert baseline.strategy == "rust_island"
        assert baseline.best_fitness != decayed.best_fitness
        assert flat.best_fitness == baseline.best_fitness
        np.testing.assert_array_equal(flat.best_genes(), baseline.best_genes())
        assert [w for w in record if issubclass(w.category, UserWarning)] == []

    @pytest.mark.parametrize("bad_rate", [-0.1, 1.5, float("nan")])
    def test_rejects_invalid_mutation_rate_end(self, bad_rate):
        """mutation_rate_end is validated like every other rate."""
        with pytest.raises(ValueError, match="mutation_rate_end"):
            self._island_ga(mutation_rate_end=bad_rate)


def _facade_callback_ga(strategy, callback=None, **overrides):
    """Build a seeded GA forced onto one of the four strategies."""
    kwargs = {
        "fitness_fn": neg_sphere_objective,
        "genome_length": 3,
        "population_size": 8,
        "generations": 6,
        "bounds": (-5.0, 5.0),
        "elitism": 2,
        "seed": 3,
        "parallel": strategy.startswith("parallel"),
        "n_workers": 1,
        "callback": callback,
    }
    if strategy.endswith("island"):
        kwargs.update(islands=2, migration_interval=1, migration_count=1)
    kwargs.update(overrides)
    return GA(**kwargs)


CALLBACK_STRATEGIES = [
    "rust",
    "rust_island",
    pytest.param(
        "parallel",
        marks=pytest.mark.skipif(is_free_threaded(), reason="forces Rust"),
    ),
    pytest.param(
        "parallel_island",
        marks=pytest.mark.skipif(is_free_threaded(), reason="forces Rust"),
    ),
]


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("strategy", CALLBACK_STRATEGIES)
class TestFacadeCallback:
    """The per-generation callback behaves identically on every strategy."""

    def test_called_once_per_generation_with_monotone_best(self, strategy):
        calls = []

        def callback(generation, best_fitness, best_genes):
            calls.append((generation, best_fitness, np.array(best_genes)))

        result = _facade_callback_ga(strategy, callback).run()

        assert result.strategy == strategy
        assert [c[0] for c in calls] == list(range(1, result.generations + 1))
        fitnesses = [c[1] for c in calls]
        assert fitnesses == sorted(fitnesses)
        assert fitnesses[-1] == result.best_fitness
        for _, fitness, genes in calls:
            assert genes.shape == (3,)
            assert neg_sphere_objective(genes) == pytest.approx(fitness)

    @pytest.mark.parametrize("stop_at", [1, 3])
    def test_true_stops_at_that_generation(self, strategy, stop_at):
        result = _facade_callback_ga(
            strategy, lambda generation, *_: generation == stop_at
        ).run()

        assert result.generations == stop_at
        assert len(result.fitness_history) == stop_at + 1

    def test_falsy_returns_do_not_stop_or_change_result(self, strategy):
        baseline = _facade_callback_ga(strategy).run()
        for value in (None, False, 0):
            result = _facade_callback_ga(strategy, lambda *_, v=value: v).run()
            assert result.generations == baseline.generations == 6
            assert result.fitness_history == baseline.fitness_history
            assert result.best_fitness == baseline.best_fitness
            np.testing.assert_array_equal(result.best_genes(), baseline.best_genes())

    def test_exception_propagates_unchanged(self, strategy):
        class Boom(Exception):
            pass

        def callback(generation, *_):
            if generation == 2:
                raise Boom("stop here")

        with pytest.raises(Boom, match="stop here"):
            _facade_callback_ga(strategy, callback).run()

    def test_callback_runs_before_early_stopping_check(self, strategy):
        seen = []
        result = _facade_callback_ga(
            strategy,
            lambda generation, *_: seen.append(generation),
            fitness_fn=plateau_fitness,
            early_stopping=1,
        ).run()

        assert seen == list(range(1, result.generations + 1))
        assert result.generations < 6


def test_callback_must_be_callable():
    with pytest.raises(ValueError, match="callback"):
        GA(neg_sphere_objective, genome_length=2, callback=42)


def test_rust_callback_none_matches_unset_engine_run():
    """Low-level engines: run() and run(callback=None) are the same run."""
    kwargs = {"genome_length": 3, "population_size": 8, "generations": 5, "seed": 1}
    first = GeneticAlgorithm(neg_sphere_objective, **kwargs).run()
    second = GeneticAlgorithm(neg_sphere_objective, **kwargs).run(None)
    assert list(first.fitness_history()) == list(second.fitness_history())


def _rank_sorted(values):
    return np.argsort(-np.asarray(values, dtype=float), kind="stable")


class TestParallelSelectionOperators:
    """Value tests for the NumPy selection operators behind the process pool."""

    N_DRAWS = 20000

    @staticmethod
    def _freq(kind, params, fitness, count=N_DRAWS, seed=0):
        picks = parallel_module._select_indices(
            kind, params, fitness, count, np.random.default_rng(seed)
        )
        return np.bincount(picks, minlength=len(fitness)) / count

    def test_truncation_cycles_through_top_slice_best_first(self):
        fitness = [5.0, 9.0, 1.0, 7.0]  # best-first order: 1, 3, 0, 2
        picks = parallel_module._select_indices(
            "truncation", {"ratio": 0.5}, fitness, 5, np.random.default_rng(0)
        )
        assert picks.tolist() == [1, 3, 1, 3, 1]

    def test_truncation_cutoff_is_clamped(self):
        fitness = [5.0, 9.0, 1.0, 7.0]
        rng = np.random.default_rng(0)
        tiny = parallel_module._select_indices("truncation", {"ratio": 1e-9}, fitness, 3, rng)
        assert tiny.tolist() == [1, 1, 1]
        full = parallel_module._select_indices("truncation", {"ratio": 1.0}, fitness, 4, rng)
        assert full.tolist() == [1, 3, 0, 2]

    def test_tournament_as_large_as_population_always_picks_best(self):
        freq = self._freq("tournament", {"size": 4.0}, [0.0, 3.0, 1.0, 2.0], count=200)
        assert freq.tolist() == [0.0, 1.0, 0.0, 0.0]

    def test_tournament_size_one_is_uniform(self):
        freq = self._freq("tournament", {"size": 1.0}, [0.0, 3.0, 1.0, 2.0])
        np.testing.assert_allclose(freq, 0.25, atol=0.02)

    def test_random_is_uniform_regardless_of_fitness(self):
        freq = self._freq("random", {}, [100.0, 0.0, -5.0, 1.0])
        np.testing.assert_allclose(freq, 0.25, atol=0.02)

    def test_rank_weights_follow_rank_not_fitness(self):
        # Ranks (worst=1 .. best=4) give 0.4 / 0.3 / 0.2 / 0.1 from best down.
        fitness = [1.0, 1000.0, -3.0, 2.0]  # best-first: idx 1, 3, 0, 2
        freq = self._freq("rank", {}, fitness)
        np.testing.assert_allclose(freq[[1, 3, 0, 2]], [0.4, 0.3, 0.2, 0.1], atol=0.02)

    def test_roulette_is_proportional_to_fitness(self):
        freq = self._freq("roulette", {}, [1.0, 3.0])
        np.testing.assert_allclose(freq, [0.25, 0.75], atol=0.02)

    def test_roulette_shifts_negative_fitness_positive(self):
        # offset = 3 + 1 = 4, so weights are 3 and 1.
        freq = self._freq("roulette", {}, [-1.0, -3.0])
        np.testing.assert_allclose(freq, [0.75, 0.25], atol=0.02)

    def test_roulette_with_all_zero_fitness_falls_back_to_uniform(self):
        freq = self._freq("roulette", {}, [0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(freq, 0.25, atol=0.02)

    def test_stochastic_universal_is_exactly_proportional(self):
        # Four evenly spaced pointers over weights 3:1 hit them 3 and 1 times.
        for seed in range(10):
            picks = parallel_module._select_indices(
                "stochastic_universal", {}, [3.0, 1.0], 4, np.random.default_rng(seed)
            )
            assert np.bincount(picks, minlength=2).tolist() == [3, 1]

    def test_stochastic_universal_uniform_fitness_selects_each_once(self):
        picks = parallel_module._select_indices(
            "stochastic_universal", {}, [2.0] * 5, 5, np.random.default_rng(3)
        )
        assert sorted(picks.tolist()) == [0, 1, 2, 3, 4]


class TestParallelCrossoverOperators:
    """Value tests for the NumPy crossover operators behind the process pool."""

    P1 = np.zeros(6)
    P2 = np.ones(6)

    def test_single_point_swaps_the_tail_at_the_drawn_point(self):
        point = int(np.random.default_rng(4).integers(0, 7))
        c1, c2 = parallel_module._recombine(
            "single_point", {}, self.P1, self.P2, np.random.default_rng(4)
        )
        np.testing.assert_array_equal(c1, [0.0] * point + [1.0] * (6 - point))
        np.testing.assert_array_equal(c2, [1.0] * point + [0.0] * (6 - point))

    def test_two_point_swaps_the_middle_segment(self):
        a, b = sorted(int(v) for v in np.random.default_rng(9).integers(0, 7, size=2))
        c1, c2 = parallel_module._recombine(
            "two_point", {}, self.P1, self.P2, np.random.default_rng(9)
        )
        expected = np.zeros(6)
        expected[a:b] = 1.0
        np.testing.assert_array_equal(c1, expected)
        np.testing.assert_array_equal(c2, 1.0 - expected)

    def test_uniform_probability_extremes(self):
        rng = np.random.default_rng(0)
        c1, c2 = parallel_module._recombine("uniform", {"probability": 0.0}, self.P1, self.P2, rng)
        np.testing.assert_array_equal(c1, self.P1)
        np.testing.assert_array_equal(c2, self.P2)
        c1, c2 = parallel_module._recombine("uniform", {"probability": 1.0}, self.P1, self.P2, rng)
        np.testing.assert_array_equal(c1, self.P2)
        np.testing.assert_array_equal(c2, self.P1)

    def test_uniform_swaps_genes_where_the_draw_is_below_probability(self):
        swap = np.random.default_rng(2).random(6) < 0.5
        c1, c2 = parallel_module._recombine(
            "uniform", {"probability": 0.5}, self.P1, self.P2, np.random.default_rng(2)
        )
        np.testing.assert_array_equal(c1, swap.astype(float))
        np.testing.assert_array_equal(c2, (~swap).astype(float))

    def test_arithmetic_blends_with_one_shared_weight(self):
        alpha = np.random.default_rng(5).random()
        c1, c2 = parallel_module._recombine(
            "arithmetic", {}, self.P1, self.P2, np.random.default_rng(5)
        )
        np.testing.assert_allclose(c1, np.full(6, 1.0 - alpha))
        np.testing.assert_allclose(c2, np.full(6, alpha))

    def test_blend_alpha_zero_stays_between_parents(self):
        c1, c2 = parallel_module._recombine(
            "blend", {"alpha": 0.0}, self.P1, self.P2, np.random.default_rng(1)
        )
        assert np.all((c1 >= 0.0) & (c1 <= 1.0))
        assert np.all((c2 >= 0.0) & (c2 <= 1.0))

    def test_blend_alpha_widens_the_sampling_interval(self):
        rng = np.random.default_rng(1)
        samples = np.concatenate(
            [
                np.concatenate(
                    parallel_module._recombine("blend", {"alpha": 0.5}, self.P1, self.P2, rng)
                )
                for _ in range(200)
            ]
        )
        assert samples.min() < -0.4 and samples.max() > 1.4
        assert samples.min() >= -0.5 and samples.max() <= 1.5

    def test_simulated_binary_matches_the_spread_factor_formula(self):
        eta = 2.0
        u = np.random.default_rng(6).random(6)
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
        c1, c2 = parallel_module._recombine(
            "simulated_binary", {"eta": eta}, self.P1, self.P2, np.random.default_rng(6)
        )
        np.testing.assert_allclose(c1, 0.5 * (1 - beta))
        np.testing.assert_allclose(c2, 0.5 * (1 + beta))
        np.testing.assert_allclose(c1 + c2, self.P1 + self.P2)


class TestParallelMutationOperators:
    """Value tests for the NumPy mutation operators behind the process pool."""

    N_GENES = 20000

    @staticmethod
    def _mutate(kind, params, genes, lower, upper, rate=1.0, seed=0):
        genes = np.array(genes, dtype=float)
        return parallel_module._mutate_genes(
            kind, params, genes, rate, lower, upper, np.random.default_rng(seed)
        )

    @pytest.mark.parametrize(
        "kind,params",
        [("gaussian", {"sigma": 0.1}), ("uniform", {}), ("polynomial", {"eta": 20.0}), ("boundary", {})],
    )
    def test_rate_zero_leaves_genes_untouched(self, kind, params):
        out = self._mutate(kind, params, [1.0, 2.0, 3.0], [-5.0] * 3, [5.0] * 3, rate=0.0)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_gaussian_sigma_is_a_fraction_of_the_gene_range(self):
        n = self.N_GENES
        out = self._mutate("gaussian", {"sigma": 0.001}, np.zeros(n), [-1000.0] * n, [1000.0] * n)
        assert out.std() == pytest.approx(2.0, rel=0.05)  # 0.001 * 2000
        assert abs(out.mean()) < 0.1

    def test_gaussian_clips_to_bounds(self):
        out = self._mutate("gaussian", {"sigma": 5.0}, np.zeros(500), [-1.0] * 500, [1.0] * 500)
        assert out.min() >= -1.0 and out.max() <= 1.0
        assert (out == 1.0).any() and (out == -1.0).any()

    def test_uniform_resamples_across_the_full_range(self):
        n = self.N_GENES
        out = self._mutate("uniform", {}, np.full(n, 3.0), [-2.0] * n, [6.0] * n)
        assert out.min() >= -2.0 and out.max() <= 6.0
        assert out.mean() == pytest.approx(2.0, abs=0.1)

    def test_boundary_sets_each_gene_to_one_of_its_bounds(self):
        n = 400
        lower = np.linspace(-5.0, -1.0, n)
        upper = np.linspace(1.0, 5.0, n)
        out = self._mutate("boundary", {}, np.zeros(n), lower, upper)
        at_lower = out == lower
        at_upper = out == upper
        assert np.all(at_lower | at_upper)
        assert 0.4 < at_lower.mean() < 0.6

    def test_polynomial_gene_on_lower_bound_only_moves_up(self):
        n = self.N_GENES
        out = self._mutate("polynomial", {"eta": 20.0}, np.full(n, -5.0), [-5.0] * n, [5.0] * n)
        assert out.min() >= -5.0
        # E[displacement] = range / (2 * (eta + 2)) = 10 / 44
        assert (out - -5.0).mean() == pytest.approx(10.0 / 44.0, rel=0.1)

    def test_polynomial_gene_on_upper_bound_only_moves_down(self):
        n = self.N_GENES
        out = self._mutate("polynomial", {"eta": 20.0}, np.full(n, 5.0), [-5.0] * n, [5.0] * n)
        assert out.max() <= 5.0
        assert (5.0 - out).mean() == pytest.approx(10.0 / 44.0, rel=0.1)

    def test_polynomial_skips_zero_width_bounds(self):
        out = self._mutate("polynomial", {"eta": 20.0}, [2.0, 0.0], [2.0, -1.0], [2.0, 1.0])
        assert out[0] == 2.0
        assert out[1] != 0.0

    def test_only_masked_genes_change(self):
        genes = np.full(2000, 1.5)
        out = self._mutate("uniform", {}, genes, [1.0] * 2000, [2.0] * 2000, rate=0.3)
        moved = (out != 1.5).mean()
        assert moved == pytest.approx(0.3, abs=0.04)


class TestResolveOperator:
    def test_native_wrappers_resolve_to_kind_and_params(self):
        resolve = parallel_module._resolve_operator
        assert resolve(
            SelectionMethod.tournament(5), "selection_method", parallel_module._SELECTION_KINDS
        ) == ("tournament", {"size": 5.0})
        assert resolve(
            CrossoverMethod.blend(0.3), "crossover_method", parallel_module._CROSSOVER_KINDS
        ) == ("blend", {"alpha": 0.3})
        assert resolve(
            MutationMethod.gaussian(0.2), "mutation_method", parallel_module._MUTATION_KINDS
        ) == ("gaussian", {"sigma": 0.2})
        assert resolve(None, "mutation_method", parallel_module._MUTATION_KINDS) is None

    def test_unsupported_kind_and_foreign_objects_are_rejected(self):
        with pytest.raises(ValueError, match="custom"):
            parallel_module._resolve_operator(
                ("custom", {}), "mutation_method", parallel_module._MUTATION_KINDS
            )
        with pytest.raises(TypeError, match="kind"):
            parallel_module._resolve_operator(
                object(), "mutation_method", parallel_module._MUTATION_KINDS
            )

    def test_tournament_larger_than_population_is_rejected(self):
        with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="tournament size"):
            ParallelGA(
                simple_fitness,
                genome_length=2,
                population_size=8,
                selection_method=SelectionMethod.tournament(9),
            )


OPERATOR_CASES = [
    ("selection_method", SelectionMethod.tournament(2)),
    ("selection_method", SelectionMethod.roulette()),
    ("selection_method", SelectionMethod.rank()),
    ("selection_method", SelectionMethod.random()),
    ("selection_method", SelectionMethod.truncation(0.5)),
    ("selection_method", SelectionMethod.stochastic_universal()),
    ("crossover_method", CrossoverMethod.single_point()),
    ("crossover_method", CrossoverMethod.two_point()),
    ("crossover_method", CrossoverMethod.uniform(0.3)),
    ("crossover_method", CrossoverMethod.blend(0.1)),
    ("crossover_method", CrossoverMethod.simulated_binary(5.0)),
    ("crossover_method", CrossoverMethod.arithmetic()),
    ("mutation_method", MutationMethod.gaussian(0.3)),
    ("mutation_method", MutationMethod.uniform()),
    ("mutation_method", MutationMethod.polynomial(10.0)),
    ("mutation_method", MutationMethod.boundary()),
]


def neg_sphere(genes: np.ndarray) -> float:
    return -float(np.sum(genes**2))


class TestParallelEngineOperators:
    """Operator overrides reach the process-pool engines end to end."""

    @staticmethod
    def _ga_kwargs():
        return {
            "fitness_fn": neg_sphere,
            "genome_length": 4,
            "population_size": 20,
            "generations": 8,
            "n_workers": 2,
            "mutation_rate": 0.3,
            "seed": 5,
            "lower_bounds": [-5.0] * 4,
            "upper_bounds": [5.0] * 4,
        }

    def test_default_run_is_bit_identical_to_pre_override_engine(self):
        """Values recorded from master before operator overrides existed."""
        with pytest.warns(DeprecationWarning):
            result = ParallelGA(**self._ga_kwargs()).run()
        assert result.best_fitness == pytest.approx(-0.3912107980672784, rel=1e-12)
        np.testing.assert_allclose(
            result.best_genes(),
            [-0.5202995163033445, -0.2828782234052866, 0.07323208405840687, 0.1873931268465184],
            rtol=1e-12,
        )

    def test_island_default_run_is_bit_identical_to_pre_override_engine(self):
        with pytest.warns(DeprecationWarning):
            result = ParallelIslandModel(
                neg_sphere,
                4,
                num_islands=2,
                island_population=10,
                generations=8,
                migration_interval=2,
                migration_count=1,
                n_workers=2,
                mutation_rate=0.3,
                mutation_rate_end=0.05,
                seed=5,
                lower_bounds=[-5.0] * 4,
                upper_bounds=[5.0] * 4,
            ).run()
        assert result.best_fitness == pytest.approx(-0.28062251338167354, rel=1e-12)

    def test_explicit_default_operators_match_no_override(self):
        with pytest.warns(DeprecationWarning):
            base = ParallelGA(**self._ga_kwargs()).run()
        with pytest.warns(DeprecationWarning):
            explicit = ParallelGA(
                **self._ga_kwargs(),
                crossover_method=CrossoverMethod.blend(0.5),
                mutation_method=MutationMethod.gaussian(0.1),
            ).run()
        assert explicit.best_fitness == base.best_fitness
        np.testing.assert_array_equal(explicit.best_genes(), base.best_genes())

    @pytest.mark.parametrize("option,method", OPERATOR_CASES, ids=lambda v: getattr(v, "kind", v))
    def test_ga_runs_with_each_operator(self, option, method):
        kwargs = self._ga_kwargs() | {"generations": 3, "n_workers": 1, option: method}
        with pytest.warns(DeprecationWarning):
            result = ParallelGA(**kwargs).run()
        assert np.isfinite(result.best_fitness)
        assert np.all(np.abs(result.best_genes()) <= 5.0)
        assert len(result.fitness_history) == result.generations + 1

    def test_overrides_change_the_search(self):
        with pytest.warns(DeprecationWarning):
            base = ParallelGA(**self._ga_kwargs()).run()
        for option, method in [
            ("selection_method", SelectionMethod.random()),
            ("crossover_method", CrossoverMethod.uniform(0.5)),
            ("mutation_method", MutationMethod.boundary()),
        ]:
            with pytest.warns(DeprecationWarning):
                other = ParallelGA(**self._ga_kwargs(), **{option: method}).run()
            assert other.best_fitness != base.best_fitness, option

    def test_island_runs_with_overrides(self):
        with pytest.warns(DeprecationWarning):
            result = ParallelIslandModel(
                neg_sphere,
                4,
                num_islands=2,
                island_population=8,
                generations=3,
                migration_interval=1,
                migration_count=1,
                n_workers=1,
                seed=5,
                lower_bounds=[-5.0] * 4,
                upper_bounds=[5.0] * 4,
                selection_method=SelectionMethod.rank(),
                crossover_method=CrossoverMethod.simulated_binary(5.0),
                mutation_method=MutationMethod.polynomial(10.0),
            ).run()
        assert np.isfinite(result.best_fitness)
        assert np.all(np.abs(result.best_genes()) <= 5.0)

    @pytest.mark.skipif(
        is_free_threaded(),
        reason="free-threaded Python always selects a Rust path",
    )
    @pytest.mark.parametrize("islands,strategy", [(1, "parallel"), (2, "parallel_island")])
    def test_facade_forwards_operators_without_warning(self, monkeypatch, islands, strategy):
        seen = []
        real = parallel_module._select_indices

        def spy(kind, *args, **kwargs):
            seen.append(kind)
            return real(kind, *args, **kwargs)

        monkeypatch.setattr(parallel_module, "_select_indices", spy)
        ga = GA(
            neg_sphere,
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
            crossover_method=CrossoverMethod.arithmetic(),
            mutation_method=MutationMethod.uniform(),
        )
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = ga.run()

        assert result.strategy == strategy
        assert set(seen) == {"rank"}
        assert [w for w in record if issubclass(w.category, UserWarning)] == []
