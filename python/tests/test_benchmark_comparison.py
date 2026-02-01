"""Benchmark comparison between ParGA and DEAP on standard test problems.

This module compares runtime performance and solution quality between:
- ParGA (Rust-based with Python bindings, auto-strategy selection)
- DEAP (Pure Python genetic algorithm library)

Test problems are well-accepted academic benchmarks:
- Sphere (De Jong's function 1)
- Rastrigin
- Rosenbrock
- Ackley
- Griewank

All problems are minimization, so we use parga.minimize().
"""

import random

import numpy as np
import pytest
from deap import base, creator, tools

from parga import minimize

# =============================================================================
# Test Problem Definitions
# =============================================================================


def sphere(x):
    """Sphere function: f(x) = sum(x_i^2). Global min at origin = 0."""
    return np.sum(np.array(x) ** 2)


def rastrigin(x):
    """Rastrigin function. Global min at origin = 0."""
    x = np.array(x)
    a = 10
    n = len(x)
    return a * n + np.sum(x**2 - a * np.cos(2 * np.pi * x))


def rosenbrock(x):
    """Rosenbrock function. Global min at (1,1,...,1) = 0."""
    x = np.array(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley(x):
    """Ackley function. Global min at origin = 0."""
    x = np.array(x)
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def griewank(x):
    """Griewank function. Global min at origin = 0."""
    x = np.array(x)
    n = len(x)
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
    return sum_term - prod_term + 1


# =============================================================================
# DEAP Setup
# =============================================================================


def setup_deap(fitness_func, n_dim, lower, upper):
    """Setup DEAP for a given problem."""
    # Clean up any previous DEAP classes
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimization
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_float", random.uniform, lower, upper)

    # Individual and population
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=n_dim,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("evaluate", lambda ind: (fitness_func(ind),))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_deap(toolbox, pop_size, n_gen, cxpb=0.8, mutpb=0.1):
    """Run DEAP algorithm and return best fitness (minimization)."""
    random.seed(42)
    pop = toolbox.population(n=pop_size)

    # Evaluate initial population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for _ in range(n_gen):
        # Select next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

    # Return best fitness (minimization - lower is better)
    best = tools.selBest(pop, 1)[0]
    return best.fitness.values[0]


# =============================================================================
# Problem Configurations
# =============================================================================

PROBLEMS = {
    "sphere": {
        "fitness": sphere,
        "bounds": (-5.12, 5.12),
        "optimal": 0.0,
    },
    "rastrigin": {
        "fitness": rastrigin,
        "bounds": (-5.12, 5.12),
        "optimal": 0.0,
    },
    "rosenbrock": {
        "fitness": rosenbrock,
        "bounds": (-5.0, 10.0),
        "optimal": 0.0,
    },
    "ackley": {
        "fitness": ackley,
        "bounds": (-5.0, 5.0),
        "optimal": 0.0,
    },
    "griewank": {
        "fitness": griewank,
        "bounds": (-600.0, 600.0),
        "optimal": 0.0,
    },
}


# =============================================================================
# Benchmark Parameters
# =============================================================================

POP_SIZE = 100
N_GEN = 100
N_DIM = 10


# =============================================================================
# Runtime Benchmarks
# =============================================================================


class TestRuntimeBenchmarks:
    """Runtime performance comparison between ParGA and DEAP."""

    @pytest.mark.parametrize("problem_name", ["sphere", "rastrigin", "rosenbrock"])
    def test_parga_runtime(self, benchmark, problem_name):
        """Benchmark ParGA runtime using unified GA interface."""
        problem = PROBLEMS[problem_name]

        def run():
            result = minimize(
                problem["fitness"],
                genome_length=N_DIM,
                bounds=problem["bounds"],
                population_size=POP_SIZE,
                generations=N_GEN,
                seed=42,
            )
            return -result.best_fitness  # Convert back to minimization value

        result = benchmark(run)
        benchmark.extra_info["best_fitness"] = result
        benchmark.extra_info["library"] = "parga"
        benchmark.extra_info["problem"] = problem_name

    @pytest.mark.parametrize("problem_name", ["sphere", "rastrigin", "rosenbrock"])
    def test_deap_runtime(self, benchmark, problem_name):
        """Benchmark DEAP runtime."""
        problem = PROBLEMS[problem_name]
        lower, upper = problem["bounds"]
        toolbox = setup_deap(problem["fitness"], N_DIM, lower, upper)

        def run():
            return run_deap(toolbox, POP_SIZE, N_GEN)

        result = benchmark(run)
        benchmark.extra_info["best_fitness"] = result
        benchmark.extra_info["library"] = "deap"
        benchmark.extra_info["problem"] = problem_name


# =============================================================================
# Solution Quality Tests
# =============================================================================


class TestSolutionQuality:
    """Solution quality comparison between ParGA and DEAP."""

    @pytest.mark.parametrize("problem_name", list(PROBLEMS.keys()))
    def test_parga_quality(self, problem_name):
        """Test ParGA solution quality using unified interface."""
        problem = PROBLEMS[problem_name]

        result = minimize(
            problem["fitness"],
            genome_length=N_DIM,
            bounds=problem["bounds"],
            population_size=POP_SIZE,
            generations=N_GEN,
            seed=42,
        )

        actual_min = -result.best_fitness
        print(
            f"\nParGA {problem_name}: min = {actual_min:.6f} "
            f"(optimal = {problem['optimal']}, strategy = {result.strategy})"
        )

        # Should find a reasonable solution
        assert actual_min < 1000, f"ParGA failed on {problem_name}"

    @pytest.mark.parametrize("problem_name", list(PROBLEMS.keys()))
    def test_deap_quality(self, problem_name):
        """Test DEAP solution quality."""
        problem = PROBLEMS[problem_name]
        lower, upper = problem["bounds"]
        toolbox = setup_deap(problem["fitness"], N_DIM, lower, upper)

        best_fitness = run_deap(toolbox, POP_SIZE, N_GEN)

        print(f"\nDEAP {problem_name}: min = {best_fitness:.6f} (optimal = {problem['optimal']})")

        # Should find a reasonable solution
        assert best_fitness < 1000, f"DEAP failed on {problem_name}"


# =============================================================================
# Island Model Tests
# =============================================================================


class TestIslandModel:
    """Island model benchmark for ParGA."""

    @pytest.mark.parametrize("problem_name", ["sphere", "rastrigin"])
    def test_parga_island_runtime(self, benchmark, problem_name):
        """Benchmark ParGA island model using unified interface."""
        problem = PROBLEMS[problem_name]

        def run():
            result = minimize(
                problem["fitness"],
                genome_length=N_DIM,
                bounds=problem["bounds"],
                population_size=POP_SIZE,
                generations=N_GEN,
                islands=4,  # Use island model
                seed=42,
            )
            return -result.best_fitness

        result = benchmark(run)
        benchmark.extra_info["best_fitness"] = result
        benchmark.extra_info["library"] = "parga_island"
        benchmark.extra_info["problem"] = problem_name


# =============================================================================
# Parallel Execution Tests
# =============================================================================


class TestParallelExecution:
    """Test parallel execution with expensive fitness functions."""

    def test_parallel_auto_detection(self, benchmark):
        """Test that expensive fitness auto-selects parallel strategy."""

        def expensive_sphere(x):
            # Simulate expensive computation
            for _ in range(500):
                _ = np.linalg.norm(x)
            return np.sum(x**2)

        def run():
            result = minimize(
                expensive_sphere,
                genome_length=N_DIM,
                bounds=(-5.12, 5.12),
                population_size=50,
                generations=10,
                seed=42,
            )
            return result.strategy

        strategy = benchmark(run)
        # Should auto-select parallel for expensive fitness
        assert strategy in ("parallel", "rust"), f"Unexpected strategy: {strategy}"


# =============================================================================
# Scalability Tests
# =============================================================================


class TestScalability:
    """Test how runtime scales with problem size."""

    @pytest.mark.parametrize("n_dim", [5, 10, 20, 50])
    def test_parga_scalability(self, benchmark, n_dim):
        """Benchmark ParGA scalability with dimension."""

        def run():
            result = minimize(
                sphere,
                genome_length=n_dim,
                bounds=(-5.12, 5.12),
                population_size=POP_SIZE,
                generations=50,
                seed=42,
            )
            return -result.best_fitness

        benchmark(run)
        benchmark.extra_info["n_dim"] = n_dim
        benchmark.extra_info["library"] = "parga"

    @pytest.mark.parametrize("n_dim", [5, 10, 20, 50])
    def test_deap_scalability(self, benchmark, n_dim):
        """Benchmark DEAP scalability with dimension."""
        toolbox = setup_deap(sphere, n_dim, -5.12, 5.12)

        def run():
            return run_deap(toolbox, POP_SIZE, 50)

        benchmark(run)
        benchmark.extra_info["n_dim"] = n_dim
        benchmark.extra_info["library"] = "deap"


# =============================================================================
# Population Size Scaling
# =============================================================================


class TestPopulationScaling:
    """Test how runtime scales with population size."""

    @pytest.mark.parametrize("pop_size", [50, 100, 200, 500])
    def test_parga_population_scaling(self, benchmark, pop_size):
        """Benchmark ParGA with different population sizes."""

        def run():
            result = minimize(
                sphere,
                genome_length=N_DIM,
                bounds=(-5.12, 5.12),
                population_size=pop_size,
                generations=50,
                seed=42,
            )
            return -result.best_fitness

        benchmark(run)
        benchmark.extra_info["pop_size"] = pop_size
        benchmark.extra_info["library"] = "parga"

    @pytest.mark.parametrize("pop_size", [50, 100, 200, 500])
    def test_deap_population_scaling(self, benchmark, pop_size):
        """Benchmark DEAP with different population sizes."""
        toolbox = setup_deap(sphere, N_DIM, -5.12, 5.12)

        def run():
            return run_deap(toolbox, pop_size, 50)

        benchmark(run)
        benchmark.extra_info["pop_size"] = pop_size
        benchmark.extra_info["library"] = "deap"


if __name__ == "__main__":
    import time

    print("=" * 70)
    print("ParGA vs DEAP Comparison")
    print("=" * 70)
    print(f"Population: {POP_SIZE}, Generations: {N_GEN}, Dimensions: {N_DIM}")
    print()

    for problem_name, problem in PROBLEMS.items():
        print(f"{problem_name.upper()} (optimal = {problem['optimal']})")
        print("-" * 50)

        # ParGA using unified minimize()
        start = time.perf_counter()
        result = minimize(
            problem["fitness"],
            genome_length=N_DIM,
            bounds=problem["bounds"],
            population_size=POP_SIZE,
            generations=N_GEN,
            seed=42,
        )
        parga_time = time.perf_counter() - start
        parga_min = -result.best_fitness

        # DEAP
        lower, upper = problem["bounds"]
        toolbox = setup_deap(problem["fitness"], N_DIM, lower, upper)
        start = time.perf_counter()
        deap_min = run_deap(toolbox, POP_SIZE, N_GEN)
        deap_time = time.perf_counter() - start

        speedup = deap_time / parga_time if parga_time > 0 else float("inf")

        print(f"  ParGA: {parga_time * 1000:8.2f} ms | min = {parga_min:.6f} | {result.strategy}")
        print(f"  DEAP:  {deap_time * 1000:8.2f} ms | min = {deap_min:.6f}")
        print(f"  Speedup: {speedup:.2f}x")
        print()

    print("=" * 70)
    print("Run full benchmarks: pytest python/tests/test_benchmark_comparison.py --benchmark-only")
    print("=" * 70)
