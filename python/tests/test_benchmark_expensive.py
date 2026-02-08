"""Benchmark: ParGA vs DEAP on expensive fitness functions.

Demonstrates that ParGA's automatic parallelization provides significant
speedup when fitness evaluation is computationally expensive, amortizing
the Python-Rust FFI and multiprocessing overhead.

Academic test problems:
1. Lennard-Jones cluster optimization (computational chemistry/physics)
2. PDE-constrained optimization via 1D heat equation (computational engineering)

On cheap fitness functions (<0.5ms), DEAP wins due to lower overhead.
On expensive functions (>0.5ms), ParGA auto-detects the cost, switches to
multiprocessing, and scales across available CPU cores.

References:
    Wales, D.J. & Doye, J.P.K. (1997). J. Phys. Chem. A, 101, 5111-5116.
"""

import random
import time

import numpy as np
import pytest
from deap import base, creator, tools

from parga import minimize

# =============================================================================
# Expensive Fitness Functions
# =============================================================================


def lennard_jones(x):
    """Lennard-Jones cluster potential for N atoms in 3D.

    Classic benchmark in computational chemistry. Genome encodes flattened
    3D coordinates of N atoms. Energy is sum of pairwise LJ interactions:
    V(r) = 4 * [(1/r)^12 - (1/r)^6].

    Known global minima:
        N=7:  -16.5054
        N=10: -28.4225
        N=13: -44.3268 (Mackay icosahedron)
        N=38: -173.9284 (truncated octahedron)
    """
    n_atoms = len(x) // 3
    positions = np.array(x).reshape(n_atoms, 3)
    energy = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            rij = positions[i] - positions[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e10
            r6 = (1.0 / r2) ** 3
            energy += 4.0 * (r6 * r6 - r6)
    return energy


def heat_equation_fit(x):
    """PDE-constrained optimization: 1D heat equation.

    Find initial temperature distribution that evolves to match a target
    Gaussian profile under the heat equation u_t = k * u_xx.

    Each fitness evaluation solves the PDE via explicit finite differences
    for 200 timesteps, then computes L2 error vs target.

    This is a standard inverse problem in computational engineering.
    """
    n = len(x)
    u = np.array(x, dtype=float)
    k = 0.1
    dx = 1.0 / (n + 1)
    dt = 0.4 * dx**2 / k  # CFL-stable timestep

    # Target: Gaussian final state
    grid = np.linspace(0, 1, n)
    target = np.exp(-((grid - 0.5) ** 2) / 0.05)

    # Integrate heat equation
    for _ in range(200):
        u_new = u.copy()
        u_new[1:-1] += k * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = 0.0  # Dirichlet BCs
        u_new[-1] = 0.0
        u = u_new

    return np.sum((u - target) ** 2)


# =============================================================================
# DEAP Setup
# =============================================================================


def setup_deap(fitness_func, n_dim, lower, upper):
    """Setup DEAP for a minimization problem."""
    if hasattr(creator, "FitnessMin"):
        del creator.FitnessMin
    if hasattr(creator, "Individual"):
        del creator.Individual

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, lower, upper)
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=n_dim,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (fitness_func(ind),))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox


def run_deap(toolbox, pop_size, n_gen, cxpb=0.8, mutpb=0.1):
    """Run DEAP GA and return best fitness."""
    random.seed(42)
    pop = toolbox.population(n=pop_size)

    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for _ in range(n_gen):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

    best = tools.selBest(pop, 1)[0]
    return best.fitness.values[0]


# =============================================================================
# Problem Configurations
# =============================================================================

# Only problems expensive enough to trigger parallel mode (>0.5ms/eval)
LJ_PROBLEMS = {
    "lj_40": {"n_atoms": 40, "bounds": (-3.0, 3.0)},
    "lj_55": {"n_atoms": 55, "bounds": (-3.5, 3.5)},
    "lj_75": {"n_atoms": 75, "bounds": (-4.0, 4.0)},
}

POP_SIZE = 100
N_GEN = 50


# =============================================================================
# Lennard-Jones Benchmarks
# =============================================================================


class TestLennardJones:
    """Lennard-Jones cluster optimization: ParGA vs DEAP."""

    @pytest.mark.parametrize("problem_name", list(LJ_PROBLEMS.keys()))
    def test_parga_lj(self, benchmark, problem_name):
        """Benchmark ParGA on Lennard-Jones clusters."""
        prob = LJ_PROBLEMS[problem_name]
        genome_length = prob["n_atoms"] * 3

        def run():
            result = minimize(
                lennard_jones,
                genome_length=genome_length,
                bounds=prob["bounds"],
                population_size=POP_SIZE,
                generations=N_GEN,
                seed=42,
            )
            return -result.best_fitness, result.strategy

        fitness, strategy = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
        benchmark.extra_info["best_fitness"] = fitness
        benchmark.extra_info["strategy"] = strategy
        benchmark.extra_info["library"] = "parga"
        benchmark.extra_info["n_atoms"] = prob["n_atoms"]

    @pytest.mark.parametrize("problem_name", list(LJ_PROBLEMS.keys()))
    def test_deap_lj(self, benchmark, problem_name):
        """Benchmark DEAP on Lennard-Jones clusters."""
        prob = LJ_PROBLEMS[problem_name]
        genome_length = prob["n_atoms"] * 3
        toolbox = setup_deap(lennard_jones, genome_length, prob["bounds"][0], prob["bounds"][1])

        def run():
            return run_deap(toolbox, POP_SIZE, N_GEN)

        fitness = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
        benchmark.extra_info["best_fitness"] = fitness
        benchmark.extra_info["library"] = "deap"
        benchmark.extra_info["n_atoms"] = prob["n_atoms"]


# =============================================================================
# Heat Equation PDE Benchmarks
# =============================================================================


class TestHeatEquation:
    """PDE-constrained optimization: ParGA vs DEAP."""

    @pytest.mark.parametrize("grid_size", [20, 50])
    def test_parga_heat(self, benchmark, grid_size):
        """Benchmark ParGA on heat equation inverse problem."""

        def run():
            result = minimize(
                heat_equation_fit,
                genome_length=grid_size,
                bounds=(-2.0, 2.0),
                population_size=POP_SIZE,
                generations=N_GEN,
                seed=42,
            )
            return -result.best_fitness, result.strategy

        fitness, strategy = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
        benchmark.extra_info["best_fitness"] = fitness
        benchmark.extra_info["strategy"] = strategy
        benchmark.extra_info["library"] = "parga"

    @pytest.mark.parametrize("grid_size", [20, 50])
    def test_deap_heat(self, benchmark, grid_size):
        """Benchmark DEAP on heat equation inverse problem."""
        toolbox = setup_deap(heat_equation_fit, grid_size, -2.0, 2.0)

        def run():
            return run_deap(toolbox, POP_SIZE, N_GEN)

        fitness = benchmark.pedantic(run, rounds=3, warmup_rounds=1)
        benchmark.extra_info["best_fitness"] = fitness
        benchmark.extra_info["library"] = "deap"


# =============================================================================
# Direct Comparison Script
# =============================================================================

if __name__ == "__main__":
    import multiprocessing

    print("=" * 70)
    print("ParGA vs DEAP: Expensive Fitness Functions")
    print("=" * 70)
    print(f"Population: {POP_SIZE}, Generations: {N_GEN}")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print()

    # Measure per-evaluation cost
    print("Per-evaluation cost (auto-parallel threshold = 0.5ms):")
    print("-" * 50)
    for name, prob in LJ_PROBLEMS.items():
        n = prob["n_atoms"]
        x = np.random.uniform(prob["bounds"][0], prob["bounds"][1], 3 * n)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            lennard_jones(x)
            times.append(time.perf_counter() - t0)
        print(f"  {name} ({n} atoms, {n*(n-1)//2} pairs): {np.median(times)*1000:.3f} ms")
    for gs in [20, 50]:
        x = np.random.uniform(-2, 2, gs)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            heat_equation_fit(x)
            times.append(time.perf_counter() - t0)
        print(f"  heat_eq (grid={gs}): {np.median(times)*1000:.3f} ms")
    print()

    # Lennard-Jones benchmarks
    print("LENNARD-JONES CLUSTER OPTIMIZATION")
    print("=" * 70)
    for name, prob in LJ_PROBLEMS.items():
        n = prob["n_atoms"]
        genome_length = n * 3
        print(f"\n{name.upper()} ({n} atoms, genome_length={genome_length})")
        print("-" * 50)

        # ParGA
        start = time.perf_counter()
        result = minimize(
            lennard_jones,
            genome_length=genome_length,
            bounds=prob["bounds"],
            population_size=POP_SIZE,
            generations=N_GEN,
            seed=42,
        )
        parga_time = time.perf_counter() - start
        parga_fitness = -result.best_fitness

        # DEAP
        toolbox = setup_deap(lennard_jones, genome_length, prob["bounds"][0], prob["bounds"][1])
        start = time.perf_counter()
        deap_fitness = run_deap(toolbox, POP_SIZE, N_GEN)
        deap_time = time.perf_counter() - start

        speedup = deap_time / parga_time if parga_time > 0 else float("inf")

        print(f"  ParGA: {parga_time:8.3f} s | energy = {parga_fitness:.4f} | {result.strategy}")
        print(f"  DEAP:  {deap_time:8.3f} s | energy = {deap_fitness:.4f}")
        print(f"  Speedup: {speedup:.2f}x")

    # Heat equation benchmarks
    print()
    print("HEAT EQUATION PDE OPTIMIZATION")
    print("=" * 70)
    for gs in [20, 50]:
        print(f"\nGrid size = {gs}")
        print("-" * 50)

        # ParGA
        start = time.perf_counter()
        result = minimize(
            heat_equation_fit,
            genome_length=gs,
            bounds=(-2.0, 2.0),
            population_size=POP_SIZE,
            generations=N_GEN,
            seed=42,
        )
        parga_time = time.perf_counter() - start
        parga_fitness = -result.best_fitness

        # DEAP
        toolbox = setup_deap(heat_equation_fit, gs, -2.0, 2.0)
        start = time.perf_counter()
        deap_fitness = run_deap(toolbox, POP_SIZE, N_GEN)
        deap_time = time.perf_counter() - start

        speedup = deap_time / parga_time if parga_time > 0 else float("inf")

        print(f"  ParGA: {parga_time:8.3f} s | L2 error = {parga_fitness:.4f} | {result.strategy}")
        print(f"  DEAP:  {deap_time:8.3f} s | L2 error = {deap_fitness:.4f}")
        print(f"  Speedup: {speedup:.2f}x")

    print()
    print("=" * 70)
