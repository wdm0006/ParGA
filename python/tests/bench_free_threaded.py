"""Benchmark: free-threaded rayon vs ProcessPoolExecutor vs single-threaded Rust.

Run with both venvs to compare:
  .venv/bin/python python/tests/bench_free_threaded.py      # GIL Python
  .venv-ft/bin/python python/tests/bench_free_threaded.py   # Free-threaded
"""

import math
import time
import warnings

import numpy as np

from parga import GA, is_free_threaded
from parga._parga import GeneticAlgorithm as RustGA


def cheap_fitness(genes):
    """~5us: NumPy sum of squares."""
    return -np.sum(genes**2)


def pure_python_fitness(genes):
    """~2ms: pure-Python computation (no NumPy contention)."""
    total = 0.0
    for i in range(len(genes)):
        x = genes[i]
        for _ in range(200):
            total += math.sin(x) * math.cos(x)
    return -sum(float(g) ** 2 for g in genes) + total * 1e-15


def numpy_fitness(genes):
    """~0.5ms: NumPy-heavy (contends under free-threading)."""
    total = 0.0
    for _ in range(50):
        total += np.sum(np.sin(genes) * np.cos(genes))
    return -np.sum(genes**2) + total * 1e-10


def sleep_fitness(genes):
    """~2ms: simulated I/O-bound work."""
    time.sleep(0.002)
    return -sum(float(g) ** 2 for g in genes)


def bench(label, fn, n_runs=3):
    """Run fn n_runs times and report median."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    median = sorted(times)[len(times) // 2]
    print(f"  {label:40s} {median:.3f}s")
    return median


def main():
    print(f"Free-threaded: {is_free_threaded()}")
    print(f"NumPy: {np.__version__}")
    print()

    pop_size = 200
    generations = 30
    genome_length = 20

    for fitness_fn, label in [
        (cheap_fitness, "cheap (~5us, NumPy)"),
        (pure_python_fitness, "pure-Python (~2ms)"),
        (numpy_fitness, "NumPy-heavy (~0.5ms)"),
        (sleep_fitness, "sleep-based (~2ms)"),
    ]:
        print(f"--- {label} fitness, pop={pop_size}, gen={generations} ---")

        # Single-threaded Rust
        def run_rust(ff=fitness_fn):
            ga = RustGA(
                fitness_fn=ff,
                genome_length=genome_length,
                population_size=pop_size,
                generations=generations,
                lower_bounds=[-5.0] * genome_length,
                upper_bounds=[5.0] * genome_length,
                seed=42,
            )
            return ga.run()

        t_rust = bench("Rust single-thread", run_rust)

        # GA auto-select (rayon on FT, rust on GIL for cheap)
        def run_ga_auto(ff=fitness_fn):
            ga = GA(
                ff,
                genome_length=genome_length,
                population_size=pop_size,
                generations=generations,
                bounds=(-5.0, 5.0),
                seed=42,
            )
            return ga.run()

        t_auto = bench("GA auto-select", run_ga_auto)

        if t_rust > 0:
            ratio = t_rust / t_auto
            if ratio > 1.05:
                print(f"  -> {ratio:.1f}x speedup from parallelism")
            else:
                print(f"  -> ~same (ratio={ratio:.2f})")

        # ProcessPoolExecutor (parallel=True forces it on GIL Python)
        if not is_free_threaded():
            def run_parallel(ff=fitness_fn):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    ga = GA(
                        ff,
                        genome_length=genome_length,
                        population_size=pop_size,
                        generations=generations,
                        bounds=(-5.0, 5.0),
                        parallel=True,
                        seed=42,
                    )
                    return ga.run()

            t_parallel = bench("ProcessPoolExecutor", run_parallel)

        print()


if __name__ == "__main__":
    main()
