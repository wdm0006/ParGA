# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "numpy",
#     "parga",
#     "deap==1.4.1",
#     "pymoo==0.6.1.3",
# ]
#
# [tool.uv.sources]
# parga = { path = ".." }
# ///
"""Compare ParGA, DEAP and pymoo on Sphere / Rastrigin / Ackley.

Run from anywhere:  uv run benchmarks/compare_libs.py [--runs 5] [--pop 100] [--gens 100]

`parga` is built from this checkout (needs a Rust toolchain). Output is a
Markdown table on stdout. Not run in CI.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
import warnings

import numpy as np

BOUNDS = (-5.12, 5.12)
CXPB = 0.8
TOURNSIZE = 3
SIGMA = 0.1
ALPHA = 0.5


def sphere(x):
    return float(np.sum(x * x))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def ackley(x):
    n = len(x)
    return float(
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x * x) / n))
        - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
        + 20
        + np.e
    )


PROBLEMS = {"Sphere": sphere, "Rastrigin": rastrigin, "Ackley": ackley}


def run_parga(fn, dim, pop, gens, seed):
    import parga

    res = parga.GA(
        lambda x: -fn(x),
        genome_length=dim,
        population_size=pop,
        generations=gens,
        bounds=BOUNDS,
        parallel=False,
        mutation_rate=1.0 / dim,
        crossover_rate=CXPB,
        elitism=2,
        tournament_size=TOURNSIZE,
        crossover_method=parga.CrossoverMethod.blend(ALPHA),
        mutation_method=parga.MutationMethod.gaussian(SIGMA),
        seed=seed,
    ).run()
    return -res.best_fitness


def run_deap(fn, dim, pop, gens, seed):
    from deap import algorithms, base, creator, tools

    if not hasattr(creator, "FitMinCmp"):
        creator.create("FitMinCmp", base.Fitness, weights=(-1.0,))
        creator.create("IndCmp", list, fitness=creator.FitMinCmp)
    random.seed(seed)
    np.random.seed(seed)
    tb = base.Toolbox()
    tb.register("attr", random.uniform, *BOUNDS)
    tb.register("individual", tools.initRepeat, creator.IndCmp, tb.attr, n=dim)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate", lambda ind: (fn(np.asarray(ind)),))
    tb.register("mate", tools.cxBlend, alpha=ALPHA)
    tb.register("mutate", tools.mutGaussian, mu=0.0, sigma=SIGMA, indpb=1.0 / dim)
    tb.register("select", tools.selTournament, tournsize=TOURNSIZE)
    hof = tools.HallOfFame(1)
    algorithms.eaSimple(
        tb.population(n=pop), tb, cxpb=CXPB, mutpb=1.0, ngen=gens, halloffame=hof, verbose=False
    )
    return hof[0].fitness.values[0]


def run_pymoo(fn, dim, pop, gens, seed):
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.optimize import minimize

    class P(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=dim, n_obj=1, xl=BOUNDS[0], xu=BOUNDS[1])

        def _evaluate(self, x, out, *args, **kwargs):
            out["F"] = fn(x)

    algo = GA(
        pop_size=pop,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=CXPB, eta=15),
        mutation=PM(prob=1.0, prob_var=1.0 / dim, eta=20),
        eliminate_duplicates=True,
    )
    res = minimize(P(), algo, ("n_gen", gens), seed=seed, verbose=False)
    return float(res.F[0])


LIBS = {"ParGA": run_parga, "DEAP": run_deap, "pymoo": run_pymoo}


def bench(runner, fn, dim, pop, gens, runs):
    runner(fn, dim, 10, 2, 12345)  # untimed warm-up: imports, lazy init
    times, fits = [], []
    for seed in range(runs):
        t0 = time.perf_counter()
        fits.append(runner(fn, dim, pop, gens, seed))
        times.append(time.perf_counter() - t0)
    return times, fits


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=5, help="seeds/repeats per cell (>=5)")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--dims", type=int, nargs="+", default=[10, 50])
    a = ap.parse_args()
    if a.runs < 5:
        ap.error("--runs must be >= 5")
    warnings.filterwarnings("ignore")

    from importlib.metadata import version

    seeds = f"0..{a.runs - 1}"
    print("# ParGA vs DEAP vs pymoo\n")
    print("## Protocol\n")
    print(f"- Population {a.pop}, generations {a.gens}, bounds {BOUNDS}, seeds {seeds} (one run per seed).")
    print("- Every library minimizes the same NumPy objective, called once per individual, single process.")
    print(f"- Versions: parga {version('parga')} (built from this checkout), deap {version('deap')}, "
          f"pymoo {version('pymoo')}, numpy {version('numpy')}.")
    print("- Time: median [min, max] seconds of wall time per full run (includes setup; one untimed warm-up run first).")
    print("- Fitness: median [min, max] of the final best objective value (lower is better).\n")
    print("## Operators (NOT identical across libraries)\n")
    print(f"- ParGA `GA(parallel=False)`: tournament(size {TOURNSIZE}) selection, BLX-alpha blend({ALPHA}) crossover "
          f"(rate {CXPB}), Gaussian(sigma {SIGMA}) mutation, per-gene rate 1/dim, elitism 2, bound repair after crossover.")
    print(f"- DEAP `eaSimple`: `selTournament`(size {TOURNSIZE}), `cxBlend`(alpha {ALPHA}, cxpb {CXPB}), "
          f"`mutGaussian`(sigma {SIGMA}, indpb 1/dim, mutpb 1.0), no elitism, no bound repair (genes may leave bounds).")
    print(f"- pymoo `GA`: default binary tournament, SBX(eta 15, prob {CXPB}), polynomial mutation (eta 20, "
          "per-gene prob 1/dim), (mu+lambda) elitist survival, duplicate elimination, bounds enforced.")
    print("- `ngen`/`n_gen` count differs by one initial-population evaluation between libraries; "
          "fitness-evaluation budgets are therefore close but not identical.\n")
    print("| Problem | Dim | Library | Time s (median [min, max]) | Best fitness (median [min, max]) | Protocol |")
    print("|---|---|---|---|---|---|")
    proto = f"pop {a.pop}, gens {a.gens}, {a.runs} seeds"
    for name, fn in PROBLEMS.items():
        for dim in a.dims:
            for lib, runner in LIBS.items():
                t, f = bench(runner, fn, dim, a.pop, a.gens, a.runs)
                print(
                    f"| {name} | {dim} | {lib} | "
                    f"{statistics.median(t):.3f} [{min(t):.3f}, {max(t):.3f}] | "
                    f"{statistics.median(f):.3e} [{min(f):.3e}, {max(f):.3e}] | {proto} |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
