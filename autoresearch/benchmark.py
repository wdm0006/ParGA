#!/usr/bin/env python3
"""Autoresearch benchmark harness for ParGA.

DO NOT MODIFY THIS FILE. This is the immutable evaluator.

Measures ParGA framework quality across three dimensions:
  1. Accuracy  — how close the GA gets to known optima
  2. Speed     — wall-clock time per run
  3. Memory    — peak RSS during each run

Runs the current train.py configuration on a fixed benchmark suite,
averages over multiple trials, and prints a single composite score.

Lower score = better. Score of 0.0 would be perfect (instant, zero-error).
"""

import importlib
import importlib.util
import json
import os
import resource
import sys
import time
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Benchmark suite — each entry defines a problem the GA must solve
# ---------------------------------------------------------------------------
BENCHMARKS = [
    {
        "name": "Sphere-30D",
        "fn": lambda x: -np.sum(x**2),
        "dims": 30,
        "bounds": (-5.12, 5.12),
        "known_optimum": 0.0,
    },
    {
        "name": "Rastrigin-30D",
        "fn": lambda x: -(10.0 * len(x) + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x))),
        "dims": 30,
        "bounds": (-5.12, 5.12),
        "known_optimum": 0.0,
    },
    {
        "name": "Rosenbrock-30D",
        "fn": lambda x: -np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2),
        "dims": 30,
        "bounds": (-5.0, 10.0),
        "known_optimum": 0.0,
    },
    {
        "name": "Ackley-30D",
        "fn": lambda x: -(
            -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / len(x)))
            - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / len(x))
            + 20.0
            + np.e
        ),
        "dims": 30,
        "bounds": (-32.768, 32.768),
        "known_optimum": 0.0,
    },
]

N_TRIALS = 3  # Average over this many seeded trials
SEEDS = [42, 137, 2024]


def get_peak_rss_mb():
    """Return peak RSS in MB (macOS/Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)  # bytes -> MB on macOS
    return usage.ru_maxrss / 1024  # KB -> MB on Linux


def run_benchmark():
    # Import train.py dynamically so changes are always picked up
    spec = importlib.util.spec_from_file_location(
        "train", os.path.join(os.path.dirname(__file__), "train.py")
    )
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    results = []

    for bench in BENCHMARKS:
        trial_errors = []
        trial_times = []
        trial_peak_rss = []

        for seed in SEEDS:
            rss_before = get_peak_rss_mb()
            t0 = time.perf_counter()

            try:
                result = train.run(
                    fitness_fn=bench["fn"],
                    dims=bench["dims"],
                    bounds=bench["bounds"],
                    seed=seed,
                )
                elapsed = time.perf_counter() - t0
                rss_after = get_peak_rss_mb()

                # Error = distance from known optimum (fitness is negated, so
                # best_fitness of 0.0 means we hit the optimum)
                error = abs(result.best_fitness - bench["known_optimum"])

                trial_errors.append(error)
                trial_times.append(elapsed)
                trial_peak_rss.append(max(0, rss_after - rss_before))

            except Exception:
                traceback.print_exc()
                trial_errors.append(1e6)
                trial_times.append(60.0)
                trial_peak_rss.append(0.0)

        bench_result = {
            "name": bench["name"],
            "mean_error": float(np.mean(trial_errors)),
            "mean_time_s": float(np.mean(trial_times)),
            "mean_peak_rss_mb": float(np.mean(trial_peak_rss)),
        }
        results.append(bench_result)

    # -----------------------------------------------------------------------
    # Composite score
    #
    # We combine accuracy, speed, and memory into a single number.
    #   accuracy_score = mean of log10(error + 1e-10)  (log-scale, lower = better)
    #   speed_score    = total wall-clock seconds across benchmarks
    #   memory_score   = max peak RSS delta across benchmarks (MB)
    #
    # Composite = accuracy_score + 2*speed_score + 0.1*memory_score
    # The weights reflect our priorities: accuracy matters most, speed second,
    # memory is a tiebreaker.
    # -----------------------------------------------------------------------
    accuracy_score = float(np.mean([np.log10(r["mean_error"] + 1e-10) for r in results]))
    speed_score = float(np.sum([r["mean_time_s"] for r in results]))
    memory_score = float(np.max([r["mean_peak_rss_mb"] for r in results]))

    composite = accuracy_score + 2.0 * speed_score + 0.1 * memory_score

    # Print detailed results
    print("=" * 70)
    print("AUTORESEARCH BENCHMARK RESULTS")
    print("=" * 70)
    for r in results:
        print(
            f"  {r['name']:20s}  "
            f"error={r['mean_error']:12.6f}  "
            f"time={r['mean_time_s']:6.3f}s  "
            f"rss={r['mean_peak_rss_mb']:6.1f}MB"
        )
    print("-" * 70)
    print(f"  Accuracy score (log):  {accuracy_score:+.4f}")
    print(f"  Speed score (total s): {speed_score:.4f}")
    print(f"  Memory score (max MB): {memory_score:.1f}")
    print(f"  COMPOSITE SCORE:       {composite:.4f}  (lower is better)")
    print("=" * 70)

    # Write machine-readable log
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "composite": composite,
        "accuracy_score": accuracy_score,
        "speed_score": speed_score,
        "memory_score": memory_score,
        "benchmarks": results,
    }

    log_path = os.path.join(os.path.dirname(__file__), "results.jsonl")
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return composite


if __name__ == "__main__":
    score = run_benchmark()
    sys.exit(0)
