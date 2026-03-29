# ParGA Autoresearch Program

## Goal

Improve ParGA as a **general-purpose genetic algorithm framework**. We are NOT tuning hyperparameters for specific benchmarks. We are making the framework itself faster, more capable, and more efficient.

## The Loop

1. Read this file and `train.py` to understand the current state.
2. Read the latest entry in `results.jsonl` to see the current best composite score.
3. Form a hypothesis about a framework improvement.
4. Implement the improvement in the ParGA source code (`src/*.rs`, `python/parga/*.py`).
5. Update `train.py` only if needed to use the new feature.
6. Rebuild: `.venv/bin/maturin develop --release`
7. Run: `.venv/bin/python autoresearch/benchmark.py`
8. If the composite score improved (lower is better), keep the changes. If not, revert all changes.
9. Go to step 1.

## What to Improve

Focus on changes that make ParGA better as a **general framework**, not just better at these specific benchmarks:

### Rust Core (`src/*.rs`)
- Faster fitness batch evaluation
- More efficient population data structures
- Better memory layout / cache locality
- SIMD or vectorized operations for genetic operators
- Smarter selection algorithms
- New crossover/mutation operators that are generally useful
- Reduced allocation / copying in the hot loop
- Better parallelism in the island model

### Python Layer (`python/parga/*.py`)
- Reduced Python<->Rust overhead
- More efficient numpy usage in the bridge
- Better GIL management patterns
- Smarter strategy selection

### Algorithmic
- Adaptive operator rates (general-purpose, not benchmark-specific)
- Better diversity maintenance mechanisms
- Improved elite management
- Restart strategies for stagnation detection
- More efficient migration in island model

## What NOT to Do

- Do NOT tune hyperparameters in `train.py` for specific benchmarks. The `train.py` config should use reasonable defaults that work across problems.
- Do NOT modify `benchmark.py`. It is immutable.
- Do NOT add benchmark-specific logic or special-casing.
- Do NOT break the existing public API. Changes should be backwards-compatible.
- Do NOT add heavy dependencies.

## The Metric

The benchmark harness (`benchmark.py`) computes a composite score:

```
composite = accuracy_score + 2*speed_score + 0.1*memory_score
```

- **accuracy_score**: mean of log10(error) across 4 benchmark functions (30D each), averaged over 3 seeded trials. Lower = better.
- **speed_score**: total wall-clock seconds across all benchmarks. Lower = better. Weighted 2x because framework speed matters.
- **memory_score**: peak RSS delta in MB. Lower = better. Weighted 0.1x as a tiebreaker.

A framework improvement should make the composite score go down without breaking anything.

## Build & Test

```bash
# Rebuild after Rust changes
.venv/bin/maturin develop --release

# Run existing tests (must still pass)
cargo test
.venv/bin/pytest python/tests/test_parga.py

# Run benchmark
.venv/bin/python autoresearch/benchmark.py
```

## Architecture Quick Reference

- `src/lib.rs` — `GeneticAlgorithm` main loop, `evaluate_population()`
- `src/fitness.rs` — `FitnessFunction` trait with `evaluate()` and `evaluate_batch()`
- `src/population.rs` — Population struct, `partition_elite()` using `select_nth_unstable_by`
- `src/island.rs` — `IslandModel` with per-island batch evaluation
- `src/operators/` — Selection, crossover, mutation implementations
- `src/python/mod.rs` — PyO3 bindings, `PyFitness` with batch GIL optimization
- `python/parga/ga.py` — `GA` class with auto strategy selection
- `python/parga/parallel.py` — `ProcessPoolExecutor` path with initializer pattern
