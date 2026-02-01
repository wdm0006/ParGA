# ParGA

High-performance parallel genetic algorithm library written in Rust with Python bindings.

This is a modern rewrite of an original MATLAB-based parallel genetic algorithm framework (see [legacy/](legacy/) for the original code). The new implementation provides significant performance improvements through Rust's zero-cost abstractions and native parallelism.

[![CI](https://github.com/wdm0006/ParGA/workflows/CI/badge.svg)](https://github.com/wdm0006/ParGA/actions)
[![Crates.io](https://img.shields.io/crates/v/parga.svg)](https://crates.io/crates/parga)
[![PyPI](https://img.shields.io/pypi/v/parga.svg)](https://pypi.org/project/parga/)

## Features

- **Unified Interface**: Single `GA` class that auto-selects the optimal execution strategy
- **Auto-Parallel**: Automatically uses process pools for expensive fitness functions
- **High Performance**: Rust core with parallel fitness evaluation
- **Island Model**: Multiple populations with configurable migration topologies
- **Flexible Operators**: Various selection, crossover, and mutation strategies
- **Python Bindings**: Seamless integration via PyO3
- **Built-in Benchmarks**: Standard optimization test functions included

## Installation

### Python

```bash
pip install parga
```

### Rust

```toml
[dependencies]
parga = "0.1"
```

## Quick Start

### Python

```python
import numpy as np
from parga import minimize, maximize, GA

# Minimize a function - the simplest interface
def sphere(x):
    return np.sum(x ** 2)

result = minimize(sphere, genome_length=10, bounds=(-5, 5), generations=100)
print(f"Minimum value: {-result.best_fitness:.6f}")
print(f"Best solution: {result.best_genes()}")
print(f"Strategy used: {result.strategy}")  # 'rust' or 'parallel'

# Or use GA directly for more control
def fitness(genes):
    return -np.sum(genes ** 2)  # Higher is better

result = GA(fitness, genome_length=10, bounds=(-5, 5)).run()
print(f"Best fitness: {result.best_fitness}")

# Force parallel execution for expensive fitness functions
result = GA(fitness, genome_length=10, parallel=True).run()

# Use island model for complex optimization landscapes
result = GA(fitness, genome_length=10, islands=4).run()
```

The `GA` class automatically selects the optimal execution strategy:
- **Rust backend**: For fast fitness functions (< 0.5ms per evaluation)
- **Parallel process pool**: For expensive fitness functions (bypasses Python GIL)
- **Island model**: When `islands > 1` for better exploration

### Rust

```rust
use parga::prelude::*;

fn main() {
    // Configure the GA
    let config = GaConfig::builder()
        .population_size(100)
        .genome_length(10)
        .generations(100)
        .mutation_rate(0.01)
        .build()
        .unwrap();

    // Use built-in Sphere function
    let fitness = Sphere;

    // Run evolution
    let mut ga: GeneticAlgorithm<RealGenome, _> = GeneticAlgorithm::new(config, fitness);
    let result = ga.run();

    println!("Best fitness: {}", result.best_fitness);
    println!("Best solution: {:?}", result.best_individual.genome.genes());
}
```

## Island Model

The island model maintains multiple populations that evolve independently with periodic migration. Use the unified `GA` interface:

```python
from parga import GA

# Simple island model - just specify the number of islands
result = GA(
    fitness,
    genome_length=20,
    population_size=400,  # Total population distributed across islands
    islands=8,            # 8 islands with 50 individuals each
    generations=200,
    migration_interval=20,
    migration_count=10,
).run()

print(f"Strategy: {result.strategy}")  # 'rust_island' or 'parallel_island'
```

For advanced control over topologies, use the low-level `IslandModel` class:

```python
from parga import IslandModel, MigrationTopology

island_ga = IslandModel(
    fitness_fn=fitness,
    genome_length=20,
    num_islands=8,
    island_population=50,
    generations=200,
    topology=MigrationTopology.ring(),  # or .star(), .ladder(), .fully_connected()
)
result = island_ga.run()
```

### Migration Topologies

- **Ring**: Each island sends migrants to the next island in a ring
- **Star**: Central island exchanges with all others
- **Ladder**: Bidirectional exchange between adjacent islands
- **Fully Connected**: All islands can exchange with all others
- **Random**: Random destination selection

## Operators

### Selection Methods

```python
from parga import SelectionMethod

# Available methods
SelectionMethod.tournament(3)      # Tournament with size 3
SelectionMethod.roulette()         # Fitness-proportionate
SelectionMethod.rank()             # Rank-based
SelectionMethod.truncation(0.5)    # Top 50%
SelectionMethod.stochastic_universal()
```

### Crossover Methods

```python
from parga import CrossoverMethod

CrossoverMethod.single_point()
CrossoverMethod.two_point()
CrossoverMethod.uniform(0.5)
CrossoverMethod.blend(0.5)          # BLX-alpha
CrossoverMethod.simulated_binary(20.0)  # SBX
CrossoverMethod.arithmetic()
```

### Mutation Methods

```python
from parga import MutationMethod

MutationMethod.gaussian(0.1)  # sigma = 0.1 * range
MutationMethod.uniform()
MutationMethod.polynomial(20.0)
MutationMethod.boundary()
```

## API Reference

### Primary Functions

```python
from parga import minimize, maximize, GA

# minimize(fitness_fn, genome_length, bounds=None, **kwargs) -> GAResult
# Convenience function for minimization problems. Automatically negates fitness.
result = minimize(sphere, genome_length=10, bounds=(-5, 5))
actual_minimum = -result.best_fitness  # Negate to get actual minimum

# maximize(fitness_fn, genome_length, bounds=None, **kwargs) -> GAResult
# Convenience function for maximization problems.
result = maximize(fitness_fn, genome_length=10, bounds=(-5, 5))

# GA(fitness_fn, genome_length, **kwargs) -> GA
# Full control over the genetic algorithm configuration.
ga = GA(
    fitness_fn,           # Function: numpy array -> float (higher is better)
    genome_length=10,     # Number of genes
    population_size=100,  # Total population
    generations=100,      # Number of generations
    bounds=(-5, 5),       # (lower, upper) or ([lowers], [uppers])
    parallel=None,        # None=auto, True=force parallel, False=force Rust
    islands=1,            # Number of islands (>1 enables island model)
    n_workers=None,       # Worker processes (default: CPU count)
    mutation_rate=0.01,   # Per-gene mutation probability
    crossover_rate=0.8,   # Crossover probability
    seed=None,            # Random seed for reproducibility
    verbose=False,        # Print strategy selection info
)
result = ga.run()
```

### GAResult

```python
result.best_fitness      # float: Best fitness value found
result.best_genes()      # np.ndarray: Best genome
result.generations       # int: Number of generations run
result.fitness_history   # list[float]: Best fitness per generation
result.strategy          # str: 'rust', 'parallel', 'rust_island', or 'parallel_island'
```

## Benchmark Functions

Built-in test functions for optimization:

```python
from parga import sphere, rastrigin, rosenbrock, ackley, griewank, schwefel
import numpy as np

x = np.array([0.0, 0.0, 0.0])
print(f"Sphere at origin: {sphere(x)}")      # 0.0
print(f"Rastrigin at origin: {rastrigin(x)}")  # 0.0
```

## Performance

ParGA is designed for high performance:

- **Auto-Strategy Selection**: Automatically chooses the fastest approach based on fitness function cost
- **Process Pool Parallelism**: Uses `ProcessPoolExecutor` to bypass Python's GIL for expensive fitness functions
- **Rust Core**: Genetic operators execute in optimized Rust code
- **Efficient Memory Layout**: Compact genome representations with zero-copy NumPy integration
- **Island Parallelism**: Each island evolves independently with configurable migration

### Strategy Selection

ParGA measures your fitness function and selects the optimal strategy:

| Fitness Cost | Islands | Strategy | Description |
|--------------|---------|----------|-------------|
| < 0.5ms | 1 | `rust` | Single-threaded Rust execution |
| ≥ 0.5ms | 1 | `parallel` | Process pool parallelism |
| < 0.5ms | > 1 | `rust_island` | Rust island model |
| ≥ 0.5ms | > 1 | `parallel_island` | Parallel island model |

You can override with `parallel=True` or `parallel=False`:

```python
# Force parallel even for cheap fitness functions
result = GA(fitness, genome_length=10, parallel=True).run()

# Force single-threaded Rust for expensive functions
result = GA(fitness, genome_length=10, parallel=False).run()
```

Typical speedups over pure Python GA libraries: **5-50x** depending on the problem.

## Development

This project uses a Makefile for common tasks. Requires [Rust](https://rustup.rs/) and [uv](https://github.com/astral-sh/uv).

```bash
# Full setup: create venv, install deps, setup pre-commit hooks
make setup

# Build and test everything
make check

# Or run individual commands:
make build          # Build debug
make build-release  # Build release
make test           # Run all tests
make test-rust      # Rust tests only
make test-python    # Python tests only
make bench          # Run benchmarks
make lint           # Run linters (ruff + clippy)
make format         # Format code (ruff + rustfmt)
make dev            # Build Python package for development
make clean          # Clean build artifacts

# See all available commands
make help
```

### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:
- **Rust**: rustfmt, clippy
- **Python**: ruff (lint + format)
- **General**: trailing whitespace, YAML/TOML validation

Install manually: `pre-commit install`

## Legacy MATLAB Code

The original MATLAB implementation is preserved in the [legacy/](legacy/) directory. It includes:

- `GAGlobe.m` - Island model manager (the inspiration for this rewrite)
- `Population.m` - Population breeding and selection
- `Member.m` - Individual genome representation
- `runGA.m` - Main runner script
- Example fitness functions for rocket thrust vectoring and CubeSat attitude control

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
