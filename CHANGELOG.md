v0.2.0
======

The first published release. ParGA has been a working library since the Rust rewrite, but it has
never been on PyPI or crates.io. This release is the point where a seeded run reproduces exactly,
the genetic operators do what their papers say, and a failing objective function reports itself
instead of being scored as an unusually hard problem.

Reproducibility

 * A seeded RNG is threaded through the whole algorithm: initial population, selection, crossover,
   mutation, restart-on-stagnation, random immigrants and local search all descend from one seed.
 * Island runs are reproducible. Each island derives its seed structurally from the master seed
   rather than sharing a generator, so parallel evolution repeats exactly.
 * Edge recombination draws from the seeded stream like every other operator.

Free-threaded Python

 * CI builds and tests against free-threaded Python 3.13t, and asserts the interpreter really is
   free-threaded rather than silently falling back to the GIL build.

Genetic operators

 * Deb's polynomial mutation uses the distance to its own boundary in the upper branch, so the
   operator is symmetric about the midpoint instead of biased toward the lower bound.
 * PMX crossover no longer inverts its mapping, which could leave the repair loop spinning forever.
 * Stochastic universal sampling walks the population at the correct stride.
 * Non-uniform mutation annealing decays as specified.
 * Real-valued offspring are clamped to their bounds after crossover, as an invariant rather than an
   afterthought.
 * Truncation selection honors the requested ratio.
 * Local search only runs on genomes that support it, and cannot corrupt the ones that do not.

Island model

 * Migration replaces the intended individuals.
 * Python island migration carries the migrant's fitness across instead of dropping it.

Errors and validation

 * An exception raised by a Python fitness callback is surfaced. It used to be scored as negative
   infinity, which turned a typo in an objective function into a very hard optimization problem.
 * Non-finite fitness values from Python are rejected, and NaN fitness no longer panics roulette or
   SUS selection.
 * GA configurations are validated before execution, and the facade warns when a chosen execution
   strategy cannot honor a backend-specific option instead of accepting and ignoring it.
 * Python operator factory parameters are validated at construction.
 * Strided NumPy arrays are accepted at the boundary.

Parallelism

 * Parallel fitness functions are serialized by value, so a process pool receives what it needs.
 * `early_stopping` and `mutation_rate_end` are honored on the process-pool paths, not just the
   Rust-native ones.

Other

 * `fitness_history` keeps exactly one entry per generation, so a restart no longer corrupts a
   convergence plot.
 * An installable `viz` extra (`pip install parga[viz]`) for `parga.viz`, keeping matplotlib out of
   the core install.
 * pyo3 and numpy updated to 0.24.
