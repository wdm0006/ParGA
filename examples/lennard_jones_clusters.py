"""Lennard-Jones Cluster Optimization Case Study.

This example demonstrates using ParGA to find minimum-energy configurations
of atomic clusters interacting via the Lennard-Jones potential - a classic
benchmark problem in computational physics and chemistry.

The Lennard-Jones potential describes interactions between neutral atoms:
    V(r) = 4*eps*[(sigma/r)^12 - (sigma/r)^6]

where:
    - r is the distance between two atoms
    - eps is the depth of the potential well (we use eps=1)
    - sigma is the distance at which potential is zero (we use sigma=1)

Finding the global minimum energy configuration is NP-hard because the
number of local minima grows exponentially with the number of atoms.
For N=13 there are >1500 distinct local minima.

Known global minima (reduced units, eps=1):
    N=2:  -1.000     N=7:  -16.505    N=12: -37.967
    N=3:  -3.000     N=8:  -19.822    N=13: -44.327 (icosahedron)
    N=4:  -6.000     N=9:  -24.113
    N=5:  -9.104     N=10: -28.422
    N=6:  -12.712    N=11: -32.765

This implementation uses:
    - Vectorised pair-distance computation
    - Island-model GA for N>=7 to escape local minima
    - Scaled population/generation counts for harder instances
    - Multiple independent restarts for N>=10

References:
    - Wales, D.J. & Doye, J.P.K. (1997) "Global Optimization by Basin-Hopping"
    - Cambridge Cluster Database: http://www-wales.ch.cam.ac.uk/CCD.html
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from parga import minimize

# Try to import visualization - optional dependency
try:
    from parga.viz import plot_3d_cluster, plot_convergence, plot_convergence_comparison

    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# Known global minima for comparison (Cambridge Cluster Database)
KNOWN_MINIMA = {
    2: -1.0,
    3: -3.0,
    4: -6.0,
    5: -9.103852,
    6: -12.712062,
    7: -16.505384,
    8: -19.821489,
    9: -24.113360,
    10: -28.422532,
    11: -32.765970,
    12: -37.967600,
    13: -44.326801,  # Famous icosahedron
}


def lennard_jones_energy(positions: np.ndarray, n_atoms: int) -> float:
    """Vectorised LJ energy calculation.

    Args:
        positions: Flattened array of shape (3*n_atoms,) containing
                  [x1, y1, z1, x2, y2, z2, ...] coordinates
        n_atoms: Number of atoms in the cluster

    Returns:
        Total potential energy in reduced units (eps=1, sigma=1)
    """
    coords = positions.reshape(n_atoms, 3)

    # Compute all pairwise distance-squared values
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)

    # Upper triangle (i < j pairs only)
    upper = np.triu_indices(n_atoms, k=1)
    r2_pairs = r2[upper]

    # Prevent division by zero (collapsed atoms)
    r2_pairs = np.maximum(r2_pairs, 1e-10)

    # LJ potential: 4 * [(1/r^12) - (1/r^6)]
    r6_inv = 1.0 / (r2_pairs ** 3)
    energy = np.sum(4.0 * (r6_inv ** 2 - r6_inv))

    return energy


# ---------------------------------------------------------------------------
# GA parameters scaled by problem difficulty
# ---------------------------------------------------------------------------

def _ga_params(n_atoms: int) -> dict:
    """Return GA parameters tuned for a given cluster size.

    Larger clusters have exponentially more local minima, so we
    increase population, generations, island count, and mutation
    rate accordingly.
    """
    if n_atoms <= 4:
        return dict(population_size=120, generations=300, islands=1,
                    mutation_rate=0.05, restarts=1)
    elif n_atoms <= 6:
        return dict(population_size=200, generations=500, islands=4,
                    mutation_rate=0.08, restarts=2)
    elif n_atoms <= 8:
        return dict(population_size=300, generations=800, islands=4,
                    mutation_rate=0.10, restarts=3)
    elif n_atoms <= 10:
        return dict(population_size=400, generations=1000, islands=6,
                    mutation_rate=0.12, restarts=5)
    else:  # N=11-13
        return dict(population_size=500, generations=1500, islands=6,
                    mutation_rate=0.15, restarts=8)


def optimize_cluster(
    n_atoms: int,
    population_size: int | None = None,
    generations: int | None = None,
    islands: int | None = None,
    mutation_rate: float | None = None,
    restarts: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Optimise a Lennard-Jones cluster configuration.

    For larger clusters this runs multiple independent restarts and
    returns the best result.
    """
    params = _ga_params(n_atoms)
    population_size = population_size or params["population_size"]
    generations = generations or params["generations"]
    islands = islands or params["islands"]
    mutation_rate = mutation_rate or params["mutation_rate"]
    restarts = restarts or params["restarts"]

    genome_length = 3 * n_atoms
    # Search bounds: optimal LJ clusters have radius ~ n^(1/3) * 0.6 sigma,
    # so we use a tight box that covers the cluster with modest margin.
    # Tighter bounds greatly improve convergence by keeping atoms compact.
    bound = max(1.5, 0.9 * (n_atoms ** (1 / 3)))

    def fitness(positions):
        return lennard_jones_energy(positions, n_atoms)

    if verbose:
        print(f"\nOptimising LJ cluster with {n_atoms} atoms")
        print(f"  Genome length: {genome_length}")
        print(f"  Search bounds: [-{bound:.2f}, {bound:.2f}]")
        print(f"  Population: {population_size}, Generations: {generations}")
        if islands > 1:
            print(f"  Island model: {islands} islands")
        if restarts > 1:
            print(f"  Independent restarts: {restarts}")

    best_energy = np.inf
    best_positions = None
    best_history = None
    best_strategy = None
    total_time = 0.0

    for r in range(restarts):
        run_seed = seed + r if seed is not None else None
        t0 = time.perf_counter()

        result = minimize(
            fitness,
            genome_length=genome_length,
            bounds=(-bound, bound),
            population_size=population_size,
            generations=generations,
            islands=islands,
            mutation_rate=mutation_rate,
            crossover_rate=0.8,
            seed=run_seed,
            verbose=False,
        )

        elapsed = time.perf_counter() - t0
        total_time += elapsed
        energy = -result.best_fitness  # minimize() negates

        if verbose and restarts > 1:
            print(f"    Restart {r+1}/{restarts}: E = {energy:.6f}  ({elapsed:.1f}s)")

        if energy < best_energy:
            best_energy = energy
            best_positions = result.best_genes()
            best_history = [-f for f in result.fitness_history]
            best_strategy = result.strategy

    if verbose:
        print(f"  Strategy: {best_strategy}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Best energy: {best_energy:.6f}")

    return {
        "n_atoms": n_atoms,
        "best_positions": best_positions,
        "best_energy": best_energy,
        "fitness_history": best_history,
        "strategy": best_strategy,
        "elapsed": total_time,
    }


def run_benchmark(output_dir: Path | None = None):
    """Run benchmark on cluster sizes from N=3 to N=13."""
    print("=" * 70)
    print("Lennard-Jones Cluster Optimization Benchmark")
    print("=" * 70)
    print("\nComparing ParGA results against known global minima.")
    print("Note: This is a challenging global optimization problem!")
    print("For N>=10 the landscape has thousands of local minima.")
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_histories = {}

    for n_atoms in [3, 4, 5, 6, 7, 8, 10, 13]:
        print(f"\n{'=' * 50}")
        print(f"N = {n_atoms} atoms")
        print("=" * 50)

        result = optimize_cluster(n_atoms=n_atoms, seed=42, verbose=True)

        known = KNOWN_MINIMA[n_atoms]
        error = abs(result["best_energy"] - known)
        pct_error = 100 * error / abs(known)

        results.append((n_atoms, result["best_energy"], known, pct_error, result["elapsed"]))
        all_histories[f"N={n_atoms}"] = result["fitness_history"]

        print(f"  Known minimum: {known:.6f}")
        print(f"  Error: {error:.6f} ({pct_error:.2f}%)")

        # Save cluster visualisation
        if HAS_VIZ and output_dir:
            plot_3d_cluster(
                result["best_positions"],
                n_atoms,
                title=f"LJ{n_atoms} Cluster (E = {result['best_energy']:.4f})",
                save_path=output_dir / f"lj_cluster_{n_atoms}.png",
                show=False,
            )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'N':>4} {'Found':>12} {'Known':>12} {'Error %':>10} {'Time (s)':>10}")
    print("-" * 50)
    for n, found, known, pct, elapsed in results:
        marker = " *" if pct < 1.0 else ""
        print(f"{n:>4} {found:>12.6f} {known:>12.6f} {pct:>9.2f}% {elapsed:>10.1f}{marker}")
    print("\n  * = within 1% of global minimum")

    # Convergence comparison plot
    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            all_histories,
            title="LJ Cluster Optimization Convergence",
            ylabel="Energy (reduced units)",
            save_path=output_dir / "lj_convergence_comparison.png",
            show=False,
        )
        print(f"\nPlots saved to {output_dir}/")

    return results


def compare_strategies(output_dir: Path | None = None):
    """Compare single GA vs island model for LJ13 (hardest case)."""
    n_atoms = 13

    print("=" * 70)
    print(f"Strategy Comparison: {n_atoms}-atom LJ Cluster (Icosahedron)")
    print("=" * 70)
    print("\nComparing single population vs island model...")
    print("LJ13 is famous because its global minimum is a perfect icosahedron.")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Single population - 3 restarts
    print("\n--- Single Population GA (3 restarts) ---")
    result_single = optimize_cluster(
        n_atoms=n_atoms,
        population_size=300,
        generations=1000,
        islands=1,
        restarts=3,
        seed=42,
        verbose=True,
    )

    # Island model - 3 restarts
    print("\n--- Island Model, 6 islands (3 restarts) ---")
    result_island = optimize_cluster(
        n_atoms=n_atoms,
        population_size=300,
        generations=1000,
        islands=6,
        restarts=3,
        seed=42,
        verbose=True,
    )

    known = KNOWN_MINIMA[n_atoms]
    print(f"\n--- Results (known minimum: {known:.6f}) ---")
    print(f"Single GA:     {result_single['best_energy']:.6f} (error: {abs(result_single['best_energy'] - known):.4f})")
    print(f"Island Model:  {result_island['best_energy']:.6f} (error: {abs(result_island['best_energy'] - known):.4f})")
    print(f"Time - Single: {result_single['elapsed']:.1f}s, Island: {result_island['elapsed']:.1f}s")

    # Comparison plot
    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            {
                "Single GA": result_single["fitness_history"],
                "Island Model (6)": result_island["fitness_history"],
            },
            title=f"LJ{n_atoms}: Single GA vs Island Model",
            ylabel="Energy (reduced units)",
            save_path=output_dir / "lj_strategy_comparison.png",
            show=False,
        )

        best_result = (
            result_island
            if result_island["best_energy"] < result_single["best_energy"]
            else result_single
        )
        plot_3d_cluster(
            best_result["best_positions"],
            n_atoms,
            title=f"Best LJ{n_atoms} Configuration (E = {best_result['best_energy']:.4f})",
            save_path=output_dir / f"lj{n_atoms}_best_cluster.png",
            show=False,
        )
        print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    import sys

    output_dir = Path(__file__).parent / "output"

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_strategies(output_dir)
    else:
        run_benchmark(output_dir)
