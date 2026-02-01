"""Lennard-Jones Cluster Optimization Case Study.

This example demonstrates using ParGA to find minimum-energy configurations
of atomic clusters interacting via the Lennard-Jones potential - a classic
benchmark problem in computational physics and chemistry.

The Lennard-Jones potential describes interactions between neutral atoms:
    V(r) = 4ε[(σ/r)^12 - (σ/r)^6]

where:
    - r is the distance between two atoms
    - ε is the depth of the potential well (we use ε=1)
    - σ is the distance at which potential is zero (we use σ=1)

Finding the global minimum energy configuration is NP-hard because the
number of local minima grows exponentially with the number of atoms.

Known global minima (reduced units, ε=1):
    N=2:  -1.0
    N=3:  -3.0
    N=4:  -6.0
    N=5:  -9.104
    N=6:  -12.712
    N=7:  -16.505
    N=8:  -19.822
    N=9:  -24.113
    N=10: -28.422
    N=13: -44.327 (icosahedron)

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


# Known global minima for comparison
KNOWN_MINIMA = {
    2: -1.0,
    3: -3.0,
    4: -6.0,
    5: -9.104,
    6: -12.712,
    7: -16.505,
    8: -19.822,
    9: -24.113,
    10: -28.422,
    13: -44.327,  # Famous icosahedron
}


def lennard_jones_energy(positions: np.ndarray, n_atoms: int) -> float:
    """Calculate total Lennard-Jones potential energy of a cluster.

    Args:
        positions: Flattened array of shape (3*n_atoms,) containing
                  [x1, y1, z1, x2, y2, z2, ...] coordinates
        n_atoms: Number of atoms in the cluster

    Returns:
        Total potential energy in reduced units (ε=1, σ=1)
    """
    # Reshape to (n_atoms, 3)
    coords = positions.reshape(n_atoms, 3)

    energy = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Distance between atoms i and j
            r2 = np.sum((coords[i] - coords[j]) ** 2)
            if r2 < 1e-10:  # Prevent division by zero
                return 1e10

            # Lennard-Jones potential: 4[(1/r)^12 - (1/r)^6]
            r6 = r2**3
            r12 = r6**2
            energy += 4.0 * (1.0 / r12 - 1.0 / r6)

    return energy


def lennard_jones_energy_vectorized(positions: np.ndarray, n_atoms: int) -> float:
    """Vectorized LJ energy calculation (faster for larger clusters).

    Args:
        positions: Flattened array of shape (3*n_atoms,)
        n_atoms: Number of atoms in the cluster

    Returns:
        Total potential energy
    """
    coords = positions.reshape(n_atoms, 3)

    # Compute all pairwise distances
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)

    # Get upper triangle (i < j pairs only)
    upper_indices = np.triu_indices(n_atoms, k=1)
    r2_pairs = r2[upper_indices]

    # Handle zero distances
    r2_pairs = np.maximum(r2_pairs, 1e-10)

    # Lennard-Jones potential
    r6 = r2_pairs**3
    r12 = r6**2
    energy = np.sum(4.0 * (1.0 / r12 - 1.0 / r6))

    return energy


def optimize_cluster(
    n_atoms: int,
    population_size: int = 100,
    generations: int = 200,
    islands: int = 1,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Optimize a Lennard-Jones cluster configuration.

    Args:
        n_atoms: Number of atoms in the cluster
        population_size: GA population size
        generations: Number of generations
        islands: Number of islands (use >1 for island model)
        seed: Random seed for reproducibility
        verbose: Print progress information

    Returns:
        Dictionary with optimization results
    """
    genome_length = 3 * n_atoms  # x, y, z for each atom

    # Create fitness function (closure over n_atoms)
    def fitness(positions):
        return lennard_jones_energy_vectorized(positions, n_atoms)

    # Initial positions typically within a small cube
    # Scale bounds based on expected cluster size
    bound = 2.0 * (n_atoms ** (1 / 3))

    if verbose:
        print(f"\nOptimizing LJ cluster with {n_atoms} atoms")
        print(f"  Genome length: {genome_length}")
        print(f"  Search bounds: [-{bound:.1f}, {bound:.1f}]")
        print(f"  Population: {population_size}, Generations: {generations}")
        if islands > 1:
            print(f"  Using island model with {islands} islands")

    start_time = time.perf_counter()

    result = minimize(
        fitness,
        genome_length=genome_length,
        bounds=(-bound, bound),
        population_size=population_size,
        generations=generations,
        islands=islands,
        mutation_rate=0.05,  # Higher mutation for this problem
        crossover_rate=0.8,
        seed=seed,
        verbose=False,
    )

    elapsed = time.perf_counter() - start_time
    best_energy = -result.best_fitness  # minimize() negates

    if verbose:
        print(f"  Strategy: {result.strategy}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Best energy: {best_energy:.6f}")

    return {
        "n_atoms": n_atoms,
        "best_positions": result.best_genes(),
        "best_energy": best_energy,
        "fitness_history": [-f for f in result.fitness_history],  # Convert to energy
        "strategy": result.strategy,
        "elapsed": elapsed,
    }


def run_benchmark(output_dir: Path | None = None):
    """Run benchmark on various cluster sizes."""
    print("=" * 70)
    print("Lennard-Jones Cluster Optimization Benchmark")
    print("=" * 70)
    print("\nComparing ParGA results against known global minima.")
    print("Note: This is a challenging global optimization problem!")
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_histories = {}

    for n_atoms in [3, 4, 5, 6, 7]:
        # Scale parameters with problem size
        pop_size = 100 + 20 * n_atoms
        generations = 200 + 50 * n_atoms

        result = optimize_cluster(
            n_atoms=n_atoms,
            population_size=pop_size,
            generations=generations,
            seed=42,
            verbose=True,
        )

        known = KNOWN_MINIMA[n_atoms]
        error = abs(result["best_energy"] - known)
        pct_error = 100 * error / abs(known)

        results.append((n_atoms, result["best_energy"], known, pct_error))
        all_histories[f"N={n_atoms}"] = result["fitness_history"]

        print(f"  Known minimum: {known:.6f}")
        print(f"  Error: {error:.6f} ({pct_error:.2f}%)")

        # Save cluster visualization
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
    print(f"{'N atoms':>8} {'Found':>12} {'Known':>12} {'Error %':>10}")
    print("-" * 44)
    for n, found, known, pct in results:
        print(f"{n:>8} {found:>12.6f} {known:>12.6f} {pct:>10.2f}%")

    # Convergence comparison plot
    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            all_histories,
            title="LJ Cluster Optimization Convergence",
            ylabel="Energy (ε)",
            save_path=output_dir / "lj_convergence_comparison.png",
            show=False,
        )
        print(f"\nPlots saved to {output_dir}/")

    return results


def compare_strategies(output_dir: Path | None = None):
    """Compare single GA vs island model for harder problem."""
    n_atoms = 7

    print("=" * 70)
    print(f"Strategy Comparison: {n_atoms}-atom LJ Cluster")
    print("=" * 70)
    print("\nComparing single population vs island model...")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Single population
    print("\n--- Single Population GA ---")
    result_single = optimize_cluster(
        n_atoms=n_atoms,
        population_size=200,
        generations=500,
        islands=1,
        seed=42,
        verbose=True,
    )

    # Island model
    print("\n--- Island Model (4 islands) ---")
    result_island = optimize_cluster(
        n_atoms=n_atoms,
        population_size=200,
        generations=500,
        islands=4,
        seed=42,
        verbose=True,
    )

    known = KNOWN_MINIMA[n_atoms]
    print("\n--- Results ---")
    print(f"Known minimum: {known:.6f}")
    print(f"Single GA:     {result_single['best_energy']:.6f} (error: {abs(result_single['best_energy'] - known):.4f})")
    print(f"Island Model:  {result_island['best_energy']:.6f} (error: {abs(result_island['best_energy'] - known):.4f})")
    print(f"Time - Single: {result_single['elapsed']:.2f}s, Island: {result_island['elapsed']:.2f}s")

    # Comparison plot
    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            {
                "Single GA": result_single["fitness_history"],
                "Island Model (4)": result_island["fitness_history"],
            },
            title=f"LJ{n_atoms}: Single GA vs Island Model",
            ylabel="Energy (ε)",
            save_path=output_dir / "lj_strategy_comparison.png",
            show=False,
        )

        # Save best cluster
        best_result = result_island if result_island["best_energy"] < result_single["best_energy"] else result_single
        plot_3d_cluster(
            best_result["best_positions"],
            n_atoms,
            title=f"Best LJ{n_atoms} Configuration (E = {best_result['best_energy']:.4f})",
            save_path=output_dir / "lj7_best_cluster.png",
            show=False,
        )
        print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    import sys

    # Default output directory
    output_dir = Path(__file__).parent / "output"

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_strategies(output_dir)
    else:
        run_benchmark(output_dir)
