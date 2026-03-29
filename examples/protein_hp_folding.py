"""Protein HP Lattice Folding Case Study.

This example demonstrates using ParGA to solve a simplified protein folding
problem using the Hydrophobic-Polar (HP) model on a 2D lattice.

The HP model is a classic benchmark in computational biology:
- Amino acids are classified as Hydrophobic (H) or Polar (P)
- The protein sequence is embedded on a discrete lattice (2D square grid)
- The goal is to minimise energy by maximising H-H contacts

Energy function:
    E = -1 * (number of non-sequential H-H contacts)

The optimal folding maximises hydrophobic core formation, mimicking
how real proteins fold to bury hydrophobic residues.

This is an NP-hard combinatorial optimisation problem with a rugged
fitness landscape -- ideal for genetic algorithms with island models.

Known optimal H-H contacts for standard 2D benchmarks:
    seq10  (10 aa):  4   Dill (1985)
    seq20  (20 aa):  9   Unger & Moult (1993)
    seq24  (24 aa):  9   Unger & Moult (1993)
    seq36  (36 aa): 14   Yue & Dill (1995)
    seq48  (48 aa): 23   Yue & Dill (1995)

References:
    - Dill, K.A. (1985) "Theory for the folding and stability of globular proteins"
    - Lau, K.F. & Dill, K.A. (1989) "A lattice statistical mechanics model of the
      conformational and sequence spaces of proteins"
    - Unger, R. & Moult, J. (1993) "Genetic Algorithms for Protein Folding Simulations"
    - Yue, K. & Dill, K.A. (1995) "Forces of Tertiary Structural Organization in
      Globular Proteins"
"""

from __future__ import annotations

import time
from enum import IntEnum
from pathlib import Path

import numpy as np

from parga import minimize

# Try to import visualization - optional dependency
try:
    from parga.viz import plot_convergence_comparison, plot_lattice_protein

    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


class Move(IntEnum):
    """Direction moves on 2D lattice (relative to previous direction)."""

    FORWARD = 0  # Continue same direction
    LEFT = 1  # Turn left 90 degrees
    RIGHT = 2  # Turn right 90 degrees


# Direction vectors: Up, Right, Down, Left
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

# Standard benchmark sequences from the literature
BENCHMARK_SEQUENCES = {
    "seq10": "HPHPPHHPHP",                                       # 10 aa
    "seq20": "HPHPPHHPHPPHPHHPPHPH",                              # 20 aa  (Unger & Moult)
    "seq24": "HHPPHPPHPPHPPHPPHPPHPPHH",                          # 24 aa  (Unger & Moult)
    "seq36": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",               # 36 aa  (Yue & Dill)
    "seq48": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",   # 48 aa  (Yue & Dill)
}

# Known best H-H contacts for each benchmark (2D square lattice)
KNOWN_OPTIMA = {
    "seq10": 4,
    "seq20": 9,
    "seq24": 9,
    "seq36": 14,
    "seq48": 23,
}


def decode_moves(genes: np.ndarray) -> list[int]:
    """Convert continuous genes in [0, 3) to discrete moves (0, 1, 2).

    Using a wider range [0, 3) instead of [0, 1) gives each discrete
    move a full unit-width interval.  This improves crossover (children
    are more likely to inherit their parent's discrete value) and makes
    Gaussian mutation with sigma = 0.1 * range = 0.3 effective at
    exploring adjacent moves.
    """
    moves = np.clip(np.floor(genes).astype(int), 0, 2)
    return moves.tolist()


def fold_protein(sequence: str, moves: list[int]) -> list[tuple[int, int]] | None:
    """Fold a protein sequence according to move instructions.

    Returns:
        List of (x, y) coordinates for each amino acid, or None if invalid
        (self-intersection detected).
    """
    if len(moves) != len(sequence) - 1:
        raise ValueError(f"Need {len(sequence) - 1} moves for sequence of length {len(sequence)}")

    positions = [(0, 0)]
    occupied = {(0, 0)}
    direction = 0  # Start facing "up"

    for move in moves:
        if move == Move.LEFT:
            direction = (direction - 1) % 4
        elif move == Move.RIGHT:
            direction = (direction + 1) % 4

        dx, dy = DIRECTIONS[direction]
        new_pos = (positions[-1][0] + dx, positions[-1][1] + dy)

        if new_pos in occupied:
            return None

        positions.append(new_pos)
        occupied.add(new_pos)

    return positions


def count_hh_contacts(sequence: str, positions: list[tuple[int, int]]) -> int:
    """Count non-sequential H-H contacts (adjacent on lattice but not in sequence)."""
    if positions is None:
        return 0

    contacts = 0
    n = len(sequence)
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    for i in range(n):
        if sequence[i] != "H":
            continue

        x, y = positions[i]
        for dx, dy in DIRECTIONS:
            neighbor = (x + dx, y + dy)
            if neighbor in pos_to_idx:
                j = pos_to_idx[neighbor]
                if abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1

    return contacts // 2  # each contact counted twice


def fold_protein_partial(sequence: str, moves: list[int]) -> tuple[list[tuple[int, int]], int]:
    """Fold a protein, returning partial result and collision count.

    Returns (positions_placed, n_collisions).  Positions list contains
    all residues placed before the first collision, plus the colliding one.
    """
    positions = [(0, 0)]
    occupied = {(0, 0)}
    direction = 0
    collisions = 0

    for move in moves:
        if move == Move.LEFT:
            direction = (direction - 1) % 4
        elif move == Move.RIGHT:
            direction = (direction + 1) % 4

        dx, dy = DIRECTIONS[direction]
        new_pos = (positions[-1][0] + dx, positions[-1][1] + dy)

        if new_pos in occupied:
            collisions += 1
            # Skip this residue but continue trying
            continue

        positions.append(new_pos)
        occupied.add(new_pos)

    return positions, collisions


def hp_fitness(genes: np.ndarray, sequence: str) -> float:
    """Fitness function for HP folding.

    Returns number of H-H contacts (higher = better) with a graded
    penalty for self-intersecting conformations.  The penalty is
    proportional to the number of collisions so the GA can learn to
    reduce them rather than seeing a flat -1000 wall.
    """
    moves = decode_moves(genes)
    positions = fold_protein(sequence, moves)

    if positions is None:
        # Count collisions for a softer penalty
        _, n_collisions = fold_protein_partial(sequence, moves)
        return -10.0 * n_collisions

    return float(count_hh_contacts(sequence, positions))


def visualize_fold_ascii(sequence: str, positions: list[tuple[int, int]]) -> str:
    """Create ASCII visualization of the folded protein."""
    if positions is None:
        return "Invalid fold (self-intersection)"

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    width = max_x - min_x + 3
    height = max_y - min_y + 3
    grid = [[" " for _ in range(width * 2)] for _ in range(height)]

    for i, (x, y) in enumerate(positions):
        gx = (x - min_x + 1) * 2
        gy = height - (y - min_y + 2)
        grid[gy][gx] = sequence[i]

    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        gx1 = (x1 - min_x + 1) * 2
        gy1 = height - (y1 - min_y + 2)
        gx2 = (x2 - min_x + 1) * 2
        gy2 = height - (y2 - min_y + 2)

        if gx1 == gx2:
            mid_y = (gy1 + gy2) // 2
            grid[mid_y][gx1] = "|"
        else:
            mid_x = (gx1 + gx2) // 2
            grid[gy1][mid_x] = "-"

    return "\n".join("".join(row) for row in grid)


# ---------------------------------------------------------------------------
# GA parameters scaled by sequence length
# ---------------------------------------------------------------------------

def _ga_params(seq_len: int) -> dict:
    """Return GA parameters tuned for a given sequence length.

    Longer sequences have exponentially more conformations, so we
    increase population, generations, use island model, and run
    multiple restarts.
    """
    if seq_len <= 10:
        return dict(population_size=200, generations=400, islands=4,
                    mutation_rate=0.15, restarts=3)
    elif seq_len <= 20:
        return dict(population_size=300, generations=600, islands=4,
                    mutation_rate=0.15, restarts=3)
    elif seq_len <= 24:
        return dict(population_size=300, generations=800, islands=4,
                    mutation_rate=0.15, restarts=5)
    elif seq_len <= 36:
        return dict(population_size=400, generations=1000, islands=6,
                    mutation_rate=0.18, restarts=5)
    else:  # 48+
        return dict(population_size=500, generations=1200, islands=6,
                    mutation_rate=0.18, restarts=8)


def optimize_folding(
    sequence: str,
    name: str = "",
    population_size: int | None = None,
    generations: int | None = None,
    islands: int | None = None,
    mutation_rate: float | None = None,
    restarts: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Optimize protein folding for a given HP sequence.

    For longer sequences this runs multiple independent restarts
    and returns the best result.
    """
    params = _ga_params(len(sequence))
    population_size = population_size or params["population_size"]
    generations = generations or params["generations"]
    islands = islands or params["islands"]
    mutation_rate = mutation_rate or params["mutation_rate"]
    restarts = restarts or params["restarts"]

    n_moves = len(sequence) - 1

    def fitness(genes):
        return -hp_fitness(genes, sequence)  # Negate for minimize()

    if verbose:
        seq_display = sequence if len(sequence) <= 40 else sequence[:37] + "..."
        print(f"\nOptimising fold for: {seq_display}")
        print(f"  Length: {len(sequence)} residues  (H: {sequence.count('H')}, P: {sequence.count('P')})")
        print(f"  Moves to optimise: {n_moves}")
        print(f"  Population: {population_size}, Generations: {generations}")
        if islands > 1:
            print(f"  Island model: {islands} islands")
        if restarts > 1:
            print(f"  Independent restarts: {restarts}")

    best_contacts = 0
    best_positions = None
    best_history = None
    best_strategy = None
    total_time = 0.0

    for r in range(restarts):
        run_seed = seed + r if seed is not None else None
        t0 = time.perf_counter()

        result = minimize(
            fitness,
            genome_length=n_moves,
            bounds=(0.0, 3.0),
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

        # Decode best solution
        moves = decode_moves(result.best_genes())
        positions = fold_protein(sequence, moves)
        contacts = count_hh_contacts(sequence, positions) if positions else 0

        if verbose and restarts > 1:
            print(f"    Restart {r+1}/{restarts}: {contacts} contacts  ({elapsed:.1f}s)")

        if contacts > best_contacts:
            best_contacts = contacts
            best_positions = positions
            best_history = result.fitness_history
            best_strategy = result.strategy

    if verbose:
        print(f"  Strategy: {best_strategy}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  H-H contacts: {best_contacts}")
        if name and name in KNOWN_OPTIMA:
            opt = KNOWN_OPTIMA[name]
            pct = 100 * best_contacts / opt if opt > 0 else 0
            print(f"  Known optimum: {opt}  ({pct:.0f}% achieved)")

    return {
        "sequence": sequence,
        "name": name,
        "positions": best_positions,
        "contacts": best_contacts,
        "fitness_history": best_history,
        "strategy": best_strategy,
        "elapsed": total_time,
    }


def run_benchmark(output_dir: Path | None = None):
    """Run benchmark on all standard HP sequences."""
    print("=" * 70)
    print("HP Protein Folding Benchmark")
    print("=" * 70)
    print("\nThe HP model simplifies protein folding:")
    print("  - H = Hydrophobic amino acid")
    print("  - P = Polar amino acid")
    print("  - Goal: Maximise H-H contacts (minimise energy)")
    print("\nKnown optima from Unger & Moult (1993) and Yue & Dill (1995).")
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_histories = {}

    for name, sequence in BENCHMARK_SEQUENCES.items():
        print(f"\n{'=' * 50}")
        print(f"Sequence: {name}  ({len(sequence)} residues)")
        print("=" * 50)

        result = optimize_folding(
            sequence,
            name=name,
            seed=42,
            verbose=True,
        )

        known = KNOWN_OPTIMA.get(name, "?")
        results.append((name, len(sequence), result["contacts"], known, result["elapsed"]))
        all_histories[name] = [-f for f in result["fitness_history"]]

        if result["positions"]:
            print(f"\nFolded structure (ASCII):")
            print(visualize_fold_ascii(sequence, result["positions"]))

            if HAS_VIZ and output_dir:
                plot_lattice_protein(
                    sequence,
                    result["positions"],
                    title=f"{name}: {result['contacts']} H-H contacts (opt={known})",
                    save_path=output_dir / f"hp_fold_{name}.png",
                    show=False,
                )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Sequence':>10} {'Length':>8} {'Found':>8} {'Known':>8} {'%':>7} {'Time (s)':>10}")
    print("-" * 54)
    for name, length, contacts, known, elapsed in results:
        if isinstance(known, int) and known > 0:
            pct = 100 * contacts / known
            marker = " *" if pct >= 100 else ""
            print(f"{name:>10} {length:>8} {contacts:>8} {known:>8} {pct:>6.0f}% {elapsed:>10.1f}{marker}")
        else:
            print(f"{name:>10} {length:>8} {contacts:>8} {'?':>8} {'':>7} {elapsed:>10.1f}")
    print("\n  * = global optimum found")

    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            all_histories,
            title="HP Folding Optimization Convergence",
            ylabel="H-H Contacts",
            save_path=output_dir / "hp_convergence_comparison.png",
            show=False,
        )
        print(f"\nPlots saved to {output_dir}/")

    return results


def compare_strategies(output_dir: Path | None = None):
    """Compare single GA vs island model on seq36."""
    name = "seq36"
    sequence = BENCHMARK_SEQUENCES[name]

    print("=" * 70)
    print(f"Strategy Comparison: {name} ({len(sequence)} residues)")
    print("=" * 70)
    print(f"Sequence: {sequence}")
    print(f"Known optimum: {KNOWN_OPTIMA[name]} H-H contacts")
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Single population - 3 restarts
    print("--- Single Population GA (3 restarts) ---")
    result_single = optimize_folding(
        sequence,
        name=name,
        population_size=300,
        generations=800,
        islands=1,
        restarts=3,
        seed=42,
        verbose=True,
    )

    # Island model - 3 restarts
    print("\n--- Island Model, 4 islands (3 restarts) ---")
    result_island = optimize_folding(
        sequence,
        name=name,
        population_size=300,
        generations=800,
        islands=4,
        restarts=3,
        seed=42,
        verbose=True,
    )

    known = KNOWN_OPTIMA[name]
    print(f"\n--- Results (known optimum: {known}) ---")
    print(f"Single GA:    {result_single['contacts']} contacts in {result_single['elapsed']:.1f}s")
    print(f"Island Model: {result_island['contacts']} contacts in {result_island['elapsed']:.1f}s")

    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            {
                "Single GA": [-f for f in result_single["fitness_history"]],
                "Island Model (4)": [-f for f in result_island["fitness_history"]],
            },
            title=f"HP Folding ({name}): Single GA vs Island Model",
            ylabel="H-H Contacts",
            save_path=output_dir / "hp_strategy_comparison.png",
            show=False,
        )

        best_result = (
            result_island
            if result_island["contacts"] >= result_single["contacts"]
            else result_single
        )
        if best_result["positions"]:
            plot_lattice_protein(
                sequence,
                best_result["positions"],
                title=f"Best {name} Fold: {best_result['contacts']} H-H contacts (opt={known})",
                save_path=output_dir / "hp_best_fold.png",
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
