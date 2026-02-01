"""Protein HP Lattice Folding Case Study.

This example demonstrates using ParGA to solve a simplified protein folding
problem using the Hydrophobic-Polar (HP) model on a 2D lattice.

The HP model is a classic benchmark in computational biology:
- Amino acids are classified as Hydrophobic (H) or Polar (P)
- The protein sequence is embedded on a discrete lattice (2D square grid)
- The goal is to minimize energy by maximizing H-H contacts

Energy function:
    E = -1 * (number of non-sequential H-H contacts)

The optimal folding maximizes hydrophobic core formation, mimicking
how real proteins fold to bury hydrophobic residues.

This is an NP-hard combinatorial optimization problem with a rugged
fitness landscape - ideal for genetic algorithms.

References:
    - Dill, K.A. (1985) "Theory for the folding and stability of globular proteins"
    - Lau, K.F. & Dill, K.A. (1989) "A lattice statistical mechanics model of the
      conformational and sequence spaces of proteins"
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
    # Simple sequences for testing
    "seq10": "HPHPPHHPHP",  # 10 residues
    "seq20": "HPHPPHHPHPPHPHHPPHPH",  # 20 residues
    # Classic benchmark sequences
    "seq24": "HHPPHPPHPPHPPHPPHPPHPPHH",  # 24 residues
    "seq36": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP",  # 36 residues
    "seq48": "PPHPPHHPPHHPPPPPHHHHHHHHHHPPPPPPHHPPHHPPHPPHHHHH",  # 48 residues
}


def decode_moves(genes: np.ndarray) -> list[int]:
    """Convert continuous genes to discrete moves (0, 1, 2)."""
    # Map [0, 3) to move indices
    moves = np.clip(np.floor(genes * 3).astype(int), 0, 2)
    return moves.tolist()


def fold_protein(sequence: str, moves: list[int]) -> list[tuple[int, int]] | None:
    """Fold a protein sequence according to move instructions.

    Args:
        sequence: HP sequence string (e.g., "HPHPPHHPHPPHH")
        moves: List of moves (0=forward, 1=left, 2=right)

    Returns:
        List of (x, y) coordinates for each amino acid, or None if invalid
        (self-intersection detected)
    """
    if len(moves) != len(sequence) - 1:
        raise ValueError(f"Need {len(sequence) - 1} moves for sequence of length {len(sequence)}")

    positions = [(0, 0)]  # First amino acid at origin
    occupied = {(0, 0)}
    direction = 0  # Start facing "up"

    for move in moves:
        # Apply turn
        if move == Move.LEFT:
            direction = (direction - 1) % 4
        elif move == Move.RIGHT:
            direction = (direction + 1) % 4
        # FORWARD keeps same direction

        # Move in current direction
        dx, dy = DIRECTIONS[direction]
        new_pos = (positions[-1][0] + dx, positions[-1][1] + dy)

        # Check for self-intersection
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

    # Build position lookup
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}

    for i in range(n):
        if sequence[i] != "H":
            continue

        x, y = positions[i]

        # Check all 4 neighbors
        for dx, dy in DIRECTIONS:
            neighbor = (x + dx, y + dy)
            if neighbor in pos_to_idx:
                j = pos_to_idx[neighbor]
                # Must be non-sequential and also H
                if abs(i - j) > 1 and sequence[j] == "H":
                    contacts += 1

    # Each contact counted twice (i->j and j->i)
    return contacts // 2


def hp_fitness(genes: np.ndarray, sequence: str) -> float:
    """Fitness function for HP folding.

    Args:
        genes: Array of continuous values in [0, 1) to be discretized
        sequence: HP sequence string

    Returns:
        Negative energy (higher is better for GA)
    """
    moves = decode_moves(genes)
    positions = fold_protein(sequence, moves)

    if positions is None:
        # Invalid fold (self-intersection) - heavy penalty
        return -1000.0

    contacts = count_hh_contacts(sequence, positions)
    # Energy = -contacts, so fitness = contacts (more contacts = better)
    return float(contacts)


def visualize_fold_ascii(sequence: str, positions: list[tuple[int, int]]) -> str:
    """Create ASCII visualization of the folded protein."""
    if positions is None:
        return "Invalid fold (self-intersection)"

    # Find bounds
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Create grid with padding
    width = max_x - min_x + 3
    height = max_y - min_y + 3
    grid = [[" " for _ in range(width * 2)] for _ in range(height)]

    # Place amino acids
    for i, (x, y) in enumerate(positions):
        gx = (x - min_x + 1) * 2
        gy = height - (y - min_y + 2)
        grid[gy][gx] = sequence[i]

    # Draw connections
    for i in range(len(positions) - 1):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        gx1 = (x1 - min_x + 1) * 2
        gy1 = height - (y1 - min_y + 2)
        gx2 = (x2 - min_x + 1) * 2
        gy2 = height - (y2 - min_y + 2)

        # Draw connector
        if gx1 == gx2:  # Vertical
            mid_y = (gy1 + gy2) // 2
            grid[mid_y][gx1] = "|"
        else:  # Horizontal
            mid_x = (gx1 + gx2) // 2
            grid[gy1][mid_x] = "-"

    return "\n".join("".join(row) for row in grid)


def optimize_folding(
    sequence: str,
    population_size: int = 200,
    generations: int = 500,
    islands: int = 1,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Optimize protein folding for a given HP sequence.

    Args:
        sequence: HP sequence string
        population_size: GA population size
        generations: Number of generations
        islands: Number of islands (use >1 for island model)
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary with optimization results
    """
    n_moves = len(sequence) - 1

    # Create fitness function with sequence closure
    def fitness(genes):
        return -hp_fitness(genes, sequence)  # Negate for minimize()

    if verbose:
        print(f"\nOptimizing fold for sequence: {sequence[:30]}{'...' if len(sequence) > 30 else ''}")
        print(f"  Length: {len(sequence)} residues")
        print(f"  H count: {sequence.count('H')}")
        print(f"  Moves to optimize: {n_moves}")

    start_time = time.perf_counter()

    result = minimize(
        fitness,
        genome_length=n_moves,
        bounds=(0.0, 1.0),  # Will be discretized to moves
        population_size=population_size,
        generations=generations,
        islands=islands,
        mutation_rate=0.1,  # Higher mutation for discrete problem
        crossover_rate=0.8,
        seed=seed,
        verbose=False,
    )

    elapsed = time.perf_counter() - start_time

    # Decode best solution
    best_moves = decode_moves(result.best_genes())
    best_positions = fold_protein(sequence, best_moves)
    best_contacts = count_hh_contacts(sequence, best_positions) if best_positions else 0

    if verbose:
        print(f"  Strategy: {result.strategy}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  H-H contacts: {best_contacts}")

    return {
        "sequence": sequence,
        "positions": best_positions,
        "contacts": best_contacts,
        "fitness_history": result.fitness_history,
        "strategy": result.strategy,
        "elapsed": elapsed,
    }


def run_benchmark(output_dir: Path | None = None):
    """Run benchmark on standard HP sequences."""
    print("=" * 70)
    print("HP Protein Folding Benchmark")
    print("=" * 70)
    print("\nThe HP model simplifies protein folding:")
    print("  - H = Hydrophobic amino acid")
    print("  - P = Polar amino acid")
    print("  - Goal: Maximize H-H contacts (minimize energy)")
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_histories = {}

    for name, sequence in list(BENCHMARK_SEQUENCES.items())[:3]:  # First 3 for speed
        print(f"\n{'='*50}")
        print(f"Sequence: {name}")
        print("=" * 50)

        result = optimize_folding(
            sequence,
            population_size=200,
            generations=300,
            seed=42,
            verbose=True,
        )

        results.append((name, len(sequence), result["contacts"]))
        all_histories[name] = [-f for f in result["fitness_history"]]  # Convert to contacts

        if result["positions"]:
            print(f"\nFolded structure (ASCII):")
            print(visualize_fold_ascii(sequence, result["positions"]))

            # Save graphical plot
            if HAS_VIZ and output_dir:
                plot_lattice_protein(
                    sequence,
                    result["positions"],
                    title=f"{name}: {result['contacts']} H-H contacts",
                    save_path=output_dir / f"hp_fold_{name}.png",
                    show=False,
                )

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Sequence':>10} {'Length':>8} {'H-H Contacts':>12}")
    print("-" * 32)
    for name, length, contacts in results:
        print(f"{name:>10} {length:>8} {contacts:>12}")

    # Convergence plot
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
    """Compare single GA vs island model."""
    sequence = BENCHMARK_SEQUENCES["seq20"]

    print("=" * 70)
    print("Strategy Comparison: HP Folding")
    print("=" * 70)
    print(f"Sequence: {sequence}")
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Single population
    print("--- Single Population GA ---")
    result_single = optimize_folding(
        sequence,
        population_size=200,
        generations=500,
        islands=1,
        seed=42,
        verbose=True,
    )

    # Island model
    print("\n--- Island Model (4 islands) ---")
    result_island = optimize_folding(
        sequence,
        population_size=200,
        generations=500,
        islands=4,
        seed=42,
        verbose=True,
    )

    print("\n--- Results ---")
    print(f"Single GA:    {result_single['contacts']} contacts in {result_single['elapsed']:.2f}s")
    print(f"Island Model: {result_island['contacts']} contacts in {result_island['elapsed']:.2f}s")

    # Comparison plots
    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            {
                "Single GA": [-f for f in result_single["fitness_history"]],
                "Island Model (4)": [-f for f in result_island["fitness_history"]],
            },
            title="HP Folding: Single GA vs Island Model",
            ylabel="H-H Contacts",
            save_path=output_dir / "hp_strategy_comparison.png",
            show=False,
        )

        # Save best fold
        best_result = result_island if result_island["contacts"] >= result_single["contacts"] else result_single
        if best_result["positions"]:
            plot_lattice_protein(
                sequence,
                best_result["positions"],
                title=f"Best seq20 Fold: {best_result['contacts']} H-H contacts",
                save_path=output_dir / "hp_best_fold.png",
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
