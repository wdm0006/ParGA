"""Alloy Composition Optimization Case Study.

This example demonstrates using ParGA to optimize the composition of
multi-component alloys for desired material properties.

Background:
    Material scientists design alloys by combining multiple elements in
    specific proportions to achieve target properties like strength,
    hardness, corrosion resistance, and density. This is a multi-objective
    optimization problem with complex non-linear relationships.

    High-Entropy Alloys (HEAs) are a recent innovation containing 5+ principal
    elements in roughly equal proportions, offering exceptional properties.

This simplified model uses empirical mixing rules and surrogate property
functions to demonstrate the optimization approach used in real materials
research (which would use DFT calculations or experimental data).

Properties modeled:
    - Yield strength (MPa)
    - Hardness (HV)
    - Density (g/cm^3)
    - Corrosion resistance index

References:
    - Yeh, J.W. et al. (2004) "High-Entropy Alloys: A New Era of Exploitation"
    - Miracle, D.B. & Senkov, O.N. (2017) "A critical review of high entropy alloys"
    - Nature Computational Materials (2019) "Genetic algorithms for computational
      materials discovery accelerated by machine learning"
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parga import minimize

# Try to import visualization - optional dependency
try:
    from parga.viz import (
        plot_composition_bar,
        plot_convergence_comparison,
        plot_properties_radar,
    )

    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


@dataclass
class Element:
    """Properties of a pure element."""

    symbol: str
    atomic_number: int
    atomic_weight: float  # g/mol
    density: float  # g/cm^3
    melting_point: float  # K
    yield_strength: float  # MPa (for pure metal)
    hardness: float  # HV (Vickers hardness)
    corrosion_index: float  # Arbitrary scale 0-100 (higher = more resistant)
    cost: float  # $/kg (approximate)


# Element database (simplified properties)
ELEMENTS = {
    "Fe": Element("Fe", 26, 55.85, 7.87, 1811, 170, 150, 40, 0.5),
    "Ni": Element("Ni", 28, 58.69, 8.91, 1728, 148, 160, 70, 15),
    "Cr": Element("Cr", 24, 52.00, 7.19, 2180, 280, 200, 85, 8),
    "Co": Element("Co", 27, 58.93, 8.90, 1768, 225, 180, 65, 30),
    "Al": Element("Al", 13, 26.98, 2.70, 933, 35, 30, 75, 2),
    "Ti": Element("Ti", 22, 47.87, 4.51, 1941, 275, 250, 90, 20),
    "Cu": Element("Cu", 29, 63.55, 8.96, 1358, 70, 50, 50, 6),
    "Mn": Element("Mn", 25, 54.94, 7.21, 1519, 240, 200, 30, 2),
    "V": Element("V", 23, 50.94, 6.11, 2183, 310, 280, 60, 25),
    "Mo": Element("Mo", 42, 95.94, 10.28, 2896, 330, 300, 80, 40),
}


def normalize_composition(fractions: np.ndarray) -> np.ndarray:
    """Normalize atomic fractions to sum to 1."""
    total = np.sum(fractions)
    if total > 0:
        return fractions / total
    return fractions


def rule_of_mixtures(fractions: np.ndarray, element_values: np.ndarray) -> float:
    """Calculate property using rule of mixtures."""
    return np.sum(fractions * element_values)


def mixing_entropy(fractions: np.ndarray) -> float:
    """Calculate configurational mixing entropy (J/mol/K).

    S_mix = -R * sum(x_i * ln(x_i))
    """
    R = 8.314  # Gas constant
    # Avoid log(0)
    f = fractions[fractions > 0.001]
    return -R * np.sum(f * np.log(f))


def calculate_properties(
    fractions: np.ndarray,
    elements: list[str],
) -> dict[str, float]:
    """Calculate alloy properties from composition.

    Args:
        fractions: Atomic fractions (will be normalized)
        elements: List of element symbols

    Returns:
        Dictionary of calculated properties
    """
    fractions = normalize_composition(fractions)
    elem_list = [ELEMENTS[e] for e in elements]

    # Extract element properties
    densities = np.array([e.density for e in elem_list])
    yield_strengths = np.array([e.yield_strength for e in elem_list])
    hardnesses = np.array([e.hardness for e in elem_list])
    corrosion = np.array([e.corrosion_index for e in elem_list])
    costs = np.array([e.cost for e in elem_list])

    # Rule of mixtures for base properties
    base_density = rule_of_mixtures(fractions, densities)
    base_strength = rule_of_mixtures(fractions, yield_strengths)
    base_hardness = rule_of_mixtures(fractions, hardnesses)
    base_corrosion = rule_of_mixtures(fractions, corrosion)
    cost = rule_of_mixtures(fractions, costs)

    # Mixing entropy
    s_mix = mixing_entropy(fractions)
    is_hea = s_mix > 11.5  # High-entropy alloy threshold (~1.5R)

    # Solid solution strengthening (simplified model)
    # Higher entropy -> more lattice distortion -> higher strength
    ss_factor = 1 + 0.5 * (s_mix / 15)  # Up to 50% boost for HEAs

    # Synergy effects (simplified)
    # Certain element pairs have beneficial interactions
    synergy = 1.0
    for i, e1 in enumerate(elements):
        for j, e2 in enumerate(elements):
            if i < j and fractions[i] > 0.05 and fractions[j] > 0.05:
                # Cr-Ni synergy for corrosion
                if {e1, e2} == {"Cr", "Ni"}:
                    synergy *= 1.1
                # Al-Ti synergy for strength
                if {e1, e2} == {"Al", "Ti"}:
                    synergy *= 1.15
                # Fe-Cr synergy for hardness
                if {e1, e2} == {"Fe", "Cr"}:
                    synergy *= 1.08

    # Final properties
    yield_strength = base_strength * ss_factor * synergy
    hardness = base_hardness * ss_factor
    corrosion_resistance = base_corrosion * (1 + 0.3 * (s_mix / 15))

    return {
        "yield_strength": yield_strength,
        "hardness": hardness,
        "density": base_density,
        "corrosion_resistance": corrosion_resistance,
        "mixing_entropy": s_mix,
        "is_hea": is_hea,
        "cost": cost,
        "composition": dict(zip(elements, fractions)),
    }


def multi_objective_fitness(
    fractions: np.ndarray,
    elements: list[str],
    target_strength: float = 500,
    target_hardness: float = 300,
    max_density: float = 8.0,
    min_corrosion: float = 60,
    weight_strength: float = 1.0,
    weight_hardness: float = 1.0,
    weight_density: float = 0.5,
    weight_corrosion: float = 0.5,
) -> float:
    """Multi-objective fitness function for alloy optimization.

    Args:
        fractions: Atomic fractions
        elements: Element symbols
        target_*: Target property values
        weight_*: Importance weights for each objective

    Returns:
        Fitness score (higher is better)
    """
    props = calculate_properties(fractions, elements)

    # Normalize scores relative to targets
    strength_score = min(props["yield_strength"] / target_strength, 1.5)
    hardness_score = min(props["hardness"] / target_hardness, 1.5)

    # Density: lower is better (penalize above max)
    if props["density"] <= max_density:
        density_score = 1.0
    else:
        density_score = max_density / props["density"]

    # Corrosion: higher is better
    corrosion_score = min(props["corrosion_resistance"] / min_corrosion, 1.5)

    # Bonus for high-entropy alloys
    hea_bonus = 1.1 if props["is_hea"] else 1.0

    # Weighted sum
    fitness = (
        weight_strength * strength_score
        + weight_hardness * hardness_score
        + weight_density * density_score
        + weight_corrosion * corrosion_score
    ) * hea_bonus

    return fitness


def optimize_alloy(
    elements: list[str],
    target_strength: float = 500,
    target_hardness: float = 300,
    max_density: float = 8.0,
    min_corrosion: float = 60,
    population_size: int = 100,
    generations: int = 200,
    islands: int = 1,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Optimize alloy composition for target properties.

    Args:
        elements: List of elements to include in the alloy
        target_*: Target property values
        population_size: GA population size
        generations: Number of generations
        islands: Number of islands
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary with optimized composition and properties
    """
    n_elements = len(elements)

    def fitness(genes):
        # genes are in [0, 1], normalize to fractions
        fractions = genes / (np.sum(genes) + 1e-10)
        return -multi_objective_fitness(
            fractions,
            elements,
            target_strength=target_strength,
            target_hardness=target_hardness,
            max_density=max_density,
            min_corrosion=min_corrosion,
        )

    if verbose:
        print(f"\nOptimizing {n_elements}-component alloy: {'-'.join(elements)}")
        print(f"  Target strength: {target_strength} MPa")
        print(f"  Target hardness: {target_hardness} HV")
        print(f"  Max density: {max_density} g/cm^3")
        print(f"  Min corrosion resistance: {min_corrosion}")

    start_time = time.perf_counter()

    result = minimize(
        fitness,
        genome_length=n_elements,
        bounds=(0.01, 1.0),  # Min 1% for each element
        population_size=population_size,
        generations=generations,
        islands=islands,
        mutation_rate=0.05,
        crossover_rate=0.8,
        seed=seed,
        verbose=False,
    )

    elapsed = time.perf_counter() - start_time

    # Decode best solution
    best_fractions = result.best_genes()
    best_fractions = normalize_composition(best_fractions)
    props = calculate_properties(best_fractions, elements)

    if verbose:
        print(f"  Strategy: {result.strategy}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"\n  Optimized composition (at%):")
        for elem, frac in props["composition"].items():
            print(f"    {elem}: {frac*100:.1f}%")
        print(f"\n  Properties:")
        print(f"    Yield strength: {props['yield_strength']:.0f} MPa")
        print(f"    Hardness: {props['hardness']:.0f} HV")
        print(f"    Density: {props['density']:.2f} g/cm^3")
        print(f"    Corrosion resistance: {props['corrosion_resistance']:.1f}")
        print(f"    Mixing entropy: {props['mixing_entropy']:.1f} J/mol/K")
        print(f"    High-entropy alloy: {'Yes' if props['is_hea'] else 'No'}")

    return {
        "elements": elements,
        "fractions": best_fractions,
        "properties": props,
        "fitness_history": result.fitness_history,
        "strategy": result.strategy,
        "elapsed": elapsed,
    }


def run_benchmark(output_dir: Path | None = None):
    """Run optimization for various alloy systems."""
    print("=" * 70)
    print("Multi-Component Alloy Composition Optimization")
    print("=" * 70)
    print("\nOptimizing alloy compositions to achieve target material properties.")
    print("This simulates computational materials discovery workflows.")
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Test cases
    test_cases = [
        # Stainless steel family
        {
            "name": "Stainless Steel",
            "elements": ["Fe", "Cr", "Ni", "Mn"],
            "targets": {"target_strength": 400, "target_hardness": 250, "min_corrosion": 70},
        },
        # Superalloy family
        {
            "name": "Ni-based Superalloy",
            "elements": ["Ni", "Cr", "Co", "Al", "Ti"],
            "targets": {"target_strength": 600, "target_hardness": 350, "max_density": 9.0},
        },
        # High-entropy alloy
        {
            "name": "High-Entropy Alloy (Cantor)",
            "elements": ["Fe", "Ni", "Cr", "Co", "Mn"],
            "targets": {"target_strength": 500, "target_hardness": 300, "min_corrosion": 65},
        },
        # Lightweight alloy
        {
            "name": "Lightweight High-Strength",
            "elements": ["Al", "Ti", "V", "Cr", "Fe"],
            "targets": {"target_strength": 450, "max_density": 5.5, "min_corrosion": 80},
        },
    ]

    results = []
    all_histories = {}

    for tc in test_cases:
        print(f"\n{'='*50}")
        print(f"Alloy System: {tc['name']}")
        print("=" * 50)

        result = optimize_alloy(
            elements=tc["elements"],
            **tc["targets"],
            population_size=100,
            generations=150,
            seed=42,
            verbose=True,
        )
        results.append((tc["name"], result))
        all_histories[tc["name"]] = result["fitness_history"]

        # Save composition and properties plots
        if HAS_VIZ and output_dir:
            safe_name = tc["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")

            plot_composition_bar(
                tc["elements"],
                result["fractions"],
                title=f"{tc['name']} Composition",
                save_path=output_dir / f"alloy_composition_{safe_name}.png",
                show=False,
            )

            plot_properties_radar(
                {
                    "Strength": result["properties"]["yield_strength"],
                    "Hardness": result["properties"]["hardness"],
                    "Corrosion Res.": result["properties"]["corrosion_resistance"],
                    "Low Density": 10 - result["properties"]["density"],  # Invert for "better"
                },
                title=f"{tc['name']} Properties",
                save_path=output_dir / f"alloy_properties_{safe_name}.png",
                show=False,
            )

    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Alloy':25} {'Strength':>10} {'Hardness':>10} {'Density':>10} {'HEA':>5}")
    print("-" * 62)
    for name, result in results:
        p = result["properties"]
        hea = "Yes" if p["is_hea"] else "No"
        print(f"{name:25} {p['yield_strength']:>10.0f} {p['hardness']:>10.0f} {p['density']:>10.2f} {hea:>5}")

    # Convergence comparison
    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            all_histories,
            title="Alloy Optimization Convergence",
            ylabel="Negative Fitness",
            save_path=output_dir / "alloy_convergence_comparison.png",
            show=False,
        )
        print(f"\nPlots saved to {output_dir}/")

    return results


def compare_strategies(output_dir: Path | None = None):
    """Compare single GA vs island model for HEA optimization."""
    elements = ["Fe", "Ni", "Cr", "Co", "Mn", "Al"]

    print("=" * 70)
    print("Strategy Comparison: 6-Component HEA Optimization")
    print("=" * 70)
    print(f"Elements: {'-'.join(elements)}")
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Single population
    print("--- Single Population GA ---")
    result_single = optimize_alloy(
        elements=elements,
        population_size=150,
        generations=300,
        islands=1,
        seed=42,
        verbose=True,
    )

    # Island model
    print("\n--- Island Model (4 islands) ---")
    result_island = optimize_alloy(
        elements=elements,
        population_size=150,
        generations=300,
        islands=4,
        seed=42,
        verbose=True,
    )

    print("\n--- Results ---")
    print(f"Single GA:    fitness={-result_single['fitness_history'][-1]:.4f} in {result_single['elapsed']:.2f}s")
    print(f"Island Model: fitness={-result_island['fitness_history'][-1]:.4f} in {result_island['elapsed']:.2f}s")

    # Comparison plots
    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            {
                "Single GA": result_single["fitness_history"],
                "Island Model (4)": result_island["fitness_history"],
            },
            title="HEA Optimization: Single GA vs Island Model",
            save_path=output_dir / "alloy_strategy_comparison.png",
            show=False,
        )

        # Save best result
        best_result = result_island if result_island["fitness_history"][-1] < result_single["fitness_history"][-1] else result_single
        plot_composition_bar(
            elements,
            best_result["fractions"],
            title="Best 6-Component HEA Composition",
            save_path=output_dir / "alloy_best_composition.png",
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
