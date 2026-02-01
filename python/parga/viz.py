"""Visualization utilities for genetic algorithm results.

This module provides common plotting functions for analyzing and
presenting genetic algorithm optimization results.

All functions check for matplotlib availability and provide helpful
error messages if it's not installed.

Example:
    >>> from parga import minimize
    >>> from parga.viz import plot_convergence, plot_solution_landscape
    >>> import numpy as np
    >>>
    >>> def sphere(x):
    ...     return np.sum(x**2)
    >>>
    >>> result = minimize(sphere, genome_length=2, bounds=(-5, 5))
    >>> plot_convergence(result.fitness_history, title="Sphere Optimization")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Callable

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Figure = None


def _check_matplotlib():
    """Raise helpful error if matplotlib is not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_convergence(
    fitness_history: list[float],
    title: str = "GA Convergence",
    xlabel: str = "Generation",
    ylabel: str = "Best Fitness",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
    show: bool = True,
    log_scale: bool = False,
    ax: "plt.Axes | None" = None,
) -> "Figure | None":
    """Plot fitness convergence over generations.

    Args:
        fitness_history: List of best fitness values per generation
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height) in inches
        save_path: If provided, save figure to this path
        show: Whether to display the plot
        log_scale: Use logarithmic y-axis
        ax: Existing axes to plot on (creates new figure if None)

    Returns:
        Figure object if ax was None, else None
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure
        created_fig = False

    generations = range(len(fitness_history))
    ax.plot(generations, fitness_history, "b-", linewidth=2, label="Best Fitness")

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig if created_fig else None


def plot_convergence_comparison(
    histories: dict[str, list[float]],
    title: str = "Convergence Comparison",
    xlabel: str = "Generation",
    ylabel: str = "Best Fitness",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
    show: bool = True,
    log_scale: bool = False,
) -> "Figure":
    """Plot multiple convergence curves for comparison.

    Args:
        histories: Dict mapping labels to fitness history lists
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Save path
        show: Whether to display
        log_scale: Use log y-axis

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (label, history), color in zip(histories.items(), colors):
        generations = range(len(history))
        ax.plot(generations, history, linewidth=2, label=label, color=color)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if log_scale:
        ax.set_yscale("log")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_2d_landscape(
    fitness_fn: "Callable[[np.ndarray], float]",
    bounds: tuple[float, float],
    resolution: int = 100,
    title: str = "Fitness Landscape",
    best_point: np.ndarray | None = None,
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
    show: bool = True,
    cmap: str = "viridis",
    contour_levels: int = 20,
) -> "Figure":
    """Plot 2D fitness landscape with optional best solution marker.

    Args:
        fitness_fn: Fitness function taking 2D array
        bounds: (lower, upper) bounds for both dimensions
        resolution: Grid resolution
        title: Plot title
        best_point: Optional best solution to mark on plot
        figsize: Figure size
        save_path: Save path
        show: Whether to display
        cmap: Colormap name
        contour_levels: Number of contour levels

    Returns:
        Figure object
    """
    _check_matplotlib()

    lower, upper = bounds
    x = np.linspace(lower, upper, resolution)
    y = np.linspace(lower, upper, resolution)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = fitness_fn(np.array([X[i, j], Y[i, j]]))

    fig, ax = plt.subplots(figsize=figsize)

    contour = ax.contourf(X, Y, Z, levels=contour_levels, cmap=cmap)
    ax.contour(X, Y, Z, levels=contour_levels, colors="black", alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax, label="Fitness")

    if best_point is not None:
        ax.scatter(
            best_point[0],
            best_point[1],
            c="red",
            s=200,
            marker="*",
            edgecolors="white",
            linewidths=2,
            zorder=5,
            label="Best Solution",
        )
        ax.legend()

    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_parameter_sensitivity(
    param_values: list[float],
    fitness_values: list[float],
    param_name: str,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> "Figure":
    """Plot how fitness varies with a parameter.

    Args:
        param_values: List of parameter values tested
        fitness_values: Corresponding fitness values
        param_name: Name of the parameter
        title: Plot title (auto-generated if None)
        figsize: Figure size
        save_path: Save path
        show: Whether to display

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(param_values, fitness_values, "bo-", linewidth=2, markersize=8)

    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(title or f"Sensitivity to {param_name}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_population_diversity(
    populations: list[np.ndarray],
    generations: list[int] | None = None,
    title: str = "Population Diversity Over Time",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> "Figure":
    """Plot population diversity (standard deviation) over generations.

    Args:
        populations: List of population arrays (each shape: [pop_size, genome_len])
        generations: Generation numbers (defaults to 0, 1, 2, ...)
        title: Plot title
        figsize: Figure size
        save_path: Save path
        show: Whether to display

    Returns:
        Figure object
    """
    _check_matplotlib()

    if generations is None:
        generations = list(range(len(populations)))

    # Calculate diversity as mean std across genes
    diversities = [np.mean(np.std(pop, axis=0)) for pop in populations]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(generations, diversities, "g-", linewidth=2)
    ax.fill_between(generations, 0, diversities, alpha=0.3, color="green")

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Mean Gene Std Dev", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_pareto_front(
    objective1: list[float],
    objective2: list[float],
    obj1_name: str = "Objective 1",
    obj2_name: str = "Objective 2",
    highlight_pareto: bool = True,
    title: str = "Multi-Objective Optimization Results",
    figsize: tuple[float, float] = (10, 8),
    save_path: str | Path | None = None,
    show: bool = True,
) -> "Figure":
    """Plot solutions in 2D objective space with optional Pareto front.

    Args:
        objective1: First objective values
        objective2: Second objective values
        obj1_name: Name of first objective
        obj2_name: Name of second objective
        highlight_pareto: Whether to highlight Pareto-optimal points
        title: Plot title
        figsize: Figure size
        save_path: Save path
        show: Whether to display

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    obj1 = np.array(objective1)
    obj2 = np.array(objective2)

    if highlight_pareto:
        # Find Pareto front (assuming minimization of both)
        is_pareto = np.ones(len(obj1), dtype=bool)
        for i in range(len(obj1)):
            for j in range(len(obj1)):
                if i != j:
                    # j dominates i if j is better in both objectives
                    if obj1[j] <= obj1[i] and obj2[j] <= obj2[i]:
                        if obj1[j] < obj1[i] or obj2[j] < obj2[i]:
                            is_pareto[i] = False
                            break

        # Plot non-Pareto points
        ax.scatter(
            obj1[~is_pareto],
            obj2[~is_pareto],
            c="lightgray",
            s=50,
            alpha=0.6,
            label="Dominated",
        )

        # Plot Pareto front
        pareto_obj1 = obj1[is_pareto]
        pareto_obj2 = obj2[is_pareto]
        sorted_idx = np.argsort(pareto_obj1)
        ax.scatter(
            pareto_obj1,
            pareto_obj2,
            c="red",
            s=100,
            marker="*",
            label="Pareto Front",
            zorder=5,
        )
        ax.plot(
            pareto_obj1[sorted_idx],
            pareto_obj2[sorted_idx],
            "r--",
            alpha=0.5,
            linewidth=2,
        )
    else:
        ax.scatter(obj1, obj2, c="blue", s=50, alpha=0.7)

    ax.set_xlabel(obj1_name, fontsize=12)
    ax.set_ylabel(obj2_name, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_fitness_distribution(
    fitness_values: list[float],
    title: str = "Fitness Distribution",
    bins: int = 30,
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> "Figure":
    """Plot histogram of fitness values in a population.

    Args:
        fitness_values: List of fitness values
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size
        save_path: Save path
        show: Whether to display

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(fitness_values, bins=bins, color="steelblue", edgecolor="black", alpha=0.7)

    ax.axvline(
        np.mean(fitness_values),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(fitness_values):.4f}",
    )
    ax.axvline(
        np.max(fitness_values),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Max: {np.max(fitness_values):.4f}",
    )

    ax.set_xlabel("Fitness", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_gene_distribution(
    best_genes: np.ndarray,
    gene_names: list[str] | None = None,
    title: str = "Best Solution Gene Values",
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> "Figure":
    """Plot bar chart of gene values in the best solution.

    Args:
        best_genes: Array of gene values
        gene_names: Optional names for each gene
        title: Plot title
        figsize: Figure size
        save_path: Save path
        show: Whether to display

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    n_genes = len(best_genes)
    if gene_names is None:
        gene_names = [f"Gene {i}" for i in range(n_genes)]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_genes))
    bars = ax.bar(range(n_genes), best_genes, color=colors, edgecolor="black")

    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_names, rotation=45, ha="right")
    ax.set_xlabel("Gene", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_3d_cluster(
    positions: np.ndarray,
    n_atoms: int,
    title: str = "Atomic Cluster Configuration",
    atom_size: float = 500,
    figsize: tuple[float, float] = (10, 10),
    save_path: str | Path | None = None,
    show: bool = True,
    elev: float = 20,
    azim: float = 45,
) -> "Figure":
    """Plot 3D atomic cluster configuration.

    Args:
        positions: Flattened array of positions [x1,y1,z1,x2,y2,z2,...]
        n_atoms: Number of atoms
        title: Plot title
        atom_size: Size of atom markers
        figsize: Figure size
        save_path: Save path
        show: Whether to display
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view

    Returns:
        Figure object
    """
    _check_matplotlib()
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    coords = positions.reshape(n_atoms, 3)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot atoms
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=atom_size,
        c=range(n_atoms),
        cmap="coolwarm",
        edgecolors="black",
        linewidths=1,
    )

    # Draw bonds (connect atoms within distance threshold)
    bond_threshold = 1.5  # Typical LJ bond length
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < bond_threshold:
                ax.plot3D(
                    [coords[i, 0], coords[j, 0]],
                    [coords[i, 1], coords[j, 1]],
                    [coords[i, 2], coords[j, 2]],
                    "gray",
                    linewidth=2,
                    alpha=0.6,
                )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.view_init(elev=elev, azim=azim)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_thrust_profile(
    target_time: np.ndarray,
    target_thrust: np.ndarray,
    sim_time: np.ndarray | None = None,
    sim_thrust: np.ndarray | None = None,
    title: str = "Thrust Profile",
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> "Figure":
    """Plot target vs simulated thrust curves.

    Args:
        target_time: Target time array
        target_thrust: Target thrust values
        sim_time: Simulated time array (optional)
        sim_thrust: Simulated thrust values (optional)
        title: Plot title
        figsize: Figure size
        save_path: Save path
        show: Whether to display

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        target_time,
        target_thrust,
        "b-",
        linewidth=2,
        label="Target Profile",
    )

    if sim_time is not None and sim_thrust is not None:
        ax.plot(
            sim_time,
            sim_thrust,
            "r--",
            linewidth=2,
            label="Optimized Design",
        )
        ax.fill_between(
            target_time,
            target_thrust,
            np.interp(target_time, sim_time, sim_thrust),
            alpha=0.2,
            color="gray",
            label="Error Region",
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Thrust (N)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_composition_bar(
    elements: list[str],
    fractions: np.ndarray,
    title: str = "Alloy Composition",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> "Figure":
    """Plot bar chart of alloy composition.

    Args:
        elements: List of element symbols
        fractions: Atomic fractions (should sum to 1)
        title: Plot title
        figsize: Figure size
        save_path: Save path
        show: Whether to display

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Sort by fraction
    sorted_idx = np.argsort(fractions)[::-1]
    elements = [elements[i] for i in sorted_idx]
    fractions = fractions[sorted_idx]

    colors = plt.cm.Set3(np.linspace(0, 1, len(elements)))
    bars = ax.bar(elements, fractions * 100, color=colors, edgecolor="black")

    # Add value labels on bars
    for bar, frac in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{frac*100:.1f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Element", fontsize=12)
    ax.set_ylabel("Atomic %", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_properties_radar(
    properties: dict[str, float],
    targets: dict[str, float] | None = None,
    title: str = "Material Properties",
    figsize: tuple[float, float] = (10, 10),
    save_path: str | Path | None = None,
    show: bool = True,
) -> "Figure":
    """Plot radar/spider chart of material properties.

    Args:
        properties: Dict of property name -> value
        targets: Optional dict of target values for comparison
        title: Plot title
        figsize: Figure size
        save_path: Save path
        show: Whether to display

    Returns:
        Figure object
    """
    _check_matplotlib()

    labels = list(properties.keys())
    values = list(properties.values())
    n = len(labels)

    # Normalize values to 0-1 range for radar plot
    if targets:
        max_vals = [max(v, targets.get(k, v)) for k, v in properties.items()]
    else:
        max_vals = values
    normalized = [v / m if m > 0 else 0 for v, m in zip(values, max_vals)]

    # Compute angles
    angles = [i / float(n) * 2 * np.pi for i in range(n)]
    angles += angles[:1]  # Complete the loop
    normalized += normalized[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    ax.plot(angles, normalized, "b-", linewidth=2, label="Achieved")
    ax.fill(angles, normalized, alpha=0.25, color="blue")

    if targets:
        target_normalized = [targets.get(k, 0) / m if m > 0 else 0 for k, m in zip(labels, max_vals)]
        target_normalized += target_normalized[:1]
        ax.plot(angles, target_normalized, "r--", linewidth=2, label="Target")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_lattice_protein(
    sequence: str,
    positions: list[tuple[int, int]],
    title: str = "Protein Fold on 2D Lattice",
    figsize: tuple[float, float] = (10, 10),
    save_path: str | Path | None = None,
    show: bool = True,
) -> "Figure":
    """Plot 2D lattice protein fold.

    Args:
        sequence: HP sequence string
        positions: List of (x, y) positions for each amino acid
        title: Plot title
        figsize: Figure size
        save_path: Save path
        show: Whether to display

    Returns:
        Figure object
    """
    _check_matplotlib()

    if positions is None:
        raise ValueError("Invalid fold - positions is None")

    fig, ax = plt.subplots(figsize=figsize)

    # Draw backbone
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ax.plot(xs, ys, "k-", linewidth=2, zorder=1)

    # Draw amino acids
    for i, (x, y) in enumerate(positions):
        if sequence[i] == "H":
            color = "red"
            label = "Hydrophobic" if i == 0 or sequence[i - 1] != "H" else None
        else:
            color = "blue"
            label = "Polar" if i == 0 or sequence[i - 1] != "P" else None

        ax.scatter(x, y, s=500, c=color, edgecolors="black", linewidths=2, zorder=2, label=label)
        ax.annotate(
            sequence[i],
            (x, y),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )

    # Draw H-H contacts
    pos_to_idx = {pos: i for i, pos in enumerate(positions)}
    for i, (x, y) in enumerate(positions):
        if sequence[i] != "H":
            continue
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (x + dx, y + dy)
            if neighbor in pos_to_idx:
                j = pos_to_idx[neighbor]
                if abs(i - j) > 1 and sequence[j] == "H":
                    ax.plot(
                        [x, neighbor[0]],
                        [y, neighbor[1]],
                        "g--",
                        linewidth=3,
                        alpha=0.6,
                        zorder=0,
                    )

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def create_summary_figure(
    fitness_history: list[float],
    best_genes: np.ndarray,
    title: str = "Optimization Summary",
    gene_names: list[str] | None = None,
    figsize: tuple[float, float] = (14, 6),
    save_path: str | Path | None = None,
    show: bool = True,
) -> "Figure":
    """Create a summary figure with convergence and solution.

    Args:
        fitness_history: Fitness values over generations
        best_genes: Best solution found
        title: Overall figure title
        gene_names: Optional names for genes
        figsize: Figure size
        save_path: Save path
        show: Whether to display

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Convergence plot
    generations = range(len(fitness_history))
    ax1.plot(generations, fitness_history, "b-", linewidth=2)
    ax1.set_xlabel("Generation", fontsize=12)
    ax1.set_ylabel("Best Fitness", fontsize=12)
    ax1.set_title("Convergence", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Gene values
    n_genes = len(best_genes)
    if gene_names is None:
        gene_names = [f"x{i}" for i in range(n_genes)]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_genes))
    ax2.bar(range(n_genes), best_genes, color=colors, edgecolor="black")
    ax2.set_xticks(range(n_genes))
    ax2.set_xticklabels(gene_names, rotation=45, ha="right")
    ax2.set_xlabel("Gene", fontsize=12)
    ax2.set_ylabel("Value", fontsize=12)
    ax2.set_title("Best Solution", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
