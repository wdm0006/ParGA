"""Smoke tests for the parga.viz plotting entry points."""

import inspect
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from parga import viz  # noqa: E402

HISTORY = [5.0, 3.0, 2.0, 1.5, 1.0]
GENES = np.array([0.5, -1.0, 2.0])
POSITIONS_3D = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
TIME = np.linspace(0.0, 1.0, 20)

# Minimal valid (args, kwargs) per public entry point.
CASES = {
    "plot_convergence": ((HISTORY,), {}),
    "plot_convergence_comparison": (({"a": HISTORY, "b": HISTORY[::-1]},), {}),
    "plot_2d_landscape": ((lambda x: float(np.sum(x**2)), (-1.0, 1.0)), {"resolution": 10}),
    "plot_parameter_sensitivity": (([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], "mutation_rate"), {}),
    "plot_population_diversity": (([np.random.default_rng(0).random((8, 3)) for _ in range(3)],), {}),
    "plot_pareto_front": (([1.0, 2.0, 3.0, 4.0], [4.0, 2.5, 3.0, 1.0]), {}),
    "plot_fitness_distribution": ((list(np.linspace(0.0, 1.0, 50)),), {}),
    "plot_gene_distribution": ((GENES,), {}),
    "plot_3d_cluster": ((POSITIONS_3D, 4), {}),
    "plot_thrust_profile": ((TIME, np.sin(TIME)), {"sim_time": TIME, "sim_thrust": np.cos(TIME)}),
    "plot_composition_bar": ((["Fe", "Ni", "Cr"], np.array([0.5, 0.3, 0.2])), {}),
    "plot_properties_radar": (({"a": 0.5, "b": 0.7, "c": 0.2},), {"targets": {"a": 0.6, "b": 0.6, "c": 0.3}}),
    "plot_lattice_protein": (("HPHP", [(0, 0), (1, 0), (1, 1), (0, 1)]), {}),
    "create_summary_figure": ((HISTORY, GENES), {}),
}


def public_entry_points():
    return sorted(
        name
        for name, obj in vars(viz).items()
        if inspect.isfunction(obj) and obj.__module__ == viz.__name__ and not name.startswith("_")
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_every_public_function_has_a_case():
    assert public_entry_points() == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_entry_point_returns_figure(name):
    args, kwargs = CASES[name]
    fig = getattr(viz, name)(*args, show=False, **kwargs)
    assert isinstance(fig, Figure)


def test_convergence_plots_supplied_history():
    fig = viz.plot_convergence(HISTORY, show=False)
    (line,) = fig.axes[0].get_lines()
    np.testing.assert_array_equal(line.get_ydata(), HISTORY)
    np.testing.assert_array_equal(line.get_xdata(), range(len(HISTORY)))


def test_convergence_comparison_plots_each_history():
    histories = {"a": HISTORY, "b": HISTORY[::-1]}
    fig = viz.plot_convergence_comparison(histories, show=False)
    plotted = {line.get_label(): list(line.get_ydata()) for line in fig.axes[0].get_lines()}
    assert plotted == histories


def test_composition_bar_heights_are_percentages():
    fractions = np.array([0.5, 0.3, 0.2])
    fig = viz.plot_composition_bar(["Fe", "Ni", "Cr"], fractions, show=False)
    heights = sorted(p.get_height() for p in fig.axes[0].patches)
    np.testing.assert_allclose(heights, sorted(fractions * 100))


def test_convergence_draws_on_supplied_axes():
    _, ax = plt.subplots()
    assert viz.plot_convergence(HISTORY, ax=ax, show=False) is None
    np.testing.assert_array_equal(ax.get_lines()[0].get_ydata(), HISTORY)


@pytest.mark.parametrize("name", sorted(CASES))
def test_missing_matplotlib_gives_guided_error(monkeypatch, name):
    monkeypatch.setattr(viz, "HAS_MATPLOTLIB", False)
    args, kwargs = CASES[name]
    with pytest.raises(ImportError, match=r"parga\[viz\]"):
        getattr(viz, name)(*args, show=False, **kwargs)
