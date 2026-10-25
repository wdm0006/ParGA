"""Tests for parga.penalized."""

import math
import pickle

import numpy as np
import pytest

from parga import GA, is_free_threaded, maximize, minimize, penalized


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def neg_sphere(x):
    return -sphere(x)


def sum_at_least_two(x):
    return 2.0 - float(np.sum(x))


def sum_is_one(x):
    return float(np.sum(x)) - 1.0


def x0_is_half(x):
    return float(x[0]) - 0.5


def nan_constraint(x):
    return float("nan")


FEASIBLE = np.array([1.0, 1.5])
INFEASIBLE = np.array([0.5, 0.5])  # violates sum >= 2 by 1.0


@pytest.mark.parametrize("mode,fn", [("maximize", neg_sphere), ("minimize", sphere)])
def test_feasible_returns_exact_value(mode, fn):
    wrapped = penalized(fn, ineq=[sum_at_least_two], mode=mode)
    assert wrapped(FEASIBLE) == fn(FEASIBLE)


def test_infeasible_maximize_value():
    wrapped = penalized(neg_sphere, ineq=[sum_at_least_two], penalty=10.0)
    assert wrapped(INFEASIBLE) == pytest.approx(-0.5 - 10.0)


def test_infeasible_minimize_value():
    wrapped = penalized(sphere, ineq=[sum_at_least_two], penalty=10.0, mode="minimize")
    assert wrapped(INFEASIBLE) == pytest.approx(0.5 + 10.0)


def test_equality_tolerance_and_violation():
    wrapped = penalized(neg_sphere, eq=[sum_is_one], penalty=100.0, eq_tol=0.01)
    assert wrapped(np.array([0.5, 0.505])) == neg_sphere(np.array([0.5, 0.505]))
    x = np.array([1.0, 1.0])  # |h| = 1, violation 0.99
    assert wrapped(x) == pytest.approx(-2.0 - 100.0 * 0.99)


def test_violations_sum_across_constraints():
    wrapped = penalized(
        neg_sphere, ineq=[sum_at_least_two], eq=[x0_is_half], penalty=1.0, eq_tol=0.1
    )
    x = np.array([0.0, 0.0])  # ineq 2.0, eq max(0, 0.5 - 0.1) = 0.4
    assert wrapped(x) == pytest.approx(-0.0 - 2.4)


def test_penalty_is_finite_for_nonfinite_violation():
    nan_wrapped = penalized(neg_sphere, ineq=[nan_constraint])
    assert math.isfinite(nan_wrapped(FEASIBLE))
    assert nan_wrapped(FEASIBLE) < neg_sphere(FEASIBLE)
    inf_wrapped = penalized(neg_sphere, ineq=[lambda x: float("inf")])
    assert math.isfinite(inf_wrapped(FEASIBLE))
    assert inf_wrapped(FEASIBLE) < neg_sphere(FEASIBLE)


@pytest.mark.parametrize("bad", [0, -1.0, float("nan"), float("inf"), True, "1"])
def test_invalid_penalty(bad):
    with pytest.raises(ValueError, match="penalty"):
        penalized(neg_sphere, penalty=bad)


@pytest.mark.parametrize("bad", [0, -1e-3, float("nan"), float("inf"), True, "1"])
def test_invalid_eq_tol(bad):
    with pytest.raises(ValueError, match="eq_tol"):
        penalized(neg_sphere, eq_tol=bad)


def test_invalid_mode():
    with pytest.raises(ValueError, match="mode"):
        penalized(neg_sphere, mode="min")


def test_picklable():
    wrapped = penalized(neg_sphere, ineq=[sum_at_least_two], eq=[sum_is_one])
    clone = pickle.loads(pickle.dumps(wrapped))
    assert clone(INFEASIBLE) == wrapped(INFEASIBLE)


def test_minimize_end_to_end_feasible():
    wrapped = penalized(sphere, ineq=[sum_at_least_two], mode="minimize")
    result = minimize(
        wrapped,
        genome_length=2,
        bounds=(-5, 5),
        population_size=60,
        generations=80,
        parallel=False,
        seed=3,
    )
    genes = result.best_genes()
    assert genes.sum() >= 2.0 - 0.05
    assert sphere(genes) == pytest.approx(2.0, abs=0.1)  # optimum (1, 1)


def test_maximize_end_to_end_feasible():
    wrapped = penalized(neg_sphere, ineq=[sum_at_least_two])
    result = maximize(
        wrapped,
        genome_length=2,
        bounds=(-5, 5),
        population_size=60,
        generations=80,
        parallel=False,
        seed=3,
    )
    genes = result.best_genes()
    assert genes.sum() >= 2.0 - 0.05
    assert -sphere(genes) == pytest.approx(-2.0, abs=0.1)


@pytest.mark.skipif(is_free_threaded(), reason="free-threaded builds always use the Rust path")
def test_parallel_pickles_wrapper():
    wrapped = penalized(neg_sphere, ineq=[sum_at_least_two])
    result = GA(
        wrapped,
        genome_length=2,
        bounds=(-5, 5),
        population_size=8,
        generations=2,
        n_workers=1,
        parallel=True,
        seed=1,
    ).run()
    assert result.strategy == "parallel"
    assert math.isfinite(result.best_fitness)
