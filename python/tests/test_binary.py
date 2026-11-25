"""Tests for the BinaryGA bindings."""

import itertools

import numpy as np
import pytest

from parga import BinaryCrossover, BinaryGA, BinaryMutation, SelectionMethod

VALUES = np.array([60, 100, 120, 80, 30, 90, 45, 70])
WEIGHTS = np.array([10, 20, 30, 15, 5, 25, 12, 18])
CAPACITY = 70
N = len(VALUES)


def knapsack(bits):
    if WEIGHTS[bits].sum() > CAPACITY:
        return -1.0
    return float(VALUES[bits].sum())


def brute_force_optimum():
    best = 0.0
    for combo in itertools.product([False, True], repeat=N):
        best = max(best, knapsack(np.array(combo)))
    return best


OPTIMUM = brute_force_optimum()


def onemax(bits):
    return float(bits.sum())


def raising_fitness(bits):
    raise ValueError("boom")


def nan_fitness(bits):
    return float("nan")


def inf_fitness(bits):
    return float("inf")


def not_a_number(bits):
    return "nope"


def run_knapsack(seed=42, **kwargs):
    return BinaryGA(
        knapsack,
        genome_length=N,
        population_size=60,
        generations=60,
        mutation_rate=0.1,
        seed=seed,
        **kwargs,
    ).run()


def test_finds_known_optimum():
    result = run_knapsack()
    assert result.best_fitness == OPTIMUM
    bits = result.best_bits()
    assert knapsack(bits) == OPTIMUM


def test_best_bits_shape_and_dtype():
    bits = run_knapsack().best_bits()
    assert bits.dtype == np.bool_
    assert bits.shape == (N,)


def test_callback_receives_bool_array():
    seen = []

    def fitness(bits):
        seen.append((bits.dtype, bits.shape))
        return float(bits.sum())

    BinaryGA(fitness, genome_length=7, population_size=10, generations=2, seed=1).run()
    assert seen
    assert all(dtype == np.bool_ and shape == (7,) for dtype, shape in seen)


def test_same_seed_reproduces_and_different_seed_differs():
    a = run_knapsack(seed=3)
    b = run_knapsack(seed=3)
    np.testing.assert_array_equal(a.best_bits(), b.best_bits())
    np.testing.assert_array_equal(a.fitness_history(), b.fitness_history())
    histories = {tuple(run_knapsack(seed=s).fitness_history()) for s in range(5)}
    assert len(histories) > 1


def test_history_length():
    result = run_knapsack()
    assert len(result.fitness_history()) == result.generations + 1


CROSSOVERS = {
    "single": BinaryCrossover.single_point,
    "two": BinaryCrossover.two_point,
    "uniform": BinaryCrossover.uniform,
}
MUTATIONS = {
    "bit_flip": BinaryMutation.bit_flip,
    "single": BinaryMutation.single_bit_flip,
}


@pytest.mark.parametrize("name", CROSSOVERS)
@pytest.mark.parametrize("mut", MUTATIONS)
def test_operators_run(name, mut):
    result = run_knapsack(crossover_method=CROSSOVERS[name](), mutation_method=MUTATIONS[mut]())
    assert result.best_bits().shape == (N,)
    assert result.best_fitness >= 0


def _history(**kwargs):
    return tuple(
        BinaryGA(
            onemax,
            genome_length=40,
            population_size=20,
            generations=15,
            mutation_rate=0.05,
            seed=11,
            **kwargs,
        )
        .run()
        .fitness_history()
    )


def test_crossover_choice_is_honored():
    # The engine silently falls back to two-point when handed the wrong variant,
    # so each explicit choice must change the seeded trajectory.
    two = _history(crossover_method=BinaryCrossover.two_point())
    assert two == _history()
    assert _history(crossover_method=BinaryCrossover.single_point()) != two
    assert _history(crossover_method=BinaryCrossover.uniform(0.5)) != two


def test_mutation_choice_is_honored():
    flip = _history(mutation_method=BinaryMutation.bit_flip())
    assert flip == _history()
    assert _history(mutation_method=BinaryMutation.single_bit_flip()) != flip


def test_set_selection():
    ga = BinaryGA(onemax, genome_length=10, population_size=20, generations=5, seed=1)
    ga.set_selection(SelectionMethod.rank())
    assert ga.run().best_bits().shape == (10,)


@pytest.mark.parametrize("fn", [raising_fitness, nan_fitness, inf_fitness, not_a_number])
def test_bad_callbacks_raise_runtime_error(fn):
    ga = BinaryGA(fn, genome_length=5, population_size=10, generations=2, seed=1)
    with pytest.raises(RuntimeError):
        ga.run()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"genome_length": 0},
        {"population_size": 0},
        {"mutation_rate": 1.5},
        {"crossover_rate": -0.1},
        {"elitism": 1000},
        {"tournament_size": 0},
    ],
)
def test_invalid_config_raises_value_error(kwargs):
    args = {"genome_length": 5, "population_size": 10, "generations": 2}
    args.update(kwargs)
    with pytest.raises(ValueError):
        BinaryGA(onemax, **args)


@pytest.mark.parametrize("prob", [-0.1, 1.1, float("nan")])
def test_invalid_uniform_prob(prob):
    with pytest.raises(ValueError):
        BinaryCrossover.uniform(prob)
