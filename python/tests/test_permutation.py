"""Tests for the PermutationGA bindings."""

import math

import numpy as np
import pytest

from parga import PermutationCrossover, PermutationGA, PermutationMutation, SelectionMethod

N = 10
ANGLES = 2 * np.pi * np.arange(N) / N
RING = np.column_stack([np.cos(ANGLES), np.sin(ANGLES)])
CITIES = RING[np.random.default_rng(0).permutation(N)]
OPTIMAL = N * 2 * math.sin(math.pi / N)


def tour_length(cities, order):
    path = cities[order]
    return float(np.linalg.norm(path - np.roll(path, -1, axis=0), axis=1).sum())


def tsp_fitness(order):
    return -tour_length(CITIES, order)


def raising_fitness(order):
    raise ValueError("boom")


def nan_fitness(order):
    return float("nan")


def inf_fitness(order):
    return float("inf")


def not_a_number(order):
    return "nope"


def run_tsp(seed=42, **kwargs):
    return PermutationGA(
        tsp_fitness, genome_length=N, population_size=100, generations=100, seed=seed, **kwargs
    ).run()


def assert_permutation(order, n):
    assert order.dtype == np.int64
    assert sorted(order.tolist()) == list(range(n))


CROSSOVERS = {
    "order": PermutationCrossover.order,
    "pmx": PermutationCrossover.partially_mapped,
    "cycle": PermutationCrossover.cycle,
    "edge": PermutationCrossover.edge_recombination,
}
MUTATIONS = {
    "swap": PermutationMutation.swap,
    "insert": PermutationMutation.insert,
    "inversion": PermutationMutation.inversion,
    "scramble": PermutationMutation.scramble,
}


class TestPermutationGA:
    def test_returns_valid_permutation_matching_fitness(self):
        result = run_tsp()
        order = result.best_order()
        assert_permutation(order, N)
        assert result.best_fitness == pytest.approx(-tour_length(CITIES, order))
        assert result.generations == 100
        assert len(result.fitness_history()) == 101

    def test_finds_known_optimal_tour(self):
        result = run_tsp(
            crossover_method=PermutationCrossover.order(),
            mutation_method=PermutationMutation.inversion(),
        )
        assert -result.best_fitness == pytest.approx(OPTIMAL, abs=1e-9)

    def test_tiny_instance_optimum(self):
        # Four corners of a unit square, labelled so the identity is the
        # crossing (length 2 + 2*sqrt(2)) tour; the optimum is the perimeter 4.
        square = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        result = PermutationGA(
            lambda o: -tour_length(square, o),
            genome_length=4,
            population_size=20,
            generations=30,
            seed=1,
        ).run()
        assert -result.best_fitness == pytest.approx(4.0)

    def test_seed_reproducible_and_seed_sensitive(self):
        a, b = run_tsp(seed=7), run_tsp(seed=7)
        np.testing.assert_array_equal(a.best_order(), b.best_order())
        np.testing.assert_array_equal(a.fitness_history(), b.fitness_history())
        other = run_tsp(seed=8)
        assert not np.array_equal(a.fitness_history(), other.fitness_history())

    @pytest.mark.parametrize("cx", CROSSOVERS)
    @pytest.mark.parametrize("mut", MUTATIONS)
    def test_every_operator_pair_gives_valid_permutation(self, cx, mut):
        result = run_tsp(crossover_method=CROSSOVERS[cx](), mutation_method=MUTATIONS[mut]())
        assert_permutation(result.best_order(), N)
        assert result.best_fitness == pytest.approx(-tour_length(CITIES, result.best_order()))

    def test_operator_choice_changes_the_run(self):
        histories = {
            name: tuple(run_tsp(mutation_method=make()).fitness_history()) for name, make in MUTATIONS.items()
        }
        assert len(set(histories.values())) > 1

    def test_crossover_choice_changes_the_run(self):
        histories = {name: tuple(run_tsp(crossover_method=make()).fitness_history()) for name, make in CROSSOVERS.items()}
        assert len(set(histories.values())) > 1

    def test_callback_receives_int64_permutations(self):
        seen = []

        def spy(order):
            seen.append((order.dtype, sorted(order.tolist())))
            return float(order[0])

        PermutationGA(spy, genome_length=5, population_size=10, generations=3, seed=0).run()
        assert seen
        assert all(dt == np.int64 and perm == list(range(5)) for dt, perm in seen)

    def test_set_selection(self):
        ga = PermutationGA(tsp_fitness, genome_length=N, population_size=30, generations=10, seed=3)
        ga.set_selection(SelectionMethod.rank())
        assert_permutation(ga.run().best_order(), N)

    def test_best_order_returns_copy(self):
        result = run_tsp()
        result.best_order()[:] = 0
        assert_permutation(result.best_order(), N)


class TestPermutationCallbackErrors:
    @pytest.mark.parametrize("fn", [raising_fitness, nan_fitness, inf_fitness, not_a_number])
    def test_bad_callback_raises_runtime_error(self, fn):
        ga = PermutationGA(fn, genome_length=5, population_size=10, generations=2, seed=0)
        with pytest.raises(RuntimeError, match="fitness function failed"):
            ga.run()

    def test_message_names_cause(self):
        with pytest.raises(RuntimeError, match="non-finite"):
            PermutationGA(nan_fitness, genome_length=5, population_size=10, generations=2).run()
        with pytest.raises(RuntimeError, match="boom"):
            PermutationGA(raising_fitness, genome_length=5, population_size=10, generations=2).run()


class TestPermutationValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"genome_length": 0},
            {"population_size": 0},
            {"mutation_rate": 1.5},
            {"mutation_rate": -0.1},
            {"mutation_rate": float("nan")},
            {"crossover_rate": 2.0},
            {"elitism": 101},
            {"tournament_size": 0},
        ],
    )
    def test_invalid_settings_raise_value_error(self, kwargs):
        args = {"genome_length": 5, **kwargs}
        with pytest.raises(ValueError):
            PermutationGA(tsp_fitness, **args)


def test_defaults_without_operator_arguments():
    result = PermutationGA(tsp_fitness, genome_length=N, population_size=20, generations=5, seed=0).run()
    assert_permutation(result.best_order(), N)
