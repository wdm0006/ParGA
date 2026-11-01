"""Introspection of the operator wrappers: kind, params, repr."""

import pytest

from parga import CrossoverMethod, MutationMethod, SelectionMethod

CASES = [
    (SelectionMethod.tournament(), "tournament", {"size": 3.0}, "SelectionMethod.tournament(size=3.0)"),
    (SelectionMethod.tournament(5), "tournament", {"size": 5.0}, "SelectionMethod.tournament(size=5.0)"),
    (SelectionMethod.roulette(), "roulette", {}, "SelectionMethod.roulette()"),
    (SelectionMethod.rank(), "rank", {}, "SelectionMethod.rank()"),
    (SelectionMethod.random(), "random", {}, "SelectionMethod.random()"),
    (SelectionMethod.truncation(), "truncation", {"ratio": 0.5}, "SelectionMethod.truncation(ratio=0.5)"),
    (SelectionMethod.truncation(0.25), "truncation", {"ratio": 0.25}, "SelectionMethod.truncation(ratio=0.25)"),
    (SelectionMethod.stochastic_universal(), "stochastic_universal", {}, "SelectionMethod.stochastic_universal()"),
    (CrossoverMethod.single_point(), "single_point", {}, "CrossoverMethod.single_point()"),
    (CrossoverMethod.two_point(), "two_point", {}, "CrossoverMethod.two_point()"),
    (CrossoverMethod.uniform(), "uniform", {"probability": 0.5}, "CrossoverMethod.uniform(probability=0.5)"),
    (CrossoverMethod.uniform(0.3), "uniform", {"probability": 0.3}, "CrossoverMethod.uniform(probability=0.3)"),
    (CrossoverMethod.blend(), "blend", {"alpha": 0.5}, "CrossoverMethod.blend(alpha=0.5)"),
    (CrossoverMethod.blend(0.2), "blend", {"alpha": 0.2}, "CrossoverMethod.blend(alpha=0.2)"),
    (CrossoverMethod.simulated_binary(), "simulated_binary", {"eta": 20.0}, "CrossoverMethod.simulated_binary(eta=20.0)"),
    (CrossoverMethod.simulated_binary(2.5), "simulated_binary", {"eta": 2.5}, "CrossoverMethod.simulated_binary(eta=2.5)"),
    (CrossoverMethod.arithmetic(), "arithmetic", {}, "CrossoverMethod.arithmetic()"),
    (MutationMethod.gaussian(), "gaussian", {"sigma": 0.1}, "MutationMethod.gaussian(sigma=0.1)"),
    (MutationMethod.gaussian(0.4), "gaussian", {"sigma": 0.4}, "MutationMethod.gaussian(sigma=0.4)"),
    (MutationMethod.uniform(), "uniform", {}, "MutationMethod.uniform()"),
    (MutationMethod.polynomial(), "polynomial", {"eta": 20.0}, "MutationMethod.polynomial(eta=20.0)"),
    (MutationMethod.polynomial(7.0), "polynomial", {"eta": 7.0}, "MutationMethod.polynomial(eta=7.0)"),
    (MutationMethod.boundary(), "boundary", {}, "MutationMethod.boundary()"),
]


@pytest.mark.parametrize(("method", "kind", "params", "text"), CASES)
def test_kind_params_repr(method, kind, params, text):
    assert method.kind == kind
    assert method.params == params
    assert repr(method) == text


def test_params_returns_fresh_dict():
    method = MutationMethod.gaussian(0.2)
    method.params["sigma"] = 99.0
    assert method.params == {"sigma": 0.2}
