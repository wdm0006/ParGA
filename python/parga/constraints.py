"""Penalty-based constraint handling for fitness callbacks."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

_MAX_PENALTY = 1e300


def _check_positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number greater than zero, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number greater than zero, got {value!r}")
    return float(value)


class Penalized:
    """Picklable fitness wrapper that applies a linear penalty to constraint violations."""

    def __init__(
        self,
        fitness_fn: Callable[[Any], float],
        ineq: Sequence[Callable[[Any], float]],
        eq: Sequence[Callable[[Any], float]],
        penalty: float,
        eq_tol: float,
        sign: float,
    ) -> None:
        self.fitness_fn = fitness_fn
        self.ineq = ineq
        self.eq = eq
        self.penalty = penalty
        self.eq_tol = eq_tol
        self.sign = sign

    def __getstate__(self) -> dict[str, Any]:
        """Serialize each callback by value for spawned worker processes."""
        from .parallel import _dumps_fitness_fn

        state = self.__dict__.copy()
        state["fitness_fn"] = _dumps_fitness_fn(self.fitness_fn)
        state["ineq"] = tuple(_dumps_fitness_fn(g) for g in self.ineq)
        state["eq"] = tuple(_dumps_fitness_fn(h) for h in self.eq)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore callbacks without importing their original modules."""
        import cloudpickle

        state["fitness_fn"] = cloudpickle.loads(state["fitness_fn"])
        state["ineq"] = tuple(cloudpickle.loads(g) for g in state["ineq"])
        state["eq"] = tuple(cloudpickle.loads(h) for h in state["eq"])
        self.__dict__.update(state)

    def violation(self, x: Any) -> float:
        """Total constraint violation at ``x`` (0.0 when feasible)."""
        total = 0.0
        for g in self.ineq:
            total += self._excess(float(g(x)))
        for h in self.eq:
            total += self._excess(abs(float(h(x))) - self.eq_tol)
        return total

    @staticmethod
    def _excess(value: float) -> float:
        if math.isnan(value):
            return math.inf
        return max(0.0, value)

    def __call__(self, x: Any) -> float:
        """Return the penalized fitness at ``x``."""
        value = self.fitness_fn(x)
        v = self.violation(x)
        if v == 0.0:
            return value
        if not math.isfinite(v):
            return value - self.sign * _MAX_PENALTY
        return value - self.sign * min(self.penalty * v, _MAX_PENALTY)


def penalized(
    fitness_fn: Callable[[Any], float],
    *,
    ineq: Iterable[Callable[[Any], float]] = (),
    eq: Iterable[Callable[[Any], float]] = (),
    penalty: float = 1e6,
    eq_tol: float = 1e-6,
    mode: str = "maximize",
) -> Penalized:
    """Wrap ``fitness_fn`` so constraint violations are penalized with finite values.

    ``ineq`` callables ``g(x)`` are satisfied when ``g(x) <= 0``; ``eq`` callables
    ``h(x)`` when ``abs(h(x)) <= eq_tol``. The result equals ``fitness_fn(x)`` on
    feasible points; otherwise ``penalty * total_violation`` is subtracted
    (``mode="maximize"``) or added (``mode="minimize"``). Use ``mode="minimize"``
    with :func:`parga.minimize`, which expects the un-negated objective.
    """
    if mode not in ("maximize", "minimize"):
        raise ValueError(f"mode must be 'maximize' or 'minimize', got {mode!r}")
    return Penalized(
        fitness_fn,
        tuple(ineq),
        tuple(eq),
        _check_positive_finite("penalty", penalty),
        _check_positive_finite("eq_tol", eq_tol),
        1.0 if mode == "maximize" else -1.0,
    )
