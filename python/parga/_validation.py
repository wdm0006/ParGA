from __future__ import annotations

import math
from collections.abc import Sequence


def validate_ga_config(
    *,
    genome_length: int,
    population_size: int,
    elitism: int,
    tournament_size: int,
    mutation_rate: float,
    crossover_rate: float,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    mutation_rate_end: float | None = None,
    random_immigrants: int | None = None,
) -> None:
    if population_size <= 0:
        raise ValueError("population size must be greater than zero")
    if genome_length <= 0:
        raise ValueError("genome_length must be greater than zero")
    if elitism < 0 or elitism > population_size:
        raise ValueError("elitism must be between 0 and population size")
    if tournament_size <= 0 or tournament_size > population_size:
        raise ValueError("tournament_size must be between 1 and population size")
    _validate_rate("mutation_rate", mutation_rate)
    _validate_rate("crossover_rate", crossover_rate)
    if mutation_rate_end is not None:
        _validate_rate("mutation_rate_end", mutation_rate_end)
    if random_immigrants is not None and not 0 <= random_immigrants <= population_size:
        raise ValueError("random_immigrants must be between 0 and population size")
    if len(lower_bounds) != genome_length:
        raise ValueError("lower_bounds length must match genome_length")
    if len(upper_bounds) != genome_length:
        raise ValueError("upper_bounds length must match genome_length")
    for index, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
        try:
            finite = math.isfinite(lower) and math.isfinite(upper)
        except TypeError as error:
            raise ValueError(f"bounds at index {index} must be numeric") from error
        if not finite:
            raise ValueError(f"bounds at index {index} must be finite")
        if lower > upper:
            raise ValueError(
                f"lower bound at index {index} must not exceed upper bound"
            )


def validate_island_config(
    *,
    num_islands: int,
    island_population: int,
    migration_interval: int,
    migration_count: int,
    **kwargs,
) -> None:
    if num_islands < 2:
        raise ValueError("num_islands must be at least 2 for migration")
    validate_ga_config(population_size=island_population, **kwargs)
    if migration_interval <= 0:
        raise ValueError("migration_interval must be greater than zero")
    if migration_count <= 0 or migration_count > island_population:
        raise ValueError(
            "migration_count must be between 1 and island_population"
        )


def _validate_rate(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
