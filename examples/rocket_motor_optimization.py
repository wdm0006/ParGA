"""Solid Rocket Motor Grain Optimization Case Study.

This example demonstrates using ParGA to optimize the grain geometry of a
solid rocket motor to achieve a desired thrust profile.

Background:
    Solid rocket motors burn propellant from the inside out. The shape of the
    internal cavity (grain geometry) determines how the burn area changes over
    time, which directly affects the thrust profile.

    Common grain geometries include:
    - Cylindrical bore (neutral burn - constant thrust)
    - Star pattern (progressive then regressive)
    - Finocyl (fin + cylinder - can achieve various profiles)

This simplified model optimizes a parameterized star grain cross-section to match
a target thrust curve, similar to real aerospace engineering problems.

Physics Model:
    - Thrust F = P_c * A_t * C_F (chamber pressure * throat area * thrust coeff)
    - Burn rate r = a * P_c^n (Saint-Venant-Wantzel equation)
    - Chamber pressure depends on burn surface area

References:
    - Sutton, G.P. "Rocket Propulsion Elements"
    - NASA Technical Reports on SRM optimization
    - openMotor: https://github.com/reilleya/openMotor
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from parga import minimize

# Try to import visualization - optional dependency
try:
    from parga.viz import plot_convergence_comparison, plot_thrust_profile

    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


class SimplifiedSRM:
    """Simplified Solid Rocket Motor model.

    This model uses a parameterized star grain geometry and simulates
    the burn progression to compute thrust over time.
    """

    def __init__(
        self,
        outer_radius: float = 0.075,  # Motor case radius (m)
        length: float = 0.25,  # Grain length (m)
        throat_area: float = 0.0004,  # Nozzle throat area (m^2)
        burn_rate_coeff: float = 3e-5,  # Burn rate coefficient (m/s/Pa^n)
        burn_rate_exp: float = 0.35,  # Burn rate pressure exponent
        propellant_density: float = 1750,  # kg/m^3
        characteristic_velocity: float = 1550,  # m/s
        thrust_coefficient: float = 1.3,  # Dimensionless
    ):
        self.R_outer = outer_radius
        self.L = length
        self.A_t = throat_area
        self.a = burn_rate_coeff
        self.n = burn_rate_exp
        self.rho = propellant_density
        self.c_star = characteristic_velocity
        self.C_F = thrust_coefficient

    def compute_star_geometry(
        self,
        core_radius: float,
        num_points: int,
        point_depth: float,
        point_width: float,
    ) -> tuple[float, float]:
        """Compute burn area and port area for star grain.

        Args:
            core_radius: Inner radius of the core (m)
            num_points: Number of star points
            point_depth: Depth of star points (m)
            point_width: Angular width of points (radians)

        Returns:
            Tuple of (burn_perimeter, port_area)
        """
        if num_points < 3:
            num_points = 3

        # Simplified star geometry calculation
        # Each point adds perimeter and modifies area

        # Base circle perimeter and area
        base_perimeter = 2 * np.pi * core_radius
        base_area = np.pi * core_radius**2

        # Star point contribution (simplified triangular points)
        point_perimeter = 2 * point_depth / np.cos(point_width / 2)
        point_area = 0.5 * point_depth * 2 * point_depth * np.tan(point_width / 2)

        total_perimeter = base_perimeter + num_points * point_perimeter
        total_area = base_area + num_points * point_area

        return total_perimeter, total_area

    def simulate_burn(
        self,
        core_radius: float,
        num_points: int,
        point_depth: float,
        point_width: float,
        dt: float = 0.01,
        max_time: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate motor burn and return thrust profile.

        Args:
            core_radius: Initial core radius
            num_points: Number of star points
            point_depth: Initial point depth
            point_width: Point angular width
            dt: Time step (s)
            max_time: Maximum simulation time (s)

        Returns:
            Tuple of (time_array, thrust_array)
        """
        times = []
        thrusts = []

        t = 0
        current_radius = core_radius
        current_depth = point_depth

        while t < max_time and current_radius < self.R_outer:
            # Get current geometry
            perimeter, port_area = self.compute_star_geometry(
                current_radius, int(num_points), current_depth, point_width
            )

            # Burn surface area
            A_b = perimeter * self.L

            # Chamber pressure from equilibrium (simplified)
            # mdot = rho * r * A_b = P_c * A_t / c_star
            # P_c = (rho * a * A_b * c_star / A_t)^(1/(1-n))
            k = self.rho * self.a * A_b * self.c_star / self.A_t
            if k > 0:
                P_c = k ** (1 / (1 - self.n))
            else:
                P_c = 0

            # Burn rate
            r = self.a * (P_c**self.n) if P_c > 0 else 0

            # Thrust
            F = P_c * self.A_t * self.C_F

            times.append(t)
            thrusts.append(F)

            # Update geometry (simplified - uniform regression)
            current_radius += r * dt
            current_depth = max(0, current_depth - r * dt * 0.5)  # Points erode

            t += dt

            # Safety check
            if len(times) > 10000:
                break

        return np.array(times), np.array(thrusts)


def create_target_thrust_curve(
    profile: str = "boost_sustain",
    duration: float = 3.0,
    peak_thrust: float = 5000,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a target thrust curve to match.

    Args:
        profile: Type of thrust profile
        duration: Total burn duration (s)
        peak_thrust: Maximum thrust (N)
        dt: Time step (s)

    Returns:
        Tuple of (time_array, thrust_array)
    """
    t = np.arange(0, duration, dt)

    if profile == "boost_sustain":
        # High initial thrust, then sustain
        thrust = np.where(t < 0.5, peak_thrust, peak_thrust * 0.4)
    elif profile == "progressive":
        # Gradually increasing thrust
        thrust = peak_thrust * (t / duration) ** 0.5
    elif profile == "regressive":
        # Gradually decreasing thrust
        thrust = peak_thrust * (1 - 0.7 * t / duration)
    elif profile == "neutral":
        # Constant thrust
        thrust = np.ones_like(t) * peak_thrust * 0.6
    else:
        thrust = np.ones_like(t) * peak_thrust * 0.5

    return t, thrust


def thrust_curve_fitness(genes: np.ndarray, target_t: np.ndarray, target_thrust: np.ndarray) -> float:
    """Fitness function comparing simulated vs target thrust curve.

    Args:
        genes: [core_radius, num_points, point_depth, point_width]
        target_t: Target time array
        target_thrust: Target thrust array

    Returns:
        Negative RMS error (higher is better for GA)
    """
    # Decode genes to motor parameters (scaled for smaller motor)
    core_radius = 0.008 + genes[0] * 0.032  # 0.008 to 0.04 m (8-40 mm)
    num_points = 3 + int(genes[1] * 7)  # 3 to 10 points
    point_depth = 0.003 + genes[2] * 0.017  # 0.003 to 0.02 m (3-20 mm)
    point_width = 0.1 + genes[3] * 0.4  # 0.1 to 0.5 radians (~6-29 degrees)

    # Create motor and simulate
    motor = SimplifiedSRM()

    try:
        sim_t, sim_thrust = motor.simulate_burn(
            core_radius=core_radius,
            num_points=num_points,
            point_depth=point_depth,
            point_width=point_width,
            dt=0.02,
            max_time=target_t[-1] + 0.5,
        )
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return -1e6  # Invalid configuration

    if len(sim_t) < 10:
        return -1e6  # Burn too short

    # Interpolate simulated thrust to target time points
    sim_thrust_interp = np.interp(target_t, sim_t, sim_thrust, left=0, right=0)

    # Compute RMS error
    rms_error = np.sqrt(np.mean((sim_thrust_interp - target_thrust) ** 2))

    # Normalize by target thrust magnitude
    rms_normalized = rms_error / (np.mean(target_thrust) + 1)

    return -rms_normalized  # Negative because we maximize fitness


def optimize_motor(
    target_profile: str = "boost_sustain",
    population_size: int = 100,
    generations: int = 200,
    islands: int = 1,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Optimize motor grain geometry for target thrust profile.

    Args:
        target_profile: Target thrust profile type
        population_size: GA population size
        generations: Number of generations
        islands: Number of islands
        seed: Random seed
        verbose: Print progress

    Returns:
        Dictionary with optimization results
    """
    # Create target curve
    target_t, target_thrust = create_target_thrust_curve(
        profile=target_profile, duration=3.0, peak_thrust=5000
    )

    def fitness(genes):
        return -thrust_curve_fitness(genes, target_t, target_thrust)

    if verbose:
        print(f"\nOptimizing motor for '{target_profile}' thrust profile")
        print(f"  Target duration: {target_t[-1]:.1f}s")
        print(f"  Peak thrust: {np.max(target_thrust):.0f}N")

    start_time = time.perf_counter()

    result = minimize(
        fitness,
        genome_length=4,
        bounds=(0.0, 1.0),
        population_size=population_size,
        generations=generations,
        islands=islands,
        mutation_rate=0.05,
        crossover_rate=0.8,
        seed=seed,
        verbose=False,
    )

    elapsed = time.perf_counter() - start_time

    # Decode best solution (matching the gene decoding in fitness function)
    genes = result.best_genes()
    best_params = {
        "core_radius": 0.008 + genes[0] * 0.032,
        "num_points": 3 + int(genes[1] * 7),
        "point_depth": 0.003 + genes[2] * 0.017,
        "point_width": 0.1 + genes[3] * 0.4,
    }

    # Get final thrust curve
    motor = SimplifiedSRM()
    sim_t, sim_thrust = motor.simulate_burn(**best_params, dt=0.02, max_time=4.0)

    if verbose:
        print(f"  Strategy: {result.strategy}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"\n  Optimized parameters:")
        print(f"    Core radius: {best_params['core_radius']*1000:.1f} mm")
        print(f"    Star points: {best_params['num_points']}")
        print(f"    Point depth: {best_params['point_depth']*1000:.1f} mm")
        print(f"    Point width: {np.degrees(best_params['point_width']):.1f} degrees")
        print(f"\n  Simulated performance:")
        print(f"    Burn time: {sim_t[-1]:.2f}s")
        print(f"    Peak thrust: {np.max(sim_thrust):.0f}N")
        print(f"    Avg thrust: {np.mean(sim_thrust):.0f}N")

    return {
        "profile": target_profile,
        "params": best_params,
        "target_t": target_t,
        "target_thrust": target_thrust,
        "sim_t": sim_t,
        "sim_thrust": sim_thrust,
        "fitness_history": result.fitness_history,
        "strategy": result.strategy,
        "elapsed": elapsed,
    }


def run_benchmark(output_dir: Path | None = None):
    """Run optimization for various thrust profiles."""
    print("=" * 70)
    print("Solid Rocket Motor Grain Optimization")
    print("=" * 70)
    print("\nOptimizing star grain geometry to match target thrust profiles.")
    print("This simulates aerospace engineering design optimization.")
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    profiles = ["neutral", "boost_sustain", "progressive", "regressive"]
    all_histories = {}

    for profile in profiles:
        print(f"\n{'='*50}")
        print(f"Target Profile: {profile}")
        print("=" * 50)

        result = optimize_motor(
            target_profile=profile,
            population_size=100,
            generations=150,
            seed=42,
            verbose=True,
        )

        all_histories[profile] = result["fitness_history"]

        # Save thrust profile plot
        if HAS_VIZ and output_dir:
            plot_thrust_profile(
                result["target_t"],
                result["target_thrust"],
                result["sim_t"],
                result["sim_thrust"],
                title=f"Motor Optimization: {profile.replace('_', ' ').title()} Profile",
                save_path=output_dir / f"srm_thrust_{profile}.png",
                show=False,
            )

    # Convergence comparison
    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            all_histories,
            title="SRM Optimization Convergence by Profile",
            ylabel="Negative RMS Error",
            save_path=output_dir / "srm_convergence_comparison.png",
            show=False,
        )
        print(f"\nPlots saved to {output_dir}/")


def compare_strategies(output_dir: Path | None = None):
    """Compare single GA vs island model for motor optimization."""
    target_profile = "boost_sustain"

    print("=" * 70)
    print(f"Strategy Comparison: {target_profile} Profile")
    print("=" * 70)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Single population
    print("\n--- Single Population GA ---")
    result_single = optimize_motor(
        target_profile=target_profile,
        population_size=150,
        generations=300,
        islands=1,
        seed=42,
        verbose=True,
    )

    # Island model
    print("\n--- Island Model (4 islands) ---")
    result_island = optimize_motor(
        target_profile=target_profile,
        population_size=150,
        generations=300,
        islands=4,
        seed=42,
        verbose=True,
    )

    print("\n--- Results ---")
    print(f"Single GA:    fitness={result_single['fitness_history'][-1]:.6f} in {result_single['elapsed']:.2f}s")
    print(f"Island Model: fitness={result_island['fitness_history'][-1]:.6f} in {result_island['elapsed']:.2f}s")

    # Comparison plots
    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            {
                "Single GA": result_single["fitness_history"],
                "Island Model (4)": result_island["fitness_history"],
            },
            title="SRM Optimization: Single GA vs Island Model",
            save_path=output_dir / "srm_strategy_comparison.png",
            show=False,
        )

        # Save best result thrust profile
        best_result = result_island if result_island["fitness_history"][-1] > result_single["fitness_history"][-1] else result_single
        plot_thrust_profile(
            best_result["target_t"],
            best_result["target_thrust"],
            best_result["sim_t"],
            best_result["sim_thrust"],
            title="Best Motor Design: Boost-Sustain Profile",
            save_path=output_dir / "srm_best_thrust.png",
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
