"""Solid Rocket Motor Grain Optimization Case Study.

This example demonstrates using ParGA to optimize the grain geometry of a
solid rocket motor to achieve a desired thrust profile.

Background:
    Solid rocket motors burn propellant from the inside out. The shape of the
    internal cavity (grain geometry) determines how the burn area changes over
    time, which directly controls the thrust profile.

    This model uses a **segmented star grain**: the motor is divided into N
    axial segments, each with its own bore radius and star-slot depth. Because
    wider-bore segments burn through to the case sooner than narrow-bore ones,
    the total burn area -- and thus thrust -- changes over time in ways that
    can be shaped by the optimizer.

Physics Model:
    - Burn surface regresses uniformly at rate r = a * P_c^n
    - Chamber pressure from mass-flow equilibrium:
        P_c = (rho * a * A_b * c* / A_t)^(1/(1-n))
    - Thrust: F = C_F * P_c * A_t
    - Star perimeter per segment:
        arcs between slots + radial slot walls + slot bottom arcs

    Gene encoding (16 genes, all in [0, 1]):
        genes[0:6]   bore radius per segment  (mapped to 10--50 mm)
        genes[6:12]  slot depth per segment    (mapped to 0--25 mm)
        genes[12]    number of star slots      (mapped to 3--12, integer)
        genes[13]    slot angular width        (mapped to 0.05--0.40 rad)
        genes[14]    nozzle throat diameter    (mapped to 12--25 mm)
        genes[15]    burn rate coefficient     (mapped to 1.5e-5--6e-5, propellant choice)

References:
    - Sutton, G.P. "Rocket Propulsion Elements"
    - NASA Technical Reports on SRM optimization
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from parga import minimize

try:
    from parga.viz import plot_convergence_comparison, plot_thrust_profile

    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False


# ---------------------------------------------------------------------------
# Motor model
# ---------------------------------------------------------------------------

class SegmentedSRM:
    """Segmented solid rocket motor with axially varying star grain.

    The motor casing is divided into ``n_seg`` axial slices.  Each slice
    has its own initial bore radius and star-slot depth, while the number
    of star slots and their angular width are shared across all segments.

    Burn regression is uniform (all surfaces recede at the same rate),
    but segments with wider initial bores reach the case wall sooner,
    causing discrete drops in total burn area and therefore in thrust.
    """

    def __init__(
        self,
        n_seg: int = 6,
        case_radius: float = 0.075,       # m  (75 mm)
        total_length: float = 0.30,        # m  (300 mm)
        burn_rate_coeff: float = 3e-5,     # m/s / Pa^n
        burn_rate_exp: float = 0.35,
        propellant_density: float = 1750,  # kg/m^3
        c_star: float = 1550,              # m/s  (characteristic velocity)
        C_F: float = 1.3,                  # thrust coefficient
    ):
        self.n_seg = n_seg
        self.R_case = case_radius
        self.L_seg = total_length / n_seg
        self.a = burn_rate_coeff
        self.n = burn_rate_exp
        self.rho = propellant_density
        self.c_star = c_star
        self.C_F = C_F

    # -- geometry helpers (vectorised over segments) -------------------------

    @staticmethod
    def star_perimeters(
        bore_r: np.ndarray,
        n_slots: int,
        slot_depth: np.ndarray,
        slot_width: float,
    ) -> np.ndarray:
        """Inner perimeter of each segment's star-grain cross-section.

        The cross-section is a circle of radius *bore_r* with *n_slots*
        radial rectangular slots of depth *slot_depth* and angular width
        *slot_width*.  The perimeter consists of:

        1. Circular arcs at bore radius between the slots.
        2. Two radial walls per slot (length = slot_depth).
        3. Circumferential arcs at the slot-bottom radius.
        """
        has_slots = slot_depth > 0
        circular = 2 * np.pi * bore_r

        gap_angle = np.maximum(0.0, 2 * np.pi / max(n_slots, 1) - slot_width)
        arc_between = bore_r * gap_angle * n_slots
        walls = n_slots * 2 * slot_depth
        bottoms = (bore_r + slot_depth) * slot_width * n_slots

        return np.where(has_slots, arc_between + walls + bottoms, circular)

    # -- simulation ----------------------------------------------------------

    def simulate(
        self,
        bore_radii: np.ndarray,
        slot_depths: np.ndarray,
        n_slots: int,
        slot_width: float,
        throat_area: float,
        burn_rate_a: float | None = None,
        dt: float = 0.01,
        max_time: float = 8.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the motor burn and return (times, thrusts).

        All segments share the same chamber pressure (well-mixed assumption).
        Burn regression is tracked via a single scalar *web_burned* since all
        surfaces recede at the same rate.
        """
        bore_radii = np.asarray(bore_radii, dtype=float)
        slot_depths = np.asarray(slot_depths, dtype=float)
        # Clamp slots so they don't exceed the case
        slot_depths = np.minimum(slot_depths, np.maximum(0, self.R_case - bore_radii - 0.001))

        a = burn_rate_a if burn_rate_a is not None else self.a

        times: list[float] = []
        thrusts: list[float] = []

        web_burned = 0.0
        t = 0.0

        while t < max_time:
            cur_r = bore_radii + web_burned
            cur_d = np.maximum(0.0, slot_depths - web_burned)
            active = cur_r < self.R_case

            if not np.any(active):
                break

            perims = self.star_perimeters(cur_r, n_slots, cur_d, slot_width)
            total_Ab = np.sum(perims[active]) * self.L_seg

            if total_Ab < 1e-8:
                break

            # Equilibrium chamber pressure
            k = self.rho * a * total_Ab * self.c_star / throat_area
            P_c = k ** (1.0 / (1.0 - self.n))

            # Burn rate & thrust
            r_dot = a * P_c ** self.n
            F = self.C_F * P_c * throat_area

            times.append(t)
            thrusts.append(F)

            web_burned += r_dot * dt
            t += dt

            if len(times) > 50_000:
                break

        if not times:
            return np.zeros(1), np.zeros(1)
        return np.array(times), np.array(thrusts)


# ---------------------------------------------------------------------------
# Target thrust curves
# ---------------------------------------------------------------------------

def create_target_thrust_curve(
    profile: str = "neutral",
    duration: float = 4.0,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a target thrust curve within the motor's achievable envelope.

    Thrust levels are calibrated to the 75 mm case / 300 mm length motor.
    A cylindrical bore at ~4 s burn time produces ~2500–3500 N average,
    so targets are set in the 1500–5000 N range.
    """
    t = np.arange(0, duration, dt)

    if profile == "neutral":
        thrust = np.full_like(t, 3000.0)
    elif profile == "boost_sustain":
        # High initial thrust decaying smoothly — achievable via
        # differential segment bore radii causing early burnout.
        thrust = 2000.0 + 2500.0 * np.exp(-1.0 * t)
    elif profile == "progressive":
        thrust = 1500 + 2500 * (t / duration) ** 0.7
    elif profile == "regressive":
        thrust = 5000 * np.exp(-0.5 * t)
    else:
        thrust = np.full_like(t, 3000.0)

    return t, thrust


# ---------------------------------------------------------------------------
# Gene decoding + fitness
# ---------------------------------------------------------------------------

N_SEG = 6

# Gene ranges
BORE_MIN, BORE_MAX = 0.010, 0.050   # m
SLOT_MIN, SLOT_MAX = 0.000, 0.025   # m
NSLOTS_MIN, NSLOTS_MAX = 3, 12
WIDTH_MIN, WIDTH_MAX = 0.05, 0.40   # rad
THROAT_D_MIN, THROAT_D_MAX = 0.012, 0.025  # m
BURN_A_MIN, BURN_A_MAX = 1.5e-5, 6.0e-5   # m/s / Pa^n  (propellant choice)


def decode_genes(genes: np.ndarray) -> dict:
    """Map [0, 1] genes to physical motor parameters."""
    bore_radii = BORE_MIN + genes[0:N_SEG] * (BORE_MAX - BORE_MIN)
    slot_depths = SLOT_MIN + genes[N_SEG:2*N_SEG] * (SLOT_MAX - SLOT_MIN)
    n_slots = NSLOTS_MIN + int(genes[2*N_SEG] * (NSLOTS_MAX - NSLOTS_MIN))
    slot_width = WIDTH_MIN + genes[2*N_SEG + 1] * (WIDTH_MAX - WIDTH_MIN)
    throat_d = THROAT_D_MIN + genes[2*N_SEG + 2] * (THROAT_D_MAX - THROAT_D_MIN)
    throat_area = np.pi * (throat_d / 2) ** 2
    burn_rate_a = BURN_A_MIN + genes[2*N_SEG + 3] * (BURN_A_MAX - BURN_A_MIN)
    return dict(
        bore_radii=bore_radii,
        slot_depths=slot_depths,
        n_slots=int(n_slots),
        slot_width=float(slot_width),
        throat_area=float(throat_area),
        throat_d=float(throat_d),
        burn_rate_a=float(burn_rate_a),
    )


def thrust_curve_fitness(
    genes: np.ndarray,
    target_t: np.ndarray,
    target_thrust: np.ndarray,
    motor: SegmentedSRM,
) -> float:
    """RMS error between simulated and target thrust, normalised.

    The simulation is capped at the target duration so the optimizer
    only sees thrust within the desired window.
    """
    params = decode_genes(genes)

    try:
        sim_t, sim_thrust = motor.simulate(
            params["bore_radii"],
            params["slot_depths"],
            params["n_slots"],
            params["slot_width"],
            params["throat_area"],
            burn_rate_a=params["burn_rate_a"],
            max_time=target_t[-1] + 0.01,
        )
    except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
        return 1e6

    if len(sim_t) < 10:
        return 1e6

    # Interpolate onto target grid; zero thrust after burnout
    sim_interp = np.interp(target_t, sim_t, sim_thrust, left=0, right=0)

    # RMS error normalised by mean target thrust
    rms = np.sqrt(np.mean((sim_interp - target_thrust) ** 2))
    norm = np.mean(np.abs(target_thrust)) + 1.0
    return rms / norm


# ---------------------------------------------------------------------------
# Optimisation wrapper
# ---------------------------------------------------------------------------

def optimize_motor(
    target_profile: str = "neutral",
    population_size: int = 200,
    generations: int = 500,
    islands: int = 4,
    seed: int | None = None,
    verbose: bool = True,
) -> dict:
    """Optimise the segmented grain to match a target thrust profile."""
    motor = SegmentedSRM(n_seg=N_SEG)
    target_t, target_thrust = create_target_thrust_curve(target_profile, duration=4.0)

    def fitness(genes):
        return thrust_curve_fitness(genes, target_t, target_thrust, motor)

    n_genes = 2 * N_SEG + 4  # bore + slot per segment + n_slots + width + throat + burn_rate

    if verbose:
        print(f"\nOptimising for '{target_profile}' profile ({n_genes} genes, {N_SEG} segments)")
        print(f"  Target duration: {target_t[-1]:.1f} s")
        print(f"  Target peak thrust: {np.max(target_thrust):.0f} N")

    t0 = time.perf_counter()

    result = minimize(
        fitness,
        genome_length=n_genes,
        bounds=(0.0, 1.0),
        population_size=population_size,
        generations=generations,
        islands=islands,
        mutation_rate=0.08,
        crossover_rate=0.8,
        seed=seed,
        parallel=False,   # Rust strategy: SRM sim is ~30ms, too fast for multiprocess overhead
        verbose=False,
    )

    elapsed = time.perf_counter() - t0
    params = decode_genes(result.best_genes())
    sim_kwargs = dict(
        bore_radii=params["bore_radii"], slot_depths=params["slot_depths"],
        n_slots=params["n_slots"], slot_width=params["slot_width"],
        throat_area=params["throat_area"], burn_rate_a=params["burn_rate_a"],
    )

    # Full simulation for true burn time reporting
    sim_t_full, sim_thrust_full = motor.simulate(**sim_kwargs)
    # Capped simulation for plotting (matches the target window)
    target_duration = target_t[-1]
    sim_t, sim_thrust = motor.simulate(**sim_kwargs, max_time=target_duration + 0.01)

    # RMS error on the target window
    sim_interp = np.interp(target_t, sim_t, sim_thrust, left=0, right=0)
    rms = np.sqrt(np.mean((sim_interp - target_thrust) ** 2))

    if verbose:
        print(f"  Strategy: {result.strategy}")
        print(f"  Time: {elapsed:.1f} s")
        print(f"  RMS error: {rms:.0f} N  ({rms / np.mean(target_thrust) * 100:.1f}% of mean target)")
        print(f"  Optimised parameters:")
        print(f"    Bore radii (mm): {', '.join(f'{r*1000:.1f}' for r in params['bore_radii'])}")
        print(f"    Slot depths (mm): {', '.join(f'{d*1000:.1f}' for d in params['slot_depths'])}")
        print(f"    Star slots: {params['n_slots']}")
        print(f"    Slot width: {np.degrees(params['slot_width']):.1f}°")
        print(f"    Throat diameter: {params['throat_d']*1000:.1f} mm")
        print(f"    Burn rate coeff: {params['burn_rate_a']*1e5:.2f} × 10⁻⁵ m/s/Pa^n")
        print(f"  Simulated performance:")
        print(f"    Burn time: {sim_t_full[-1]:.2f} s")
        print(f"    Peak thrust: {np.max(sim_thrust_full):.0f} N")
        print(f"    Avg thrust: {np.mean(sim_thrust_full):.0f} N")

    return dict(
        profile=target_profile,
        params=params,
        target_t=target_t,
        target_thrust=target_thrust,
        sim_t=sim_t,
        sim_thrust=sim_thrust,
        rms_error=rms,
        fitness_history=result.fitness_history,
        strategy=result.strategy,
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# Benchmark & comparison runners
# ---------------------------------------------------------------------------

def run_benchmark(output_dir: Path | None = None):
    """Optimise for four different thrust profiles."""
    print("=" * 70)
    print("Solid Rocket Motor — Segmented Star-Grain Optimisation")
    print("=" * 70)
    print(f"\nMotor: {N_SEG}-segment star grain, 75 mm case radius, 300 mm length")
    print(f"Genes: {2*N_SEG+4} (bore radius + slot depth per segment + globals)")
    print()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    profiles = ["neutral", "boost_sustain", "progressive", "regressive"]
    all_histories = {}
    results_summary = []

    for profile in profiles:
        print(f"\n{'=' * 50}")
        print(f"Target: {profile.replace('_', ' ').title()}")
        print("=" * 50)

        result = optimize_motor(
            target_profile=profile,
            population_size=100,
            generations=250,
            islands=4,
            seed=42,
            verbose=True,
        )

        all_histories[profile] = result["fitness_history"]
        results_summary.append(
            (profile, result["rms_error"], result["elapsed"], result["strategy"])
        )

        if HAS_VIZ and output_dir:
            plot_thrust_profile(
                result["target_t"],
                result["target_thrust"],
                result["sim_t"],
                result["sim_thrust"],
                title=f"SRM: {profile.replace('_', ' ').title()} Profile",
                save_path=output_dir / f"srm_thrust_{profile}.png",
                show=False,
            )

    # Summary table
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Profile':20} {'RMS (N)':>10} {'RMS %':>8} {'Time (s)':>10} {'Strategy':>10}")
    print("-" * 60)
    for profile, rms, elapsed, strategy in results_summary:
        target_t, target_thrust = create_target_thrust_curve(profile)
        pct = rms / np.mean(target_thrust) * 100
        print(f"{profile:20} {rms:>10.0f} {pct:>7.1f}% {elapsed:>10.1f} {strategy:>10}")

    if HAS_VIZ and output_dir:
        plot_convergence_comparison(
            all_histories,
            title="SRM Optimisation Convergence",
            ylabel="Negative RMS Error",
            save_path=output_dir / "srm_convergence_comparison.png",
            show=False,
        )
        print(f"\nPlots saved to {output_dir}/")

    return results_summary


def compare_strategies(output_dir: Path | None = None):
    """Compare single GA vs island model on boost-sustain."""
    target_profile = "boost_sustain"

    print("=" * 70)
    print(f"Strategy Comparison: {target_profile.replace('_', ' ').title()} Profile")
    print("=" * 70)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Single Population GA ---")
    result_single = optimize_motor(
        target_profile=target_profile,
        population_size=80,
        generations=200,
        islands=1,
        seed=42,
        verbose=True,
    )

    print("\n--- Island Model (4 islands) ---")
    result_island = optimize_motor(
        target_profile=target_profile,
        population_size=80,
        generations=200,
        islands=4,
        seed=42,
        verbose=True,
    )

    print("\n--- Results ---")
    print(f"Single GA:    RMS = {result_single['rms_error']:.0f} N  in {result_single['elapsed']:.1f} s")
    print(f"Island Model: RMS = {result_island['rms_error']:.0f} N  in {result_island['elapsed']:.1f} s")

    if HAS_VIZ and output_dir:
        best = result_island if result_island["rms_error"] < result_single["rms_error"] else result_single
        plot_thrust_profile(
            best["target_t"], best["target_thrust"],
            best["sim_t"], best["sim_thrust"],
            title=f"Best SRM Design: {target_profile.replace('_', ' ').title()}",
            save_path=output_dir / "srm_best_thrust.png",
            show=False,
        )
        plot_convergence_comparison(
            {
                "Single GA": result_single["fitness_history"],
                "Island Model (4)": result_island["fitness_history"],
            },
            title="SRM: Single GA vs Island Model",
            save_path=output_dir / "srm_strategy_comparison.png",
            show=False,
        )
        print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    import sys

    output_dir = Path(__file__).parent / "output"

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_strategies(output_dir)
    else:
        run_benchmark(output_dir)
