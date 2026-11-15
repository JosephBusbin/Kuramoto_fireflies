from typing import Callable
import numpy as np
from scipy.integrate import solve_ivp
from manim import Axes, BLUE, Create, DashedLine, RED, Scene, Text, WHITE



# Exact PRC Helpers & Factory from peleg-lab/led_firesync prc_simulator.py
# Paper: https://www.biorxiv.org/content/10.64898/2026.01.19.700439v1.full
# github: https://github.com/peleg-lab/led_firesync/blob/master/simulation/prc_simulator.py
# This code was taken straight from thier github in order to replicate thier estimation of the phase response.
# It is thier original work and coding and is only being used for the purposes of replicating thier simulation and education.
# If they did not already have it coded, I would probably not have attempted this.

# I recommend to use a lower number oscillators for reasonable simulation time, N=100
# Because this simulates from finite oscilaors, you will see finite size effects in the resulting plot that results in some unsmoothness which is natural.

def wrap1(u: np.ndarray) -> np.ndarray:
    """Wrap values into the interval (-1, 1] via u - 2*floor((u + 1)/2)."""
    u_arr = np.asarray(u, dtype=float)
    return u_arr - 2.0 * np.floor((u_arr + 1.0) / 2.0)


def wrap_half(phi: np.ndarray) -> np.ndarray:
    """Half-wrap unit-cycle phase differences into (-0.5, 0.5]."""
    phi_arr = np.asarray(phi, dtype=float)
    return ((phi_arr + 0.5) % 1.0) - 0.5


def _alpha(kappa: float, phi_c: float, eps: float = 1e-15) -> float:
    """Compute the normalization factor 2 / (tanh(k*(0.5 - phi_c)) + tanh(k*(0.5 + phi_c)))."""
    t1 = np.tanh(kappa * (0.5 - phi_c))
    t2 = np.tanh(kappa * (0.5 + phi_c))
    denom = t1 + t2
    # Guard against division by zero if kappa approaches 0
    tiny = np.where(
        np.abs(denom) < eps,
        np.sign(denom) * eps + (denom == 0.0) * eps,
        denom,
    )
    return float(2.0 / tiny)


def R_tanh_step(
    phi: np.ndarray,
    Ymax: float,
    kappa: float,
    phi_c: float,
    y0: float,
    sgn: float = 1.0,
) -> np.ndarray:
    """Evaluate the measured period-ratio model R_tanh(phi_eval) from prc_simulator.py."""
    phi_eval = wrap_half(phi)
    a = _alpha(kappa, phi_c)
    core = sgn * a * (y0 + np.tanh(kappa * (phi_eval - phi_c)))
    return 1.0 + Ymax * wrap1(core)


def Z_from_R(
    phi: np.ndarray, R_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Convert the period ratio R(phi_eval) into the impulse PRC Z(phi_eval) = 1 - R(phi_eval)."""
    return 1.0 - R_func(phi)


def make_tanh_prc(
    Ymax: float = 0.25,
    kappa: float = 5.0,
    phi_c: float = 0.0,
    y0: float = 0.0,
    sgn: float = 1.0,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Factory returning (R_func, Z_func) callables consistent with Z = 1 - R."""
    def R(phi: np.ndarray) -> np.ndarray:
        return R_tanh_step(phi, Ymax=Ymax, kappa=kappa, phi_c=phi_c, y0=y0, sgn=sgn)

    def Z(phi: np.ndarray) -> np.ndarray:
        return Z_from_R(phi, R)

    return R, Z


# Coupled Oscillator ODE & K-Sweep Solver for N = 100 Oscillators

def kuramoto_firesync_ode(
    t: float,
    theta: np.ndarray,
    omega: np.ndarray,
    k_coupling: float,
    z_func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Compute d(theta)/dt for N oscillators coupled via the empirical Z(phi_eval) PRC."""
    # Compute pairwise unit-cycle phase differences: (theta_i - theta_j) / (2*pi)
    unit_phase_diff = (theta[:, None] - theta[None, :]) / (2.0 * np.pi)

    # Evaluate impulse PRC Z(phi_eval) across all N x N pairs and average over j (axis=1)
    z_matrix = z_func(unit_phase_diff)
    mean_z_impulse = np.mean(z_matrix, axis=1)

    # Multiply by 2*pi*K to convert unit-cycle phase shifts into radians/sec
    return omega + (2.0 * np.pi * k_coupling) * mean_z_impulse


def solve_firesync_ensemble(
    omega: np.ndarray,
    theta0: np.ndarray,
    k_coupling: float,
    z_func: Callable[[np.ndarray], np.ndarray],
    t_max: float = 30.0,
    num_steps: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the N=100 firefly ensemble for a single coupling strength K."""
    # Create the evaluation time grid and integrate via adaptive RK45
    t_eval = np.linspace(0.0, t_max, num_steps)
    solution = solve_ivp(
        fun=kuramoto_firesync_ode,
        t_span=(0.0, t_max),
        y0=theta0,
        t_eval=t_eval,
        args=(omega, k_coupling, z_func),
        method="RK45",
    )

    # Compute order parameter r(t) = sqrt(Rx^2 + Ry^2) across all N oscillators at each step
    rx = np.mean(np.cos(solution.y), axis=0)
    ry = np.mean(np.sin(solution.y), axis=0)
    r_trajectory = np.hypot(rx, ry)

    return solution.t, r_trajectory


def sweep_firesync_coupling(
    #Here we use only 100 oscillators 
    n_oscillators: int = 100,
    k_max: float = 4.0,
    num_k: int = 35,
    omega_std: float = 0.6,
    t_max: float = 30.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop across K values for N=100 oscillators and save the steady-state r values."""
    # Build the empirical PRC callable Z(phi) from the Peleg Lab factory function
    _, z_func = make_tanh_prc(Ymax=0.25, kappa=5.0, phi_c=0.0, y0=0.0, sgn=1.0)

    # Sample N=100 Gaussian frequencies (in the co-rotating mean=0 frame) and initial phases
    rng = np.random.default_rng(seed)
    omega = rng.normal(loc=0.0, scale=omega_std, size=n_oscillators)
    theta0 = rng.uniform(low=0.0, high=2.0 * np.pi, size=n_oscillators)

    # Prepare the K grid and output array for saved steady-state r values
    k_vals = np.linspace(0.0, k_max, num_k)
    r_steady = np.empty(num_k, dtype=float)

    # Solve the ensemble for each K and average the final 25% of r(t) to smooth finite-N noise
    for idx, k_val in enumerate(k_vals):
        _, r_trajectory = solve_firesync_ensemble(
            omega=omega,
            theta0=theta0,
            k_coupling=float(k_val),
            z_func=z_func,
            t_max=t_max,
            num_steps=80,
        )
        r_steady[idx] = float(np.mean(r_trajectory[-20:]))

    return k_vals, r_steady



# Construct the Manim Bifurcation Scene in Q1

class LedFiresyncBifurcation(Scene):
    """Animate the r vs. K transition for N=100 oscillators using prc_simulator.py."""

    def construct(self) -> None:
        # 1. Set sweep bounds for the N=100 empirical PRC ensemble
        k_max = 4.0
        num_k = 35

        # 2. Run the K sweep and collect the saved r_steady values
        k_vals, r_steady = sweep_firesync_coupling(
            n_oscillators=100,
            k_max=k_max,
            num_k=num_k,
            omega_std=0.6,
            t_max=30.0,
        )

        # 3. Locate the empirical critical threshold Kc where r rises past 0.25
        above_kc = np.where(r_steady > 0.25)[0]
        k_critical = float(k_vals[above_kc[0]]) if len(above_kc) > 0 else 1.0

        # 4. Configure first-quadrant axes identical to InfiniteKuramotoBifurcation
        axes = Axes(
            x_range=[0, k_max, 1],
            y_range=[0, 1.0, 0.2],
            x_length=9,
            y_length=5,
            axis_config={"include_numbers": True, "color": WHITE},
            tips=False,
        )
        labels = axes.get_axis_labels(x_label="K", y_label="r")

        # 5. Draw a dashed vertical reference line at the empirical Kc
        kc_bottom = axes.c2p(k_critical, 0.0)
        kc_top = axes.c2p(k_critical, 1.0)
        kc_line = DashedLine(kc_bottom, kc_top, color=RED)
        kc_label = Text("Kc", font_size=24, color=RED).next_to(kc_top, direction=[0, 1, 0])

        # 6. Plot the saved (k_vals, r_steady) curve and animate in the first quadrant
        bifurcation_graph = axes.plot_line_graph(
            x_values=k_vals,
            y_values=r_steady,
            line_color=BLUE,
            add_vertex_dots=False,
        )

        self.add(axes, labels, kc_line, kc_label)
        self.play(Create(bifurcation_graph), run_time=6.5)
        self.wait(3.0)