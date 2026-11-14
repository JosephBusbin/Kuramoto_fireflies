import numpy as np
from scipy.integrate import solve_ivp
from manim import Axes, BLUE, Create, DashedLine, RED, Scene, Text, WHITE
#import the model and solver from the order parameter simulation
from kuramoto_ord_par_sim import kuramoto_oa_ode, solve_order_parameter

def sweep_coupling_strength(
    k_max: float = 6.0,
    num_k: int = 80,
    gamma: float = 1.0,
    r0: float = 0.01,
    t_max: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep across K values and record the converged steady-state r for each K."""
    # Create the horizontal axis grid of coupling strengths K
    k_vals = np.linspace(0.0, k_max, num_k)
    r_steady = np.empty(num_k, dtype=float)

    # Integrate each K to t_max and store only the final converged value r(t_max)
    for idx, k_val in enumerate(k_vals):
        _, r_trajectory = solve_order_parameter( # Here we use tuple unpacking '_,'
            k_coupling=float(k_val),
            gamma=gamma,
            r0=r0,
            t_max=t_max,
            num_steps=2,
        )
        r_steady[idx] = r_trajectory[-1]

    return k_vals, r_steady


class InfiniteKuramotoBifurcation(Scene):
    """Animate the steady-state order parameter r as a function of coupling K."""

    def construct(self) -> None:
        # 1. Define model and sweep parameters (for gamma = 1.0, Kc = 2*gamma = 2.0)
        gamma = 1.0
        k_critical = 2.0 * gamma
        k_max = 6.0

        # 2. Compute the steady-state bifurcation curve (K, r_infinity)
        k_vals, r_steady = sweep_coupling_strength(
            k_max=k_max,
            num_k=80,
            gamma=gamma,
            r0=0.01,
            t_max=120.0,
        )

        # 3. Set up first-quadrant axes with K on the horizontal axis and r on the vertical
        axes = Axes(
            x_range=[0, k_max, 1],
            y_range=[0, 1.0, 0.2],
            x_length=9,
            y_length=5,
            axis_config={"include_numbers": True, "color": WHITE},
            tips=False,
        )
        labels = axes.get_axis_labels(x_label="K", y_label="r")

        # 4. Create a dashed vertical marker at the critical threshold Kc = 2*gamma
        kc_bottom = axes.c2p(k_critical, 0.0)
        kc_top = axes.c2p(k_critical, 1.0)
        kc_line = DashedLine(kc_bottom, kc_top, color=RED)
        kc_label = Text("Kc", font_size=24, color=RED).next_to(kc_top, direction=[0, 1, 0])

        # 5. Plot the numerically converged r(K) curve across the bifurcation
        bifurcation_graph = axes.plot_line_graph(
            x_values=k_vals,
            y_values=r_steady,
            line_color=BLUE,
            add_vertex_dots=False,
        )

        # 6. Render static axes and Kc marker, then animate the bifurcation curve
        self.add(axes, labels, kc_line, kc_label)
        self.play(Create(bifurcation_graph), run_time=6.5)
        self.wait(3.0)