import numpy as np
from scipy.integrate import solve_ivp
from manim import Axes, BLUE, Create, Scene, WHITE


def kuramoto_oa_ode(
    t: float, r: np.ndarray, k_coupling: float, gamma: float
) -> np.ndarray:
    """Right-hand side of the infinite-N Ott-Antonsen Kuramoto ODE."""
    # Evaluate dr/dt = (K/2 - gamma)*r - (K/2)*r^3
    return (0.5 * k_coupling - gamma) * r - 0.5 * k_coupling * (r**3)


def solve_order_parameter(
    k_coupling: float = 4.0,
    gamma: float = 1.0,
    r0: float = 0.05,
    t_max: float = 10.0,
    num_steps: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate r(t) over [0, t_max] on a discrete time grid."""
    # Generate evenly spaced evaluation time points and integrate via RK45
    t_eval = np.linspace(0.0, t_max, num_steps)
    solution = solve_ivp(
        fun=kuramoto_oa_ode,
        t_span=(0.0, t_max),
        y0=[r0],
        t_eval=t_eval,
        args=(k_coupling, gamma),
        method="RK45",
    )
    return solution.t, solution.y[0]


class InfiniteKuramotoOrderParameter(Scene): # Incoherent -> Coherent
    """Animate the time evolution of r(t) in the first quadrant. Starting from incoherence under high coupling."""
    # build the animation
    def construct(self) -> None:
        # set values for the solving of the order parameter and solve it
        k_coupling = 4.0
        gamma = 1.5
        r0 = 0.05
        t_max = 10.0

        t_vals, r_vals = solve_order_parameter( # returns a tuples of values for the t and r to plot
            k_coupling=k_coupling,
            gamma=gamma,
            r0=r0,
            t_max=t_max,
            num_steps=50,
        )

        # set the axis for our graph
        axes = Axes(
            x_range=[0, t_max, 2],
            y_range=[0, 1.0, 0.2],
            x_length=9,
            y_length=5,
            axis_config={"include_numbers": True, "color": WHITE},
            tips=False,
        )
        labels = axes.get_axis_labels(x_label="t", y_label="r(t)") #label the axes

        # instatiate the graph object add add out x,y and visual specificaitons
        graph = axes.plot_line_graph(
            x_values=t_vals,
            y_values=r_vals,
            line_color=BLUE,
            add_vertex_dots=False,
        )

        # add to the self animation object our axes, labels, and play the construction of the graph
        self.add(axes, labels)
        self.play(Create(graph), run_time=5.0)
        self.wait(2.0) #hold at the end for two seconds

class InfiniteKuramotoOrderParameterIncoherence(Scene): # Coherent -> Incoherent
    """Animate the time evolution of r(t) in the first quadrant. Starting from coherence under low coupling."""

    def construct(self) -> None:
        k_coupling = 0.5
        gamma = 1.5
        r0  = 0.95
        t_max = 10.0

        t_vals, r_vals = solve_order_parameter(
            k_coupling=k_coupling,
            gamma=gamma,
            r0=r0,
            t_max=t_max,
            num_steps=50
        )

        axes = Axes(
            x_range=[0, t_max, 2],
            y_range=[0, 1, 0.2],
            x_length=9,
            y_length=5,
            axis_config={"include_numbers": True, "color": WHITE},
            tips=False,
        )
        labels = axes.get_axis_labels(x_label="t", y_label="r(t)")

        graph = axes.plot_line_graph(
            x_values=t_vals,
            y_values=r_vals,
            line_color=BLUE,
            add_vertex_dots=False,
        )

        self.add(axes,labels)
        self.play(Create(graph), run_time=5.0)
        self.wait(2.0)