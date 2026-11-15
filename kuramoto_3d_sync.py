import numpy as np
from scipy.integrate import solve_ivp
from manim import (
    BLUE,
    DEGREES,
     Sphere,
    ThreeDAxes,
    ThreeDScene,
    VGroup,
    ValueTracker,
    YELLOW,
    interpolate_color,
    linear,
)


def kuramoto_mean_field_ode(
    t: float,
    theta: np.ndarray,
    omega: np.ndarray,
    k_coupling: float,
) -> np.ndarray:
    """Compute d(theta)/dt for N all-to-all coupled oscillators in O(N) time."""
    # Compute mean-field order parameter components Rx and Ry across all N oscillators
    rx = np.mean(np.cos(theta))
    ry = np.mean(np.sin(theta))

    # Use trig identity sin(theta_j - theta_i) = sin(theta_j)cos(theta_i) - cos(theta_j)sin(theta_i)
    return omega + k_coupling * (ry * np.cos(theta) - rx * np.sin(theta))


def solve_kuramoto_subset(
    n_total: int = 1000,
    n_sample: int = 35,
    k_coupling: float = 2.5,
    omega_mean: float = 3.0,
    omega_std: float = 0.8,
    t_max: float = 20.0,
    num_steps: int = 400,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate n_total Gaussian oscillators and return a consistent subset of n_sample."""
    # Seed the random number generator for reproducible frequencies and initial phases
    rng = np.random.default_rng(seed)

    # Draw natural frequencies from a Gaussian distribution and initial phases uniformly
    omega = rng.normal(loc=omega_mean, scale=omega_std, size=n_total)
    theta0 = rng.uniform(low=0.0, high=2.0 * np.pi, size=n_total)

    # Integrate the full N-oscillator system over the discrete time grid
    t_eval = np.linspace(0.0, t_max, num_steps)
    solution = solve_ivp(
        fun=kuramoto_mean_field_ode,
        t_span=(0.0, t_max),
        y0=theta0,
        t_eval=t_eval,
        args=(omega, k_coupling),
        method="RK45",
    )

    # Slice the first n_sample rows to consistently track the exact same oscillators over time
    sampled_phases = solution.y[:n_sample, :]
    return solution.t, sampled_phases


class Kuramoto3DSynchronization(ThreeDScene):
    """Rotate around a 3D cloud of sampled Kuramoto oscillators as they phase-lock."""

    def construct(self) -> None:
        # 1. Define simulation and animation parameters
        n_total = 1000
        n_sample = 35
        t_max = 20.0
        num_steps = 400
        anim_run_time = 20.0

        # 2. Solve the full system and extract the consistent subset of oscillator phases
        _, sampled_phases = solve_kuramoto_subset(
            n_total=n_total,
            n_sample=n_sample,
            k_coupling=2.5,
            omega_mean=4.0,
            omega_std=1.0,
            t_max=t_max,
            num_steps=num_steps,
        )

        # 3. Configure the 3D coordinate axes and start ambient camera rotation
        axes = ThreeDAxes(
            x_range=[-4, 4, 2],
            y_range=[-4, 4, 2],
            z_range=[-3, 3, 2],
            x_length=7,
            y_length=7,
            z_length=5,
        )
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.10)

        # 4. Generate fixed 3D coordinates and low-poly spheres for the sampled subset
        rng = np.random.default_rng(101)
        positions = rng.uniform(low=-2.5, high=2.5, size=(n_sample, 3))

        spheres = VGroup()
        for pos in positions:
            sphere = Sphere(radius=0.10, resolution=(6, 6))
            sphere.move_to(pos)
            sphere.set_color(BLUE)
            spheres.add(sphere)

        # 5. Bind an updater that maps the current time step to each sphere's flash color
        step_tracker = ValueTracker(0.0)

        def update_spheres(mob: VGroup) -> None:
            idx = int(np.clip(step_tracker.get_value(), 0, num_steps - 1))
            current_phases = sampled_phases[:, idx]
            # Sharp, smooth peak near theta = 0 (mod 2*pi) for the yellow phase flash
            flash_weights = ((1.0 + np.cos(current_phases)) * 0.5) ** 8
            for sphere, weight in zip(mob, flash_weights):
                sphere.set_color(interpolate_color(BLUE, YELLOW, float(weight)))

        spheres.add_updater(update_spheres)

        # 6. Add objects to the 3D scene and animate through all time steps
        self.add(axes, spheres)
        self.play(
            step_tracker.animate.set_value(num_steps - 1),
            run_time=anim_run_time,
            rate_func=linear,
        )
        self.stop_ambient_camera_rotation()
        self.wait(1.5)
