# Kuramoto_fireflies
A project stemming from Nonlinear Dynamics that investigates some aspects of using the kuramoto model to represent the synchronization of firefly bioluminescence.

# Visualizing Synchronization in the Kuramoto Model

Welcome! This repository contains a suite of four Python and Manim animations exploring how populations of coupled nonlinear oscillators spontaneously synchronize. Starting from the exact analytical continuum limit ($N \to \infty$), we work our way up to 3D visualizations of finite Gaussian populations and conclude by testing an empirical Phase Response Curve (PRC) measured from real North American *Photuris frontalis* fireflies.

Every script separates the numerical differential equation solver (`scipy.integrate.solve_ivp`) from the visual scene (`manim`), making it easy to tweak parameters, swap coupling rules, and watch nonlinear dynamics come to life. Sometimes they include the same solver logic to make them enitrly self contained and little easier to play with independently from each other.

For the full mathematical derivations behind these simulations, please see:
* **Foundational Theory & Bifurcations:** Strogatz (2000), Acebrón et al. (2005), and Chapter 14 of Strogatz's *Nonlinear Dynamics and Chaos* (3rd ed., 2024).
* **Exact Continuum Reduction ($N \to \infty$):** Ott & Antonsen (2008).
* **Empirical Firefly PRC (`led_firesync`):** Martin, Nechyporenko, Jayaram, & Peleg (2026).

---

## Acknowledgments & External Code Attribution

* **Empirical Firefly PRC Code (`peleg-lab/led_firesync`):** The phase-wrapping helpers and hyperbolic tangent PRC functions (`wrap1`, `wrap_half`, `_alpha`, `R_tanh_step`, `Z_from_R`, and `make_tanh_prc`) in `kuramoto_led_firesync_bifurcation.py` are **not our original work**. They are taken directly from [`simulation/prc_simulator.py`](https://github.com/peleg-lab/led_firesync/blob/master/simulation/prc_simulator.py) by **Owen Martin, Nataliya Nechyporenko, Kaushik Jayaram, and Orit Peleg** (Peleg Lab, University of Colorado Boulder) accompanying their paper *Excitation–inhibition interactions mediate firefly flash synchronization* (2026).
* **Major Contributors:** Theoretical framing is guided by **Steven H. Strogatz's** *Nonlinear Dynamics and Chaos* (3rd ed., 2024). All animations are powered by **Manim Community** (`manim`), and numerical integrations use **SciPy** (`scipy.integrate.solve_ivp`) and **NumPy**.
Additional thanks goes to **Ryan Flanagan** for thoughful discussion, code review, and patience as my partner for the group project. 
---

## Quick Start & Environment Setup

Activate your Conda environment containing `manim`, `scipy`, and `numpy`, or create the `kuramoto_project_env` from the provided `enviroment.yml` YAML file and activate it. Then run any script from your terminal:

```bash
# Fast 480p15 preview render (recommended while experimenting with parameters)
manim -pql <script_name.py> <SceneClassName>

# High-definition 1080p60 render (recommended for final presentations)
manim -pqh <script_name.py> <SceneClassName>

```

Rendering in 480p15 is the easiest way to get started, but I don't think that it has the same effect.

---

## Shared Parameters You Can Tune

Because all four scripts share the same modular solver architecture, you can adjust the physics of the system by editing these variables at the top of each script's `construct()` or solver function:

* **`k_coupling` ($K$) / `k_max`:** The coupling strength between oscillators. Increasing $K$ past the critical threshold $K_c$ triggers a phase transition from random incoherence ($r \approx 0$) to collective synchronization ($r \to 1$).
* **`gamma` ($\gamma$):** The half-width at half-maximum of the Lorentzian (Cauchy) natural frequency distribution used in the $N \to \infty$ models. Sets the critical threshold at $K_c = 2\gamma$.
* **`omega_mean` ($\omega_0$) & `omega_std` ($\sigma$):** The mean and standard deviation of the Gaussian natural frequency distribution $\mathcal{N}(\omega_0, \sigma^2)$ used in the finite $N$ simulations. Larger `omega_std` means more unruly oscillators that require a stronger $K$ to lock.
* **`r0` ($r_0$):** The initial macroscopic order parameter in the infinite $N$ ODE ($0 < r_0 \le 1$). Must be strictly greater than $0$ so the solver isn't trapped on the unstable $r = 0$ equilibrium above $K_c$.
* **`n_total` / `n_oscillators` ($N$) & `n_sample` ($M$):** Total number of simulated oscillators in the ensemble and the smaller subset extracted for 3D rendering.
* **`t_max` & `num_steps`:** The total simulation time horizon and the number of discrete time snapshots passed to Manim.

---

## The 4 Simulations

### 1. Infinite $N$ Order Parameter Over Time (`kuramoto_ord_par_sim.py`)

* **What it shows:** Plots two scenes. First the time evolution of the macroscopic coherence $r(t)$ in the first quadrant as an infinite population of oscillators transitions from near-incoherence ($r_0 = 0.05$) to a locked steady state. The second is the has modified parameters to start at an initially synchronized population ($r_0 = 0.95$) and see it decay back to zero via linear Landau damping.  
* **The Approach:** Uses the **Ott–Antonsen (2008) ansatz** for a Lorentzian frequency distribution, which collapses the infinite-dimensional continuity PDE into a single exact scalar ODE: $\dot{r} = (\frac{K}{2} - \gamma)r - \frac{K}{2}r^3$.
* **How to run:**
The first:
```bash
manim -pqh kuramoto_ord_par_sim.py InfiniteKuramotoOrderParameter

```
The second: 
```bash
manim -pqh kuramoto_ord_par_sim.py InfiniteKuramotoOrderParameterIncoherence

```


### 2. 3D Rotating Ensemble & Phase Flashing (`kuramoto_3d_sync.py`)

* **What it shows:** Orbits a 3D cloud of blue spheres that flash bright yellow each time their phase passes through zero ($\theta_i \equiv 0 \pmod{2\pi}$). What starts as chaotic sparkling gradually locks into a unified, rhythmic pulse.
* **The Approach:** Simulates a large finite population ($N = 1000$) with Gaussian frequencies in $\mathcal{O}(N)$ time using the mean-field trigonometric identity $\frac{1}{N}\sum_j \sin(\theta_j - \theta_i) = R_y \cos\theta_i - R_x \sin\theta_i$. To keep the 3D scene fast and uncluttered, we slice the first $M = 35$ rows (`solution.y[:n_sample, :]`) so we track the exact same 35 oscillators across every frame.
* **How to run:** Please note that it will take a much longer time than the others to render, but is worth it. Run with -pql to render versions faster, but I still recommend the final be the higher quality rendering.
```bash
manim -pqh kuramoto_3d_sync.py Kuramoto3DSynchronization

```

* **Play around with this:** Adjust `omega_mean` to speed up or slow down the overall flashing cadence, or lower `k_coupling` to `0.5` (below $K_c \approx 1.28$) to see the spheres flash out of sync indefinitely.


### 3. Infinite $N$ Pitchfork Bifurcation Diagram (`kuramoto_bifurcation.py`)

* **What it shows:** Draws the classic steady-state phase transition curve ($r_\infty$ vs. $K$) with a dashed red marker at the critical threshold $K_c = 2\gamma$.
* **The Approach:** Reuses `kuramoto_oa_ode` and `solve_order_parameter` from Simulation 1 inside a parameter sweep (`sweep_coupling_strength`). For each $K \in [0, 6]$, it integrates the ODE out to $t_{\max} = 120$ (long enough to overcome critical slowing down right near $K_c$) and records the final converged value `r_trajectory[-1]`.
* **How to run:**
```bash
manim -pqh kuramoto_bifurcation.py InfiniteKuramotoBifurcation

```


* **Try this:** Change `gamma` from `1.0` to `1.5` in `construct()` to watch the critical threshold $K_c$ shift rightward to $3.0$.

### 4. Empirical Firefly PRC Bifurcation Diagram (`kuramoto_led_firesync_bifurcation.py`)

* **What it shows:** Generates an empirical $r$ vs. $K$ bifurcation diagram for $N = 100$ Gaussian oscillators coupled via the experimentally measured biphasic (excitation–inhibition) Phase Response Curve of *Photuris frontalis* fireflies.
* **The Approach:** Integrates the Peleg Lab's exact `make_tanh_prc` factory from `prc_simulator.py` into a vectorized $100 \times 100$ pairwise interaction matrix. Pairwise radian differences are converted into half-wrapped unit-cycle phases $\phi_{\text{eval}} \in (-0.5, 0.5]$ at each step. Because $N = 100$ exhibits finite-size jitter of order $\mathcal{O}(1/\sqrt{N})$, the solver averages $r(t)$ over the final 25% of each trajectory to produce a smooth bifurcation curve.
* **How to run:**
```bash
manim -pqh kuramoto_led_firesync_bifurcation.py LedFiresyncBifurcation

```