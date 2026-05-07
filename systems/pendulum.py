"""
Simple Pendulum: the ground-truth dynamical system used as benchmark.

State vector: [theta, theta_dot]
  theta     - angle from vertical (rad)
  theta_dot - angular velocity (rad/s)

Equation of motion: theta_ddot = -(g/L) * sin(theta)

Lagrangian (m=1, L=1):
  T = 0.5 * theta_dot^2
  V = g * (1 - cos(theta))
  L = T - V
"""

import numpy as np
from scipy.integrate import solve_ivp


class SimplePendulum:
    def __init__(self, g: float = 9.81, length: float = 1.0, mass: float = 1.0):
        self.g = g
        self.L = length
        self.m = mass

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def acceleration(self, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
        """True angular acceleration from Newton/Euler-Lagrange."""
        return -(self.g / self.L) * np.sin(theta)

    def _ode(self, t, state):
        theta, theta_dot = state
        return [theta_dot, self.acceleration(theta, theta_dot)]

    def rollout(
        self,
        theta0: float,
        theta_dot0: float,
        t_span: tuple,
        n_points: int = 500,
    ) -> dict:
        """Integrate the ODE and return a trajectory dict."""
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        sol = solve_ivp(
            self._ode,
            t_span,
            [theta0, theta_dot0],
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )
        theta = sol.y[0]
        theta_dot = sol.y[1]
        theta_ddot = self.acceleration(theta, theta_dot)
        return {
            "t": sol.t,
            "theta": theta,
            "theta_dot": theta_dot,
            "theta_ddot": theta_ddot,
        }

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def kinetic_energy(self, theta_dot: np.ndarray) -> np.ndarray:
        return 0.5 * self.m * self.L**2 * theta_dot**2

    def potential_energy(self, theta: np.ndarray) -> np.ndarray:
        return self.m * self.g * self.L * (1.0 - np.cos(theta))

    def total_energy(self, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
        return self.kinetic_energy(theta_dot) + self.potential_energy(theta)

    # ------------------------------------------------------------------
    # Lagrangian (for reference / DeLaN ground-truth)
    # ------------------------------------------------------------------

    def lagrangian(self, theta: np.ndarray, theta_dot: np.ndarray) -> np.ndarray:
        return self.kinetic_energy(theta_dot) - self.potential_energy(theta)

    # ------------------------------------------------------------------
    # Finite differences  (mimics real sensor data — no acceleration labels)
    # ------------------------------------------------------------------

    @staticmethod
    def finite_differences(t: np.ndarray, theta: np.ndarray) -> dict:
        """
        Estimate theta_dot and theta_ddot from position observations only,
        using second-order central differences (forward/backward at boundaries).

        This is what you would do with a real encoder: you only measure position
        and must differentiate numerically to get velocity and acceleration.
        """
        dt = np.diff(t)

        # --- velocity (central differences) ---
        theta_dot = np.empty_like(theta)
        theta_dot[1:-1] = (theta[2:] - theta[:-2]) / (t[2:] - t[:-2])
        theta_dot[0]    = (theta[1] - theta[0])   / dt[0]           # forward
        theta_dot[-1]   = (theta[-1] - theta[-2]) / dt[-1]          # backward

        # --- acceleration (central differences on uniform grid) ---
        # Use mean dt for robustness; the grid from solve_ivp is uniform
        dt_mean = np.mean(dt)
        theta_ddot = np.empty_like(theta)
        theta_ddot[1:-1] = (theta[2:] - 2 * theta[1:-1] + theta[:-2]) / dt_mean**2
        theta_ddot[0]    = theta_ddot[1]    # copy interior
        theta_ddot[-1]   = theta_ddot[-2]

        return {"theta_dot": theta_dot, "theta_ddot": theta_ddot}

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_trajectories: int = 20,
        theta0_range: tuple = (-np.pi * 0.8, np.pi * 0.8),
        theta_dot0_range: tuple = (-2.0, 2.0),
        t_span: tuple = (0.0, 5.0),
        n_points_per_traj: int = 200,
        seed: int = 42,
    ) -> dict:
        """
        Generate a dataset of (theta, theta_dot, theta_ddot) tuples from
        multiple trajectories with random initial conditions.
        """
        rng = np.random.default_rng(seed)
        all_theta, all_theta_dot, all_theta_ddot, all_t = [], [], [], []

        for _ in range(n_trajectories):
            th0 = rng.uniform(*theta0_range)
            thd0 = rng.uniform(*theta_dot0_range)
            traj = self.rollout(th0, thd0, t_span, n_points_per_traj)
            all_theta.append(traj["theta"])
            all_theta_dot.append(traj["theta_dot"])
            all_theta_ddot.append(traj["theta_ddot"])
            all_t.append(traj["t"])

        return {
            "theta": np.concatenate(all_theta),
            "theta_dot": np.concatenate(all_theta_dot),
            "theta_ddot": np.concatenate(all_theta_ddot),
            "t": np.concatenate(all_t),
        }
