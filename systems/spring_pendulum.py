"""
Spring pendulum in 2D polar coordinates.
Matches the system from the pytorch-lagrangian-nn notebook.

Generalised coordinates: q = (r, theta)
  r     — spring length
  theta — angle from vertical (downward = 0)

Lagrangian (m=1):
  T = 0.5*(rdot^2 + r^2*thetadot^2)
  V = g*r*(1 - cos(theta)) + k*(r - r0)^2
  L = T - V

Euler-Lagrange equations:
  r_ddot     = r*thetadot^2 - g*(1 - cos(theta)) - 2*k*(r - r0)
  theta_ddot = (-g*sin(theta) - 2*rdot*thetadot) / r

This is a 2-DOF conservative system.  Total energy E = T + V is conserved.
"""

import numpy as np
from scipy.integrate import solve_ivp


class SpringPendulum:
    def __init__(self, g: float = 10.0, k: float = 10.0, r0: float = 1.0):
        self.g  = g
        self.k  = k
        self.r0 = r0

    # ------------------------------------------------------------------
    # True equations of motion
    # ------------------------------------------------------------------

    def acceleration(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """
        Args:
            q    : (..., 2)  [r, theta]
            qdot : (..., 2)  [rdot, thetadot]
        Returns:
            qddot: (..., 2)  [r_ddot, theta_ddot]
        """
        r, theta       = q[..., 0],    q[..., 1]
        rdot, thetadot = qdot[..., 0], qdot[..., 1]

        r_ddot     = r * thetadot**2 - self.g * (1 - np.cos(theta)) - 2 * self.k * (r - self.r0)
        theta_ddot = (-self.g * np.sin(theta) - 2 * rdot * thetadot) / r

        return np.stack([r_ddot, theta_ddot], axis=-1)

    def _ode(self, t, state):
        q    = state[:2]
        qdot = state[2:]
        qddot = self.acceleration(q, qdot)
        return np.concatenate([qdot, qddot])

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def rollout(self, q0: np.ndarray, qdot0: np.ndarray,
                t_span: tuple, n_points: int = 1000) -> dict:
        """
        Integrate the ODE and return a trajectory dict.

        Returns:
            dict with keys: t, q (N,2), qdot (N,2), qddot (N,2), xy (N,2)
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)
        x0 = np.concatenate([q0, qdot0])
        sol = solve_ivp(self._ode, t_span, x0, t_eval=t_eval,
                        method="RK45", rtol=1e-10, atol=1e-12)

        q    = sol.y[:2].T                          # (N, 2)
        qdot = sol.y[2:].T                          # (N, 2)
        qddot = self.acceleration(q, qdot)          # (N, 2)
        xy    = self.polar_to_xy(q)                 # (N, 2)

        return {"t": sol.t, "q": q, "qdot": qdot, "qddot": qddot, "xy": xy}

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------

    def kinetic_energy(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        r, theta   = q[..., 0], q[..., 1]
        rdot, tdot = qdot[..., 0], qdot[..., 1]
        return 0.5 * (rdot**2 + (r * tdot)**2)

    def potential_energy(self, q: np.ndarray) -> np.ndarray:
        r, theta = q[..., 0], q[..., 1]
        return self.g * r * (1 - np.cos(theta)) + self.k * (r - self.r0)**2

    def total_energy(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        return self.kinetic_energy(q, qdot) + self.potential_energy(q)

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    @staticmethod
    def polar_to_xy(q: np.ndarray) -> np.ndarray:
        """Convert (r, theta) to Cartesian (x, y)."""
        r, theta = q[..., 0], q[..., 1]
        x =  r * np.sin(theta)
        y = -r * np.cos(theta)
        return np.stack([x, y], axis=-1)

    # ------------------------------------------------------------------
    # Dataset for dynamics models
    # ------------------------------------------------------------------

    def generate_dataset(self, q0, qdot0, t_span, n_points) -> dict:
        """
        Generate (q, qdot, qddot) dataset for dynamics model training.
        Returns arrays ready to pass to train_dynamics_model().
        """
        traj = self.rollout(q0, qdot0, t_span, n_points)
        # Reshape to match the convention in training.py:
        # theta    -> first generalised coordinate  (here: r)
        # theta_dot-> first velocity                (here: rdot)
        # For multi-DOF: store as (N, n_dof) arrays
        return {
            "q":     traj["q"],      # (N, 2)
            "qdot":  traj["qdot"],   # (N, 2)
            "qddot": traj["qddot"],  # (N, 2)
            "t":     traj["t"],
        }
