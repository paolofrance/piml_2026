"""
1D underdamped mass-spring-damper (MCK system).

ODE:  m x'' + c x' + k x = 0
      x'' + 2*delta*x' + omega0^2 * x = 0
      (delta = c/(2m),  omega0 = sqrt(k/m))

Analytical solution (underdamped, delta < omega0):
  x(t) = exp(-delta*t) * 2*A * cos(phi + omega*t)
  omega = sqrt(omega0^2 - delta^2)
  phi   = arctan(-delta / omega)
  A     = 1 / (2 * cos(phi))

Initial conditions: x(0) = 1, x'(0) = 0
"""

import numpy as np
import torch


class MassSpringDamper:
    def __init__(self, delta: float = 2.0, omega0: float = 20.0):
        assert delta < omega0, "System must be underdamped (delta < omega0)"
        self.delta  = delta
        self.omega0 = omega0
        self.omega  = np.sqrt(omega0**2 - delta**2)
        self.phi    = np.arctan(-delta / self.omega)
        self.A      = 1.0 / (2.0 * np.cos(self.phi))
        # ODE coefficients (m=1 normalisation)
        self.c = 2 * delta       # damping coefficient
        self.k = omega0**2       # spring constant

    # ------------------------------------------------------------------
    # Analytical solution (numpy)
    # ------------------------------------------------------------------

    def solution(self, t: np.ndarray) -> np.ndarray:
        return np.exp(-self.delta * t) * 2 * self.A * np.cos(self.phi + self.omega * t)

    def velocity(self, t: np.ndarray) -> np.ndarray:
        """Exact first derivative dx/dt."""
        e  = np.exp(-self.delta * t)
        c  = np.cos(self.phi + self.omega * t)
        s  = np.sin(self.phi + self.omega * t)
        return 2 * self.A * e * (-self.delta * c - self.omega * s)

    def acceleration(self, t: np.ndarray) -> np.ndarray:
        """Exact second derivative d²x/dt² = -omega0²*x - 2*delta*xdot."""
        return -self.omega0**2 * self.solution(t) - self.c * self.velocity(t)

    # ------------------------------------------------------------------
    # Analytical solution (torch)
    # ------------------------------------------------------------------

    def solution_torch(self, t: torch.Tensor) -> torch.Tensor:
        e   = torch.exp(-self.delta * t)
        cos = torch.cos(self.phi + self.omega * t)
        return e * 2 * self.A * cos
