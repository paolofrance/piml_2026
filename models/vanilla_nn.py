"""
Vanilla Neural Network for dynamics learning.

Architecture: MLP that learns the black-box mapping
    (q, qdot) -> qddot

No physics structure is enforced. The network learns purely from data.
This is the baseline against which all physics-informed models are compared.

Training loss: MSE between predicted and true accelerations.
"""

import torch
import torch.nn as nn


class VanillaNN(nn.Module):
    """
    Standard MLP dynamics model.  Supports arbitrary n_dof.

    Input : [q, qdot]   shape (B, 2*n_dof)
    Output: [qddot]     shape (B, n_dof)
    """

    def __init__(self, n_dof: int = 1, hidden_dim: int = 64, n_layers: int = 3,
                 activation=nn.Tanh):
        super().__init__()
        self.n_dof = n_dof
        layers = [nn.Linear(2 * n_dof, hidden_dim), activation()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, n_dof))
        self.net = nn.Sequential(*layers)

    def forward(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q    : (B, n_dof) generalized positions
            qdot : (B, n_dof) generalized velocities
        Returns:
            qddot: (B, n_dof) predicted accelerations
        """
        x = torch.cat([q, qdot], dim=-1)
        return self.net(x)

    def predict_acceleration(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        return self.forward(q, qdot)
