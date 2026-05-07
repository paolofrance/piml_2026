"""
Lagrangian Neural Network (LNN)
Reference: Cranmer et al. 2020 (https://arxiv.org/abs/2003.04630)

Key idea: instead of directly learning qddot = f(q, qdot),
learn a scalar Lagrangian  L(q, qdot) and derive the equations
of motion analytically via the Euler-Lagrange equations.

Euler-Lagrange: d/dt(dL/dqdot) - dL/dq = 0
Expanded:       M(q,qdot) * qddot = dL/dq - C(q,qdot) * qdot

where:
  M_ij = d²L / dqdot_i dqdot_j       (generalised mass matrix, must be SPD)
  C_i  = sum_j (d²L/dqdot_i dq_j) * qdot_j   (Coriolis-like term)

Solving:  qddot = M^{-1} (dL/dq - C)

All second derivatives are computed via PyTorch autograd.
The network is an MLP; physical structure emerges through how it is used.

Supports arbitrary n_dof (1-DOF pendulum or n-DOF systems).
"""

import torch
import torch.nn as nn


class LNN(nn.Module):
    """
    Lagrangian Neural Network.  Supports n_dof >= 1.

    Learns L(q, qdot) as a scalar MLP, then derives qddot via EL + autograd.

    Input  (forward): q (B, n_dof), qdot (B, n_dof)
    Output           : qddot (B, n_dof)
    """

    def __init__(self, n_dof: int = 1, hidden_dim: int = 64, n_layers: int = 3,
                 activation=nn.Softplus):
        super().__init__()
        self.n_dof = n_dof
        layers = [nn.Linear(2 * n_dof, hidden_dim), activation()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def lagrangian(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """Scalar Lagrangian L(q, qdot).  Returns (B, 1)."""
        return self.net(torch.cat([q, qdot], dim=-1))

    def forward(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """
        Compute qddot from the Euler-Lagrange equations for any n_dof.

        Strategy: compute the n_dof x n_dof mass matrix M and Coriolis vector C
        row-by-row using autograd, then solve  M * qddot = (dL/dq - C).

        Complexity: O(n_dof) autograd passes per forward (one per DOF).
        """
        q_r    = q.detach().requires_grad_(True)     # (B, n)
        qdot_r = qdot.detach().requires_grad_(True)  # (B, n)
        n      = q_r.shape[-1]

        L     = self.lagrangian(q_r, qdot_r)          # (B, 1)
        L_sum = L.sum()

        dL_dqdot = torch.autograd.grad(L_sum, qdot_r, create_graph=True)[0]  # (B, n)
        dL_dq    = torch.autograd.grad(L_sum, q_r,    create_graph=True)[0]  # (B, n)

        # Build M (B, n, n) and coriolis (B, n) row by row
        M_rows  = []
        C_rows  = []
        for i in range(n):
            # d(dL/dqdot_i)/d(qdot) — row i of the mass matrix
            M_row = torch.autograd.grad(
                dL_dqdot[:, i].sum(), qdot_r, create_graph=True
            )[0]                                                    # (B, n)
            M_rows.append(M_row)

            # d(dL/dqdot_i)/d(q) dotted with qdot — Coriolis element i
            C_row = torch.autograd.grad(
                dL_dqdot[:, i].sum(), q_r, create_graph=True
            )[0]                                                    # (B, n)
            C_rows.append((C_row * qdot_r).sum(-1, keepdim=True))  # (B, 1)

        M         = torch.stack(M_rows, dim=1)   # (B, n, n)
        coriolis  = torch.cat(C_rows,   dim=-1)  # (B, n)

        # Regularise M to guarantee invertibility (Softplus keeps M > 0 naturally,
        # but numerical noise can still make it singular for small networks)
        eps_I = 1e-2 * torch.eye(n, device=q.device, dtype=q.dtype)
        rhs   = (dL_dq - coriolis).unsqueeze(-1)          # (B, n, 1)
        qddot = torch.linalg.solve(M + eps_I, rhs).squeeze(-1)  # (B, n)

        return qddot

    def predict_acceleration(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        return self.forward(q, qdot)
