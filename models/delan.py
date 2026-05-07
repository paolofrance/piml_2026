"""
Deep Lagrangian Networks (DeLaN)
Reference: Lutter et al. 2019 (https://arxiv.org/abs/1907.04490)

DeLaN improves on LNN by explicitly decomposing the Lagrangian into
kinetic and potential energy:

    L(q, qdot) = T(q, qdot) - V(q)

    T = 0.5 * qdot^T  M(q)  qdot          (kinetic energy)
    V = V_net(q)  >= 0                     (learned potential)

M(q) is parameterised as a Cholesky decomposition:

    M(q) = L(q) L(q)^T + eps * I

where L(q) is a lower-triangular matrix network.  This guarantees:
  - M is symmetric positive definite  (physically required)
  - Invertibility is ensured by eps*I

The EL solve uses the same row-by-row autograd approach as LNN,
so the method works for any n_dof.

Key advantages over plain LNN:
  - SPD mass matrix guaranteed by design (no eps needed in EL)
  - Potential energy non-negative by construction
  - T and V can be inspected separately (interpretable)

Supports arbitrary n_dof.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, act):
    layers = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class DeLaN(nn.Module):
    """
    Deep Lagrangian Network with structured L = T - V decomposition.
    Supports n_dof >= 1.
    """

    def __init__(self, n_dof: int = 1, hidden_dim: int = 64, n_layers: int = 3,
                 activation=nn.Tanh, eps: float = 1e-3):
        super().__init__()
        self.n_dof = n_dof
        self.eps   = eps

        n_chol = n_dof * (n_dof + 1) // 2      # lower-triangular entries
        self.mass_net      = _mlp(n_dof, hidden_dim, n_chol, n_layers, activation)
        self.potential_net = _mlp(n_dof, hidden_dim, 1,      n_layers, activation)

    # ------------------------------------------------------------------
    # Structured energy components
    # ------------------------------------------------------------------

    def mass_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        SPD mass matrix via Cholesky:  M = L L^T + eps*I.
        Diagonal of L is exponentiated to ensure positive diagonal.

        Args:
            q: (B, n_dof)
        Returns:
            M: (B, n_dof, n_dof)
        """
        B, n = q.shape[0], self.n_dof
        entries = self.mass_net(q)                   # (B, n*(n+1)//2)

        Lmat = torch.zeros(B, n, n, device=q.device, dtype=q.dtype)
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                Lmat[:, i, j] = (torch.exp(entries[:, idx]) if i == j
                                 else entries[:, idx])
                idx += 1

        return Lmat @ Lmat.transpose(-1, -2) + self.eps * torch.eye(n, device=q.device)

    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        """Learned V(q) >= 0 (softplus).  Returns (B, 1)."""
        return F.softplus(self.potential_net(q))

    def kinetic_energy(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """T = 0.5 * qdot^T M(q) qdot.  Returns (B, 1)."""
        M   = self.mass_matrix(q)           # (B, n, n)
        qd  = qdot.unsqueeze(-1)            # (B, n, 1)
        return 0.5 * (qd.transpose(-1, -2) @ M @ qd).squeeze(-1)  # (B, 1)

    def lagrangian(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """L = T - V.  Returns (B, 1)."""
        return self.kinetic_energy(q, qdot) - self.potential_energy(q)

    # ------------------------------------------------------------------
    # Euler-Lagrange equations  (general n_dof, same as LNN)
    # ------------------------------------------------------------------

    def forward(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """
        Compute qddot via EL on the structured Lagrangian.
        Row-by-row Hessian approach — works for any n_dof.

        Args:
            q    : (B, n_dof)
            qdot : (B, n_dof)
        Returns:
            qddot: (B, n_dof)
        """
        q_r    = q.detach().requires_grad_(True)
        qdot_r = qdot.detach().requires_grad_(True)
        n      = q_r.shape[-1]

        L     = self.lagrangian(q_r, qdot_r)
        L_sum = L.sum()

        dL_dqdot = torch.autograd.grad(L_sum, qdot_r, create_graph=True)[0]  # (B, n)
        dL_dq    = torch.autograd.grad(L_sum, q_r,    create_graph=True)[0]  # (B, n)

        M_rows, C_rows = [], []
        for i in range(n):
            M_row = torch.autograd.grad(
                dL_dqdot[:, i].sum(), qdot_r, create_graph=True
            )[0]
            M_rows.append(M_row)

            C_row = torch.autograd.grad(
                dL_dqdot[:, i].sum(), q_r, create_graph=True
            )[0]
            C_rows.append((C_row * qdot_r).sum(-1, keepdim=True))

        M        = torch.stack(M_rows, dim=1)   # (B, n, n)
        coriolis = torch.cat(C_rows,   dim=-1)  # (B, n)

        # M is already SPD by design — eps_I is a small safety margin
        eps_I = 1e-4 * torch.eye(n, device=q.device, dtype=q.dtype)
        rhs   = (dL_dq - coriolis).unsqueeze(-1)
        qddot = torch.linalg.solve(M + eps_I, rhs).squeeze(-1)

        return qddot

    def predict_acceleration(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        return self.forward(q, qdot)
