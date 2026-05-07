"""
delan_friction.py
=================
DeLaN with explicit friction vs plain DeLaN vs VanillaNN on a dissipative
spring pendulum (2-DOF).

System  : Spring pendulum with viscous friction
          r̈  = r·θ̇² − g(1−cosθ) − 2k(r−r₀) − b_r · ṙ
          θ̈  = (−g·sinθ − 2ṙ·θ̇) / r  −  b_θ · θ̇
          Parameters: g=10, k=10, r₀=1, b_r=0.1, b_θ=0.3
Data    : 80 (q, qdot, qddot) samples from t ∈ [0, 3 s]
Eval    : RK4 rollout over t ∈ [0, 8 s]

Three models compared:
  DeLaN-F   — DeLaN + learnable Rayleigh dissipation matrix B (PSD via Cholesky)
              EOM:  M(q) q̈ = rhs_conservative − B(q) q̇
              B PSD guarantees energy always decreases: physically correct
  DeLaN     — standard conservative DeLaN (no friction structure)
              Knows physics is Lagrangian but assumes energy conservation — wrong model
  VanillaNN — unstructured (q, q̇) → q̈ regression

Key insight: DeLaN-F's dissipation structure correctly models the energy decay,
while conservative DeLaN predicts undamped oscillations and VanillaNN drifts.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from systems import SpringPendulum
from models import DeLaN, VanillaNN
from utils.training import train_dynamics_model

SEED   = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {
    "True":      "#2c3e50",
    "DeLaN-F":   "#9b59b6",
    "DeLaN":     "#2ecc71",
    "VanillaNN": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Dissipative system
# ---------------------------------------------------------------------------

class FrictionSpringPendulum(SpringPendulum):
    """
    Spring pendulum with viscous friction in both generalized coordinates.

    Rayleigh dissipation function: D = ½ (b_r ṙ² + b_θ θ̇²)
    Generalised friction forces  : Q = −∂D/∂q̇ = −[b_r ṙ,  b_θ θ̇]
    """

    def __init__(self, g=10.0, k=10.0, r0=1.0, b_r=0.1, b_theta=0.3):
        super().__init__(g, k, r0)
        self.b_r     = b_r
        self.b_theta = b_theta

    def acceleration(self, q, qdot):
        r, theta       = q[..., 0],    q[..., 1]
        rdot, thetadot = qdot[..., 0], qdot[..., 1]

        r_ddot     = (r * thetadot**2
                      - self.g * (1 - np.cos(theta))
                      - 2 * self.k * (r - self.r0)
                      - self.b_r * rdot)
        theta_ddot = ((-self.g * np.sin(theta) - 2 * rdot * thetadot) / r
                      - self.b_theta * thetadot)

        return np.stack([r_ddot, theta_ddot], axis=-1)


# ---------------------------------------------------------------------------
# DeLaN with Rayleigh dissipation
# ---------------------------------------------------------------------------

def _mlp(in_dim, hidden_dim, out_dim, n_layers, act):
    layers = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class DeLaNFriction(nn.Module):
    """
    Deep Lagrangian Network with explicit Rayleigh dissipation.

    Energy is structured as:
        T(q, q̇) = ½ q̇ᵀ M(q) q̇       (SPD mass matrix via Cholesky)
        V(q)    = V_net(q) ≥ 0         (learned potential)
        D(q, q̇) = ½ q̇ᵀ B(q) q̇        (Rayleigh dissipation, B PSD)

    Equations of motion:
        M(q) q̈ = ∂L/∂q − [∂/∂q(∂L/∂q̇)] q̇ − B(q) q̇

    B PSD ⟹  dE/dt = −q̇ᵀ B q̇ ≤ 0  (energy can only decrease — by design).
    """

    def __init__(self, n_dof=2, hidden_dim=64, n_layers=3,
                 activation=nn.Softplus, eps=1e-3, eps_b=1e-3):
        super().__init__()
        self.n_dof = n_dof
        self.eps   = eps
        self.eps_b = eps_b

        n_chol = n_dof * (n_dof + 1) // 2
        self.mass_net      = _mlp(n_dof, hidden_dim, n_chol, n_layers, activation)
        self.potential_net = _mlp(n_dof, hidden_dim, 1,      n_layers, activation)
        self.friction_net  = _mlp(n_dof, hidden_dim, n_chol, n_layers, activation)

    def _cholesky_matrix(self, net_out, eps):
        """Build PSD matrix from lower-Cholesky entries. Returns (B, n, n)."""
        B, n = net_out.shape[0], self.n_dof
        Lmat = torch.zeros(B, n, n, device=net_out.device, dtype=net_out.dtype)
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                Lmat[:, i, j] = (torch.exp(net_out[:, idx]) if i == j
                                 else net_out[:, idx])
                idx += 1
        return Lmat @ Lmat.transpose(-1, -2) + eps * torch.eye(n, device=net_out.device)

    def mass_matrix(self, q):
        return self._cholesky_matrix(self.mass_net(q), self.eps)

    def friction_matrix(self, q):
        return self._cholesky_matrix(self.friction_net(q), self.eps_b)

    def potential_energy(self, q):
        return F.softplus(self.potential_net(q))

    def kinetic_energy(self, q, qdot):
        M  = self.mass_matrix(q)
        qd = qdot.unsqueeze(-1)
        return 0.5 * (qd.transpose(-1, -2) @ M @ qd).squeeze(-1)

    def lagrangian(self, q, qdot):
        return self.kinetic_energy(q, qdot) - self.potential_energy(q)

    def forward(self, q, qdot):
        q_r    = q.detach().requires_grad_(True)
        qdot_r = qdot.detach().requires_grad_(True)
        n      = q_r.shape[-1]

        L_val = self.lagrangian(q_r, qdot_r)
        L_sum = L_val.sum()

        dL_dqdot = torch.autograd.grad(L_sum, qdot_r, create_graph=True)[0]
        dL_dq    = torch.autograd.grad(L_sum, q_r,    create_graph=True)[0]

        M_rows, C_rows = [], []
        for i in range(n):
            M_row = torch.autograd.grad(
                dL_dqdot[:, i].sum(), qdot_r, create_graph=True)[0]
            M_rows.append(M_row)
            C_row = torch.autograd.grad(
                dL_dqdot[:, i].sum(), q_r, create_graph=True)[0]
            C_rows.append((C_row * qdot_r).sum(-1, keepdim=True))

        M_mat    = torch.stack(M_rows, dim=1)       # (B, n, n)
        coriolis = torch.cat(C_rows, dim=-1)        # (B, n)

        # Rayleigh friction: −B(q) q̇
        B_mat    = self.friction_matrix(q_r)        # (B, n, n)
        friction = (B_mat @ qdot_r.unsqueeze(-1)).squeeze(-1)  # (B, n)

        eps_I = 1e-4 * torch.eye(n, device=q.device, dtype=q.dtype)
        rhs   = (dL_dq - coriolis - friction).unsqueeze(-1)
        return torch.linalg.solve(M_mat + eps_I, rhs).squeeze(-1)

    def predict_acceleration(self, q, qdot):
        return self.forward(q, qdot)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

sp    = FrictionSpringPendulum(g=10.0, k=10.0, r0=1.0, b_r=0.1, b_theta=0.3)
Q0    = np.array([1.1, 0.5])
QDOT0 = np.array([0.0, 0.0])
T_TRAIN = (0.0, 3.0)
T_EVAL  = (0.0, 8.0)
N_TRAIN, N_EVAL = 80, 800

train_ds  = sp.generate_dataset(Q0, QDOT0, T_TRAIN, N_TRAIN)
test_traj = sp.rollout(Q0, QDOT0, T_EVAL, N_EVAL)

t_eval  = test_traj["t"]
q_true  = test_traj["q"]
qd_true = test_traj["qdot"]
xy_true = test_traj["xy"]
E_true  = sp.total_energy(q_true, qd_true)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

EPOCHS = 3_000

models = {
    "DeLaN-F":   DeLaNFriction(n_dof=2, hidden_dim=64, n_layers=3, activation=nn.Softplus),
    "DeLaN":     DeLaN(        n_dof=2, hidden_dim=64, n_layers=3, activation=nn.Softplus),
    "VanillaNN": VanillaNN(    n_dof=2, hidden_dim=64, n_layers=3),
}
for name, model in models.items():
    print(f"\nTraining {name} ...")
    hist = train_dynamics_model(model, train_ds, n_epochs=EPOCHS,
                                batch_size=32, device=DEVICE,
                                verbose=True, log_every=500)
    print(f"  loss={hist['train_loss'][-1]:.4e}  t={hist['wall_time']:.1f}s")

# ---------------------------------------------------------------------------
# RK4 rollout
# ---------------------------------------------------------------------------

def rk4(model, q0, qdot0, t_array, device="cpu"):
    q, qdot = q0.copy(), qdot0.copy()
    qs = [q.copy()]
    def acc(qi, qdi):
        qt  = torch.tensor(qi[None],  dtype=torch.float32, device=device)
        qdt = torch.tensor(qdi[None], dtype=torch.float32, device=device)
        with torch.enable_grad():
            return model.predict_acceleration(qt, qdt).detach().cpu().numpy()[0]
    for dt in np.diff(t_array):
        k1v = qdot;              k1a = acc(q,              qdot)
        k2v = qdot + .5*dt*k1a; k2a = acc(q + .5*dt*k1v, qdot + .5*dt*k1a)
        k3v = qdot + .5*dt*k2a; k3a = acc(q + .5*dt*k2v, qdot + .5*dt*k2a)
        k4v = qdot +    dt*k3a; k4a = acc(q +    dt*k3v, qdot +    dt*k3a)
        q    = q    + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
        qdot = qdot + (dt/6)*(k1a + 2*k2a + 2*k3a + k4a)
        qs.append(q.copy())
    return np.array(qs)

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

split = T_TRAIN[1]
in_m, out_m = t_eval <= split, t_eval > split

preds = {}
for name, model in models.items():
    model.eval()
    q_pred  = rk4(model, Q0, QDOT0, t_eval, DEVICE)
    q_pred  = np.clip(q_pred, -20, 20)
    qd_pred = np.gradient(q_pred, t_eval, axis=0)
    xy_pred = SpringPendulum.polar_to_xy(q_pred)
    E_pred  = sp.total_energy(q_pred, qd_pred)
    rmse_in  = float(np.sqrt(np.mean((xy_pred[in_m]  - xy_true[in_m]) **2)))
    rmse_out = float(np.sqrt(np.mean((xy_pred[out_m] - xy_true[out_m])**2)))
    e_drift  = float(np.mean(np.abs(E_pred - E_true) / (np.abs(E_true) + 1e-8)))
    preds[name] = {"xy": xy_pred, "E": E_pred,
                   "interp": rmse_in, "extrap": rmse_out, "edrift": e_drift}

print(f"\n{'Model':<12} {'Interp RMSE':>13} {'Extrap RMSE':>13} {'E-error':>10}")
print("-" * 52)
for name, p in preds.items():
    print(f"{name:<12} {p['interp']:>13.4f} {p['extrap']:>13.4f} {p['edrift']:>10.4f}")
gain = preds["VanillaNN"]["extrap"] / (preds["DeLaN-F"]["extrap"] + 1e-12)
print(f"\nDeLaN-F extrap improvement over VanillaNN: {gain:.1f}×")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
fig.suptitle(
    f"DeLaN with friction vs DeLaN vs VanillaNN — Dissipative Spring Pendulum\n"
    f"b_r={sp.b_r}, b_θ={sp.b_theta}  |  "
    f"{N_TRAIN} training pts from [0, {T_TRAIN[1]} s]  |  shaded = extrapolation",
    fontsize=12, fontweight="bold",
)

for ax, ci, lbl in zip(axes[:2], [0, 1], ["x (m)", "y (m)"]):
    ax.plot(t_eval, xy_true[:, ci], color=COLORS["True"], lw=1, ls="--",
            label="Ground truth", zorder=5)
    for name in ["DeLaN-F", "DeLaN", "VanillaNN"]:
        xy_plot = np.clip(preds[name]["xy"][:, ci], -5, 5)
        ax.plot(t_eval, xy_plot, color=COLORS[name], lw=1.8,
                label=f"{name}  (extrap RMSE={preds[name]['extrap']:.4f})")
    ax.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
    ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
    ax.set_ylabel(lbl, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Energy decay
ax = axes[2]
ax.plot(t_eval, E_true, color=COLORS["True"], lw=1, ls="--",
        label="Ground truth (dissipating)", zorder=5)
for name in ["DeLaN-F", "DeLaN", "VanillaNN"]:
    ax.plot(t_eval, preds[name]["E"], color=COLORS[name], lw=1.8,
            label=f"{name}  (mean |ΔE/E|={preds[name]['edrift']:.3f})")
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("Total energy (J)", fontsize=11)
ax.set_title("Energy — dissipation should cause monotone decay", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Error
ax = axes[3]
for name in ["DeLaN-F", "DeLaN", "VanillaNN"]:
    xy_p = preds[name]["xy"]
    err  = np.sqrt(np.sum((np.clip(xy_p, -5, 5) - xy_true)**2, axis=1))
    ax.semilogy(t_eval, err + 1e-6, color=COLORS[name], lw=1.6, label=name)
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("‖xy error‖  (log scale)", fontsize=11)
ax.set_xlabel("Time (s)", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
