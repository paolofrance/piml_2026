"""
lnn_friction.py
===============
LNN with explicit friction vs plain LNN vs VanillaNN on a dissipative
simple pendulum (1-DOF).

System  : Simple pendulum with viscous friction
          θ̈ = −(g/L) sin(θ) − b · θ̇
          Parameters: g=9.81, L=1, b=0.3
Data    : 20 (θ, θ̇, θ̈) samples from t ∈ [0, 3 s]
          θ₀=2.5 rad (near-separatrix — same as lnn_vs_vanilla.py)
Eval    : RK4 rollout over t ∈ [0, 12 s]

Three models compared:
  LNN-F     — LNN + learnable Rayleigh dissipation coefficient b(θ) > 0
              EOM:  M(θ,θ̇) θ̈ = dL/dθ − C θ̇ − b(θ) θ̇
              b(θ) > 0 by construction → energy strictly decreasing
  LNN       — standard conservative LNN (no friction structure)
              Lagrangian structure but assumes energy conservation — wrong model
  VanillaNN — unstructured (θ, θ̇) → θ̈ regression

Key insight: LNN-F's dissipation structure correctly tracks the decaying
amplitude, while conservative LNN maintains constant energy (physically wrong
for this system) and VanillaNN drifts.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from systems import SimplePendulum
from models import LNN, VanillaNN
from utils.training import train_dynamics_model

SEED   = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {
    "True":      "#2c3e50",
    "LNN-F":     "#9b59b6",
    "LNN":       "#3498db",
    "VanillaNN": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Dissipative system
# ---------------------------------------------------------------------------

class FrictionPendulum(SimplePendulum):
    """
    Simple pendulum with viscous friction.

    Rayleigh dissipation function: D = ½ b θ̇²
    Generalised friction force   : Q = −∂D/∂θ̇ = −b θ̇
    """

    def __init__(self, g=9.81, length=1.0, b=0.3):
        super().__init__(g=g, length=length)
        self.b = b

    def acceleration(self, theta, theta_dot):
        return -(self.g / self.L) * np.sin(theta) - self.b * theta_dot


# ---------------------------------------------------------------------------
# LNN with Rayleigh dissipation
# ---------------------------------------------------------------------------

class LNNFriction(nn.Module):
    """
    Lagrangian Neural Network with explicit Rayleigh dissipation (1-DOF).

    Learns a scalar Lagrangian L(θ, θ̇) and a positive friction coefficient
    b(θ) > 0 via softplus.  Equations of motion:

        M(θ, θ̇) θ̈ = dL/dθ − C(θ, θ̇) θ̇ − b(θ) θ̇

    where M = d²L/dθ̇², C = d²L/(dθ̇ dθ).

    b(θ) > 0  ⟹  dE/dt = −b(θ) θ̇² ≤ 0  (energy can only decrease).
    """

    def __init__(self, n_dof=1, hidden_dim=64, n_layers=3):
        super().__init__()
        assert n_dof == 1, "LNNFriction is designed for 1-DOF systems"
        self.n_dof = n_dof

        # Lagrangian network: (q, qdot) → scalar L
        layers = [nn.Linear(2, hidden_dim), nn.Softplus()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Softplus()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.lag_net = nn.Sequential(*layers)

        # Friction network: q → positive scalar b(q)
        f_layers = [nn.Linear(1, hidden_dim), nn.Softplus()]
        for _ in range(n_layers - 1):
            f_layers += [nn.Linear(hidden_dim, hidden_dim), nn.Softplus()]
        f_layers.append(nn.Linear(hidden_dim, 1))
        self.friction_net = nn.Sequential(*f_layers)

    def lagrangian(self, q, qdot):
        return self.lag_net(torch.cat([q, qdot], dim=-1))

    def friction_coeff(self, q):
        """Positive friction coefficient b(q) > 0 via softplus."""
        return F.softplus(self.friction_net(q))

    def forward(self, q, qdot):
        q_r    = q.detach().requires_grad_(True)
        qdot_r = qdot.detach().requires_grad_(True)

        L_val = self.lagrangian(q_r, qdot_r)
        L_sum = L_val.sum()

        dL_dqdot = torch.autograd.grad(L_sum, qdot_r, create_graph=True)[0]  # (B, 1)
        dL_dq    = torch.autograd.grad(L_sum, q_r,    create_graph=True)[0]  # (B, 1)

        # Mass (scalar for 1-DOF): M = d²L/dθ̇²
        M_val = torch.autograd.grad(
            dL_dqdot[:, 0].sum(), qdot_r, create_graph=True)[0]              # (B, 1)

        # Coriolis scalar: C = d(dL/dθ̇)/dθ · θ̇
        C_val = torch.autograd.grad(
            dL_dqdot[:, 0].sum(), q_r, create_graph=True)[0] * qdot_r        # (B, 1)

        # Rayleigh friction: −b(θ) · θ̇
        b_val    = self.friction_coeff(q_r)                                   # (B, 1)
        friction = b_val * qdot_r                                             # (B, 1)

        eps = 1e-2
        qddot = (dL_dq - C_val - friction) / (M_val + eps)
        return qddot

    def predict_acceleration(self, q, qdot):
        return self.forward(q, qdot)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

pend   = FrictionPendulum(g=9.81, length=1.0, b=0.3)
THETA0, TDOT0 = 2.5, 0.0
T_TRAIN = (0.0, 3.0)
T_EVAL  = (0.0, 12.0)
N_TRAIN, N_EVAL = 20, 1200

traj_train = pend.rollout(THETA0, TDOT0, T_TRAIN, N_TRAIN)
traj_eval  = pend.rollout(THETA0, TDOT0, T_EVAL,  N_EVAL)

train_ds = {
    "theta":      traj_train["theta"],
    "theta_dot":  traj_train["theta_dot"],
    "theta_ddot": traj_train["theta_ddot"],
}

t_eval  = traj_eval["t"]
th_true = traj_eval["theta"]
td_true = traj_eval["theta_dot"]
E_true  = pend.total_energy(th_true, td_true)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

EPOCHS = 3_000

models = {
    "LNN-F":     LNNFriction(n_dof=1, hidden_dim=64, n_layers=3),
    "LNN":       LNN(        n_dof=1, hidden_dim=64, n_layers=3),
    "VanillaNN": VanillaNN(  n_dof=1, hidden_dim=64, n_layers=3),
}
for name, model in models.items():
    print(f"\nTraining {name} ...")
    hist = train_dynamics_model(model, train_ds, n_epochs=EPOCHS,
                                batch_size=16, device=DEVICE,
                                verbose=True, log_every=500)
    print(f"  loss={hist['train_loss'][-1]:.4e}  t={hist['wall_time']:.1f}s")

# ---------------------------------------------------------------------------
# RK4 rollout
# ---------------------------------------------------------------------------

def rk4(model, theta0, thetadot0, t_array, device="cpu"):
    q, qd = float(theta0), float(thetadot0)
    qs, qds = [q], [qd]
    def acc(qi, qdi):
        qt  = torch.tensor([[qi]],  dtype=torch.float32, device=device)
        qdt = torch.tensor([[qdi]], dtype=torch.float32, device=device)
        with torch.enable_grad():
            return model.predict_acceleration(qt, qdt).detach().cpu().item()
    for dt in np.diff(t_array):
        k1v = qd;              k1a = acc(q,              qd)
        k2v = qd + .5*dt*k1a; k2a = acc(q + .5*dt*k1v, qd + .5*dt*k1a)
        k3v = qd + .5*dt*k2a; k3a = acc(q + .5*dt*k2v, qd + .5*dt*k2a)
        k4v = qd +    dt*k3a; k4a = acc(q +    dt*k3v, qd +    dt*k3a)
        q  = q  + (dt/6)*(k1v + 2*k2v + 2*k3v + k4v)
        qd = qd + (dt/6)*(k1a + 2*k2a + 2*k3a + k4a)
        qs.append(q); qds.append(qd)
    return np.array(qs), np.array(qds)

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

split = T_TRAIN[1]
in_m, out_m = t_eval <= split, t_eval > split

preds = {}
for name, model in models.items():
    model.eval()
    th, td   = rk4(model, THETA0, TDOT0, t_eval, DEVICE)
    E_pred   = pend.total_energy(th, td)
    rmse_in  = float(np.sqrt(np.mean((th[in_m]  - th_true[in_m]) **2)))
    rmse_out = float(np.sqrt(np.mean((th[out_m] - th_true[out_m])**2)))
    e_err    = float(np.mean(np.abs(E_pred - E_true) / (np.abs(E_true) + 1e-8)))
    preds[name] = {"theta": th, "theta_dot": td, "E": E_pred,
                   "interp": rmse_in, "extrap": rmse_out, "eerr": e_err}

print(f"\n{'Model':<12} {'Interp RMSE':>13} {'Extrap RMSE':>13} {'E-error':>10}")
print("-" * 52)
for name, p in preds.items():
    print(f"{name:<12} {p['interp']:>13.4f} {p['extrap']:>13.4f} {p['eerr']:>10.4f}")
gain = preds["VanillaNN"]["extrap"] / (preds["LNN-F"]["extrap"] + 1e-12)
print(f"\nLNN-F extrap improvement over VanillaNN: {gain:.1f}×")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
fig.suptitle(
    f"LNN with friction vs LNN vs VanillaNN — Dissipative Simple Pendulum\n"
    f"b={pend.b}  |  θ₀={THETA0} rad (near-separatrix)  |  "
    f"{N_TRAIN} training pts from [0, {T_TRAIN[1]} s]  |  shaded = extrapolation",
    fontsize=12, fontweight="bold",
)

# Trajectory: θ(t)
ax = axes[0]
ax.plot(t_eval, th_true, color=COLORS["True"], lw=1, ls="--",
        label="Ground truth", zorder=5)
for name in ["LNN-F", "LNN", "VanillaNN"]:
    th_plot = np.clip(preds[name]["theta"], -15, 15)
    ax.plot(t_eval, th_plot, color=COLORS[name], lw=1.8,
            label=f"{name}  (extrap RMSE={preds[name]['extrap']:.4f})")
ax.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("θ (rad)", fontsize=11)
ax.set_title("Trajectory rollout", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Trajectory: θ̇(t)
ax = axes[1]
ax.plot(t_eval, td_true, color=COLORS["True"], lw=1, ls="--",
        label="Ground truth", zorder=5)
for name in ["LNN-F", "LNN", "VanillaNN"]:
    td_plot = np.clip(preds[name]["theta_dot"], -20, 20)
    ax.plot(t_eval, td_plot, color=COLORS[name], lw=1.8, label=name)
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("θ̇ (rad/s)", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Energy decay
ax = axes[2]
ax.plot(t_eval, E_true, color=COLORS["True"], lw=1, ls="--",
        label="Ground truth (dissipating)", zorder=5)
for name in ["LNN-F", "LNN", "VanillaNN"]:
    ax.plot(t_eval, preds[name]["E"], color=COLORS[name], lw=1.8,
            label=f"{name}  (mean |ΔE/E|={preds[name]['eerr']:.3f})")
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("Total energy (J)", fontsize=11)
ax.set_title("Energy — friction causes monotone decay", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Absolute error (log scale)
ax = axes[3]
for name in ["LNN-F", "LNN", "VanillaNN"]:
    err = np.abs(np.clip(preds[name]["theta"], -50, 50) - th_true)
    ax.semilogy(t_eval, err + 1e-6, color=COLORS[name], lw=1.6, label=name)
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("|θ error|  (log scale)", fontsize=11)
ax.set_xlabel("Time (s)", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
