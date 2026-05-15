"""
delan_friction_pendulum.py
==========================
DeLaN with explicit friction vs plain DeLaN vs VanillaNN on a dissipative
simple pendulum (1-DOF).

System  : Simple pendulum with viscous friction
          θ̈ = −(g/L) sin(θ) − b · θ̇
          Parameters: g=9.81, L=1, b=0.3
Data    : 20 (θ, θ̇, θ̈) samples from t ∈ [0, 3 s]
          θ₀=2.5 rad (near-separatrix — same as lnn_vs_vanilla.py)
Eval    : RK4 rollout over t ∈ [0, 12 s]

DeLaN is a stricter structure than LNN: it explicitly decomposes L = T − V,
where T uses an SPD mass matrix M(q) parameterised via Cholesky, and V(q) ≥ 0
via softplus.  This guarantees physically consistent kinetic and potential
energy terms, not just a Lagrangian that 'happens to work'.

Three models compared:
  DeLaN-F   — DeLaN + learnable Rayleigh dissipation matrix B(q) (PSD via Cholesky)
              EOM:  M(q) θ̈ = dL/dθ − C θ̇ − B(q) θ̇
              B(q) > 0 by construction → energy strictly decreasing
  DeLaN     — standard conservative DeLaN (no friction structure)
              Strongest conservative Lagrangian structure, but still wrong model
  VanillaNN — unstructured (θ, θ̇) → θ̈ regression

Key insight: neither VanillaNN nor the conservative DeLaN can correctly
extrapolate a dissipative trajectory.  DeLaN-F adds one structured component
(a learned PSD dissipation term) and recovers the correct long-run behaviour.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from systems import SimplePendulum
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
# DeLaN with Rayleigh dissipation  (general n_dof, used here with n_dof=1)
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

    Structured energy decomposition:
        T(q, q̇) = ½ q̇ᵀ M(q) q̇       M SPD via Cholesky: M = LLᵀ + εI
        V(q)    = V_net(q) ≥ 0         softplus output
        D(q, q̇) = ½ q̇ᵀ B(q) q̇        B SPD via Cholesky: B = L_B L_Bᵀ + ε_B I

    Equations of motion:
        M(q) q̈ = ∂L/∂q − [∂/∂q(∂L/∂q̇)] q̇ − B(q) q̇

    B SPD ⟹  dE/dt = −q̇ᵀ B q̇ ≤ 0  (energy can only decrease — by design).
    Works for any n_dof ≥ 1.
    """

    def __init__(self, n_dof=1, hidden_dim=64, n_layers=3,
                 activation=nn.Softplus, eps=1e-3, eps_b=1e-3):
        super().__init__()
        self.n_dof = n_dof
        self.eps   = eps
        self.eps_b = eps_b

        n_chol = n_dof * (n_dof + 1) // 2
        self.mass_net      = _mlp(n_dof, hidden_dim, n_chol, n_layers, activation)
        self.potential_net = _mlp(n_dof, hidden_dim, 1,      n_layers, activation)
        self.friction_net  = _mlp(n_dof, hidden_dim, n_chol, n_layers, activation)

    def _cholesky_matrix(self, entries, eps):
        B, n = entries.shape[0], self.n_dof
        Lmat = torch.zeros(B, n, n, device=entries.device, dtype=entries.dtype)
        idx = 0
        for i in range(n):
            for j in range(i + 1):
                Lmat[:, i, j] = (torch.exp(entries[:, idx]) if i == j
                                 else entries[:, idx])
                idx += 1
        return Lmat @ Lmat.transpose(-1, -2) + eps * torch.eye(n, device=entries.device)

    def mass_matrix(self, q):
        return self._cholesky_matrix(self.mass_net(q), self.eps)

    def friction_matrix(self, q):
        return self._cholesky_matrix(self.friction_net(q), self.eps_b)

    def potential_energy(self, q):
        return F.softplus(self.potential_net(q))

    def lagrangian(self, q, qdot):
        M  = self.mass_matrix(q)
        qd = qdot.unsqueeze(-1)
        T  = 0.5 * (qd.transpose(-1, -2) @ M @ qd).squeeze(-1)
        V  = self.potential_energy(q)
        return T - V

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

        M_mat    = torch.stack(M_rows, dim=1)
        coriolis = torch.cat(C_rows, dim=-1)

        B_mat    = self.friction_matrix(q_r)
        friction = (B_mat @ qdot_r.unsqueeze(-1)).squeeze(-1)

        eps_I = 1e-4 * torch.eye(n, device=q.device, dtype=q.dtype)
        rhs   = (dL_dq - coriolis - friction).unsqueeze(-1)
        return torch.linalg.solve(M_mat + eps_I, rhs).squeeze(-1)

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
    "DeLaN-F":   DeLaNFriction(n_dof=1, hidden_dim=64, n_layers=3, activation=nn.Softplus),
    "DeLaN":     DeLaN(        n_dof=1, hidden_dim=64, n_layers=3, activation=nn.Softplus),
    "VanillaNN": VanillaNN(    n_dof=1, hidden_dim=64, n_layers=3),
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
gain = preds["VanillaNN"]["extrap"] / (preds["DeLaN-F"]["extrap"] + 1e-12)
print(f"\nDeLaN-F extrap improvement over VanillaNN: {gain:.1f}×")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
fig.suptitle(
    f"DeLaN with friction vs DeLaN vs VanillaNN — Dissipative Simple Pendulum\n"
    f"b={pend.b}  |  θ₀={THETA0} rad (near-separatrix)  |  "
    f"{N_TRAIN} training pts from [0, {T_TRAIN[1]} s]  |  shaded = extrapolation",
    fontsize=12, fontweight="bold",
)

ax = axes[0]
ax.plot(t_eval, th_true, color=COLORS["True"], lw=1, ls="--",
        label="Ground truth", zorder=5)
for name in ["DeLaN-F", "DeLaN", "VanillaNN"]:
    th_plot = np.clip(preds[name]["theta"], -15, 15)
    ax.plot(t_eval, th_plot, color=COLORS[name], lw=1.8,
            label=f"{name}  (extrap RMSE={preds[name]['extrap']:.4f})")
ax.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("θ (rad)", fontsize=11)
ax.set_title("Trajectory rollout", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_eval, td_true, color=COLORS["True"], lw=1, ls="--",
        label="Ground truth", zorder=5)
for name in ["DeLaN-F", "DeLaN", "VanillaNN"]:
    td_plot = np.clip(preds[name]["theta_dot"], -20, 20)
    ax.plot(t_eval, td_plot, color=COLORS[name], lw=1.8, label=name)
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("θ̇ (rad/s)", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(t_eval, E_true, color=COLORS["True"], lw=1, ls="--",
        label="Ground truth (dissipating)", zorder=5)
for name in ["DeLaN-F", "DeLaN", "VanillaNN"]:
    ax.plot(t_eval, preds[name]["E"], color=COLORS[name], lw=1.8,
            label=f"{name}  (mean |ΔE/E|={preds[name]['eerr']:.3f})")
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("Total energy (J)", fontsize=11)
ax.set_title("Energy — friction causes monotone decay", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[3]
for name in ["DeLaN-F", "DeLaN", "VanillaNN"]:
    err = np.abs(np.clip(preds[name]["theta"], -50, 50) - th_true)
    ax.semilogy(t_eval, err + 1e-6, color=COLORS[name], lw=1.6, label=name)
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("|θ error|  (log scale)", fontsize=11)
ax.set_xlabel("Time (s)", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Animation — dissipative pendulum physical view (DeLaN)
# ---------------------------------------------------------------------------
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

_L = pend.L
_s = max(1, len(t_eval) // 200)
_ta   = t_eval[::_s]
_pnames = ["True", "DeLaN-F", "DeLaN", "VanillaNN"]
_thd  = {nm: (th_true[::_s] if nm=="True" else preds[nm]["theta"][::_s])
         for nm in _pnames}
_nf = len(_ta)

fig_a, (ax_ph, ax_tr) = plt.subplots(1, 2, figsize=(13, 6))
fig_a.suptitle(
    f"DeLaN-F vs DeLaN vs VanillaNN — dissipative pendulum  (b={pend.b})",
    fontsize=11)

ax_ph.set_xlim(-1.4*_L, 1.4*_L); ax_ph.set_ylim(-1.4*_L, 0.3*_L)
ax_ph.set_aspect("equal"); ax_ph.axis("off")
ax_ph.plot(0, 0, 'k+', ms=10, mew=2, zorder=10)
_circ = plt.Circle((0,0), _L, color="lightgray", fill=False, ls="--", lw=0.8)
ax_ph.add_patch(_circ)
_rods, _bobs = {}, {}
for nm in _pnames:
    rod, = ax_ph.plot([], [], '-', color=COLORS[nm],
                      lw=(2 if nm=="True" else 1.5),
                      ls=("--" if nm=="True" else "-"), zorder=3)
    bob, = ax_ph.plot([], [], 'o', color=COLORS[nm],
                      ms=(10 if nm=="True" else 8),
                      markeredgecolor="white", markeredgewidth=0.5, zorder=5)
    _rods[nm] = rod; _bobs[nm] = bob
_leg_h = [ax_ph.plot([], [], '-', color=COLORS[nm], label=nm)[0] for nm in _pnames]
ax_ph.legend(handles=_leg_h, fontsize=8, loc="lower right")
_ttx = ax_ph.text(0.02, 0.98, "", transform=ax_ph.transAxes,
                  ha="left", va="top", fontsize=9, color="gray")

ax_tr.set_xlim(t_eval[0], t_eval[-1]); ax_tr.set_ylim(-4, 3)
ax_tr.set_xlabel("t (s)"); ax_tr.set_ylabel("θ (rad)"); ax_tr.grid(True, alpha=0.3)
ax_tr.axvline(split, color="gray", ls=":", lw=1.2)
ax_tr.plot(t_eval, th_true, color=COLORS["True"], lw=1, ls="--", label="True")
for nm in ["DeLaN-F", "DeLaN", "VanillaNN"]:
    ax_tr.plot(t_eval, np.clip(preds[nm]["theta"], -15, 15),
               color=COLORS[nm], lw=1.5, label=nm)
ax_tr.legend(fontsize=8)
_cur, = ax_tr.plot([], [], color="k", lw=1.2, zorder=10)

def _upd(i):
    for nm in _pnames:
        th = float(_thd[nm][i])
        bx = _L * np.sin(th); by = -_L * np.cos(th)
        _rods[nm].set_data([0, bx], [0, by])
        _bobs[nm].set_data([bx], [by])
    _cur.set_data([_ta[i], _ta[i]], ax_tr.get_ylim())
    _ttx.set_text(f"t={_ta[i]:.2f} s")

_anim = FuncAnimation(fig_a, _upd, frames=_nf, interval=40, blit=False)
_ap = os.path.join(RESULTS_DIR, "delan_friction_pendulum_anim.mp4")
try:
    _anim.save(_ap, writer=FFMpegWriter(fps=25, bitrate=1800))
except Exception:
    _ap = _ap.replace(".mp4", ".gif"); _anim.save(_ap, writer=PillowWriter(fps=20))
print(f"Saved: {_ap}")
plt.show()
