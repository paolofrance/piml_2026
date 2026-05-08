"""
lnn_friction.py
===============
LNN with implicit friction learning vs plain LNN vs VanillaNN on a dissipative
simple pendulum (1-DOF).

System  : Simple pendulum with viscous friction
          θ̈ = −(g/L) sin(θ) − b · θ̇
          Parameters: g=9.81, L=1, b=0.3
Data    : 20 (θ, θ̇, θ̈) samples from t ∈ [0, 3 s]
          θ₀=2.5 rad (near-separatrix)
Eval    : RK4 rollout over t ∈ [0, 12 s]

Friction method (Ragusano et al., LNN-MBPO, Politecnico di Milano 2024):
  Instead of constraining friction to the Rayleigh form b(θ)·θ̇ > 0, a separate
  unconstrained DNN F(θ,θ̇) is trained jointly with the Lagrangian network.
  The DNN is supervised by the "implied friction" derived directly from the data:

      F_impl = M(θ,θ̇) · (θ̈_true − θ̈_LNN)

  where M = ∂²L/∂θ̇² is the scalar mass from the Lagrangian.
  Combined prediction: θ̈ = θ̈_LNN + F_hat / M

  Joint loss:
      l_dyn  = MSE(θ̈_LNN + F_hat/M, θ̈_true)   — trains both networks
      l_fric = MSE(F_hat, F_impl)                — trains friction DNN only
      l_total = α · l_dyn + β · l_fric

  No positivity constraint — the network can learn arbitrary generalised forces.

Three models compared:
  LNN-F   — LNN + implicit friction DNN (paper approach)
  LNN     — standard conservative LNN (no friction)
  VanillaNN — unstructured (θ, θ̇) → θ̈ regression
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_func
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

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
    """Simple pendulum with viscous friction: θ̈ = -(g/L)sin(θ) - b·θ̇."""

    def __init__(self, g=9.81, length=1.0, b=0.3):
        super().__init__(g=g, length=length)
        self.b = b

    def acceleration(self, theta, theta_dot):
        return -(self.g / self.L) * np.sin(theta) - self.b * theta_dot


# ---------------------------------------------------------------------------
# LNN with implicit friction (Ragusano et al.)
# ---------------------------------------------------------------------------

def _mlp(in_dim, hidden_dim, out_dim, n_layers, act):
    layers = [nn.Linear(in_dim, hidden_dim), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class LNNFriction(nn.Module):
    """
    1-DOF LNN + unconstrained friction DNN.

    The Lagrangian network learns the conservative part of the dynamics.
    A separate DNN F(θ, θ̇) — no positivity constraint — estimates the
    generalised friction force.  It is trained to match the implied friction
    derived from the residual between data and the conservative LNN prediction.

    Forward:    θ̈ = θ̈_LNN + F_hat / M
    Friction target:  F_impl = M · (θ̈_true − θ̈_LNN)
    Joint loss:  α · MSE(θ̈_full, θ̈_true) + β · MSE(F_hat, F_impl)
    """

    def __init__(self, hidden_dim=64, n_layers=3, eps=1e-2, alpha=1.0, beta=0.5):
        super().__init__()
        self.eps   = eps
        self.alpha = alpha
        self.beta  = beta
        self.lag_net  = _mlp(2, hidden_dim, 1, n_layers, nn.Softplus)
        self.fric_net = _mlp(2, hidden_dim, 1, n_layers, nn.Tanh)

    # ---- Euler-Lagrange step, returns (qddot, M) ----

    def _el_forward(self, q, qdot):
        """Compute θ̈_LNN and mass M = ∂²L/∂θ̇² via autograd."""
        q_r    = q.detach().requires_grad_(True)
        qdot_r = qdot.detach().requires_grad_(True)

        L_val = self.lag_net(torch.cat([q_r, qdot_r], dim=-1))
        L_sum = L_val.sum()

        dL_dqdot = torch.autograd.grad(L_sum, qdot_r, create_graph=True)[0]
        dL_dq    = torch.autograd.grad(L_sum, q_r,    create_graph=True)[0]

        M = torch.autograd.grad(dL_dqdot[:, 0].sum(), qdot_r,
                                create_graph=True)[0]                          # (B,1)
        C = torch.autograd.grad(dL_dqdot[:, 0].sum(), q_r,
                                create_graph=True)[0] * qdot_r                # (B,1)

        qddot_lnn = (dL_dq - C) / (M + self.eps)
        return qddot_lnn, M

    # ---- Training loss (joint) ----

    def compute_loss(self, q, qdot, qddot_true):
        qddot_lnn, M = self._el_forward(q, qdot)
        F_hat        = self.fric_net(torch.cat([q, qdot], dim=-1))

        # Full prediction: θ̈_full = θ̈_LNN + F_hat / M
        qddot_full = qddot_lnn + F_hat / (M + self.eps)
        l_dyn      = F_func.mse_loss(qddot_full, qddot_true)

        # Implied friction — detach LNN quantities so only fric_net gets gradient
        F_impl = M.detach() * (qddot_true - qddot_lnn.detach())
        l_fric = F_func.mse_loss(F_hat, F_impl)

        return self.alpha * l_dyn + self.beta * l_fric, l_dyn, l_fric

    # ---- Inference ----

    def predict_acceleration(self, q, qdot):
        qddot_lnn, M = self._el_forward(q, qdot)
        F_hat        = self.fric_net(torch.cat([q, qdot], dim=-1))
        return qddot_lnn + F_hat / (M + self.eps)


# ---------------------------------------------------------------------------
# Custom training loop for LNNFriction
# ---------------------------------------------------------------------------

def train_lnn_friction(model, dataset, n_epochs=3000, batch_size=16,
                       lr=1e-3, device="cpu", verbose=True, log_every=500):
    model = model.to(device)
    model.train()

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(-1)

    ds     = TensorDataset(_t(dataset["theta"]),
                           _t(dataset["theta_dot"]),
                           _t(dataset["theta_ddot"]))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    history = {"train_loss": [], "l_dyn": [], "l_fric": [], "wall_time": 0.0}
    t0 = time.time()

    for ep in range(1, n_epochs + 1):
        ep_tot = ep_dyn = ep_fric = 0.0
        for q_b, qd_b, qdd_b in loader:
            q_b   = q_b.to(device)
            qd_b  = qd_b.to(device)
            qdd_b = qdd_b.to(device)
            opt.zero_grad()
            total, l_dyn, l_fric = model.compute_loss(q_b, qd_b, qdd_b)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()
            n_b = q_b.shape[0]
            ep_tot  += total.item() * n_b
            ep_dyn  += l_dyn.item() * n_b
            ep_fric += l_fric.item() * n_b
        N = len(ds)
        history["train_loss"].append(ep_tot  / N)
        history["l_dyn"].append(ep_dyn  / N)
        history["l_fric"].append(ep_fric / N)
        sch.step()
        if verbose and ep % log_every == 0:
            print(f"  Epoch {ep:5d}/{n_epochs}  total={ep_tot/N:.4e}"
                  f"  dyn={ep_dyn/N:.4e}  fric={ep_fric/N:.4e}")

    history["wall_time"] = time.time() - t0
    return history


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

print("Training LNN-F (implicit friction, paper approach) ...")
lnn_f    = LNNFriction(hidden_dim=64, n_layers=3, alpha=1.0, beta=0.5)
hist_f   = train_lnn_friction(lnn_f, train_ds, n_epochs=EPOCHS, batch_size=16,
                               lr=1e-3, device=DEVICE, verbose=True, log_every=500)
print(f"  loss={hist_f['train_loss'][-1]:.4e}  t={hist_f['wall_time']:.1f}s")

models_base = {
    "LNN":       LNN(      n_dof=1, hidden_dim=64, n_layers=3),
    "VanillaNN": VanillaNN(n_dof=1, hidden_dim=64, n_layers=3),
}
for name, model in models_base.items():
    print(f"\nTraining {name} ...")
    hist = train_dynamics_model(model, train_ds, n_epochs=EPOCHS,
                                batch_size=16, device=DEVICE,
                                verbose=True, log_every=500)
    print(f"  loss={hist['train_loss'][-1]:.4e}  t={hist['wall_time']:.1f}s")

all_models = {"LNN-F": lnn_f, **models_base}

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
for name, model in all_models.items():
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
# Learned friction profile (on true eval trajectory)
# ---------------------------------------------------------------------------

lnn_f.eval()
with torch.no_grad():
    th_t = torch.tensor(th_true, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    td_t = torch.tensor(td_true, dtype=torch.float32).unsqueeze(-1).to(DEVICE)
    F_hat_eval = lnn_f.fric_net(torch.cat([th_t, td_t], dim=-1)).cpu().numpy().squeeze()

# True generalised friction force for pendulum (m=1, M_true=1): F = -b·θ̇
F_true_eval = -pend.b * td_true

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(5, 1, figsize=(11, 17), sharex=True)
fig.suptitle(
    f"LNN-F (implicit friction) vs LNN vs VanillaNN — Dissipative Simple Pendulum\n"
    f"b={pend.b}  |  θ₀={THETA0} rad  |  "
    f"{N_TRAIN} training pts from [0, {T_TRAIN[1]} s]  |  shaded = extrapolation",
    fontsize=12, fontweight="bold",
)

ax = axes[0]
ax.plot(t_eval, th_true, color=COLORS["True"], lw=1, ls="--", label="Ground truth", zorder=5)
for name in ["LNN-F", "LNN", "VanillaNN"]:
    ax.plot(t_eval, np.clip(preds[name]["theta"], -15, 15), color=COLORS[name], lw=1.8,
            label=f"{name}  (extrap RMSE={preds[name]['extrap']:.4f})")
ax.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("θ (rad)", fontsize=11)
ax.set_title("Trajectory rollout", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(t_eval, td_true, color=COLORS["True"], lw=1, ls="--", label="Ground truth", zorder=5)
for name in ["LNN-F", "LNN", "VanillaNN"]:
    ax.plot(t_eval, np.clip(preds[name]["theta_dot"], -20, 20),
            color=COLORS[name], lw=1.8, label=name)
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("θ̇ (rad/s)", fontsize=11)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(t_eval, E_true, color=COLORS["True"], lw=1, ls="--",
        label="Ground truth (dissipating)", zorder=5)
for name in ["LNN-F", "LNN", "VanillaNN"]:
    ax.plot(t_eval, preds[name]["E"], color=COLORS[name], lw=1.8,
            label=f"{name}  (|ΔE/E|={preds[name]['eerr']:.3f})")
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("Total energy (J)", fontsize=11)
ax.set_title("Energy — friction causes monotone decay", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[3]
ax.plot(t_eval, F_true_eval, color=COLORS["True"], lw=1.5, ls="--",
        label=f"True: F = −b·θ̇  (b={pend.b})", zorder=5)
ax.plot(t_eval, F_hat_eval, color=COLORS["LNN-F"], lw=1.8,
        label="Learned: F̂(θ, θ̇)")
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("F_friction", fontsize=11)
ax.set_title("Learned vs true generalised friction force", fontsize=10)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[4]
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

# ---------------------------------------------------------------------------
# Animation — dissipative pendulum physical view
# ---------------------------------------------------------------------------
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

_L = pend.L
_s = max(1, len(t_eval) // 200)
_ta   = t_eval[::_s]
_pnames = ["True", "LNN-F", "LNN", "VanillaNN"]
_thd  = {nm: (th_true[::_s] if nm == "True" else preds[nm]["theta"][::_s])
         for nm in _pnames}
_nf = len(_ta)

fig_a, (ax_ph, ax_tr) = plt.subplots(1, 2, figsize=(13, 6))
fig_a.suptitle("LNN-F vs LNN vs VanillaNN — dissipative pendulum  (b=0.3)", fontsize=11)

ax_ph.set_xlim(-1.4*_L, 1.4*_L); ax_ph.set_ylim(-1.4*_L, 0.3*_L)
ax_ph.set_aspect("equal"); ax_ph.axis("off")
ax_ph.plot(0, 0, 'k+', ms=10, mew=2, zorder=10)
_circ = plt.Circle((0, 0), _L, color="lightgray", fill=False, ls="--", lw=0.8)
ax_ph.add_patch(_circ)
_rods, _bobs = {}, {}
for nm in _pnames:
    rod, = ax_ph.plot([], [], '-', color=COLORS[nm],
                      lw=(2 if nm == "True" else 1.5),
                      ls=("--" if nm == "True" else "-"), zorder=3)
    bob, = ax_ph.plot([], [], 'o', color=COLORS[nm],
                      ms=(10 if nm == "True" else 8),
                      markeredgecolor="white", markeredgewidth=0.5, zorder=5)
    _rods[nm] = rod; _bobs[nm] = bob
_leg_ax = [ax_ph.plot([], [], '-', color=COLORS[nm], label=nm)[0] for nm in _pnames]
ax_ph.legend(handles=_leg_ax, fontsize=8, loc="lower right")
_ttx = ax_ph.text(0.02, 0.98, "", transform=ax_ph.transAxes,
                  ha="left", va="top", fontsize=9, color="gray")

ax_tr.set_xlim(t_eval[0], t_eval[-1]); ax_tr.set_ylim(-4, 3)
ax_tr.set_xlabel("t (s)"); ax_tr.set_ylabel("θ (rad)"); ax_tr.grid(True, alpha=0.3)
ax_tr.axvline(split, color="gray", ls=":", lw=1.2)
ax_tr.plot(t_eval, th_true, color=COLORS["True"], lw=1, ls="--", label="True")
for nm in ["LNN-F", "LNN", "VanillaNN"]:
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
_ap = os.path.join(RESULTS_DIR, "lnn_friction_anim.mp4")
try:
    _anim.save(_ap, writer=FFMpegWriter(fps=25, bitrate=1800))
except Exception:
    _ap = _ap.replace(".mp4", ".gif"); _anim.save(_ap, writer=PillowWriter(fps=20))
print(f"Saved: {_ap}")
plt.show()
