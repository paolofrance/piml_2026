"""
lnn_vs_vanilla.py
=================
LNN vs VanillaNN on the simple pendulum.

System  : θ̈ = -(g/L) sin(θ)   (g=9.81, L=1, conservative)
Data    : 20 (q, qdot, qddot) samples from t ∈ [0, 3 s]
          θ₀=2.5 rad  (near-separatrix — strongly nonlinear)
Eval    : RK4 rollout over t ∈ [0, 12 s]

LNN learns a scalar Lagrangian L(q, q̇) and derives the equations of motion
via Euler-Lagrange + autograd — energy conservation is encoded by design.
VanillaNN fits the same (q, q̇) → q̈ mapping without any physics structure.

Key insight: with only 20 data points and a highly nonlinear trajectory,
LNN's Lagrangian structure keeps the rollout on the correct energy shell far
beyond the training window.  VanillaNN drifts and diverges.
"""

import os, sys
import numpy as np
import torch
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from systems import SimplePendulum
from models import LNN, VanillaNN
from utils.training import train_dynamics_model

SEED   = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {"True": "#2c3e50", "LNN": "#3498db", "VanillaNN": "#e74c3c"}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

pend   = SimplePendulum(g=9.81, length=1.0)
THETA0, TDOT0 = 2.5, 0.0
T_TRAIN = (0.0, 3.0)
T_EVAL  = (0.0, 12.0)
N_TRAIN, N_EVAL = 20, 1200

traj_train = pend.rollout(THETA0, TDOT0, T_TRAIN, N_TRAIN)
traj_eval  = pend.rollout(THETA0, TDOT0, T_EVAL,  N_EVAL)

train_ds = {"theta":      traj_train["theta"],
            "theta_dot":  traj_train["theta_dot"],
            "theta_ddot": traj_train["theta_ddot"]}

t_eval  = traj_eval["t"]
th_true = traj_eval["theta"]
td_true = traj_eval["theta_dot"]
E_true  = pend.total_energy(th_true, td_true)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

EPOCHS = 2_000

models = {
    "LNN":       LNN(      n_dof=1, hidden_dim=64, n_layers=3),
    "VanillaNN": VanillaNN(n_dof=1, hidden_dim=64, n_layers=3),
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
    th, td = rk4(model, THETA0, TDOT0, t_eval, DEVICE)
    E_pred   = pend.total_energy(th, td)
    rmse_in  = float(np.sqrt(np.mean((th[in_m]  - th_true[in_m]) **2)))
    rmse_out = float(np.sqrt(np.mean((th[out_m] - th_true[out_m])**2)))
    e_drift  = float(np.mean(np.abs(E_pred - E_true[0]) / (np.abs(E_true[0]) + 1e-8)))
    preds[name] = {"theta": th, "E": E_pred,
                   "interp": rmse_in, "extrap": rmse_out, "edrift": e_drift}

print(f"\n{'Model':<12} {'Interp RMSE':>13} {'Extrap RMSE':>13} {'E-drift':>10}")
print("-" * 52)
for name, p in preds.items():
    print(f"{name:<12} {p['interp']:>13.4f} {p['extrap']:>13.4f} {p['edrift']:>10.4f}")
gain = preds["VanillaNN"]["extrap"] / (preds["LNN"]["extrap"] + 1e-12)
print(f"\nLNN extrap improvement over VanillaNN: {gain:.1f}×")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle(f"LNN vs VanillaNN — Simple Pendulum  (θ₀=2.5 rad, near-separatrix)\n"
             f"{N_TRAIN} training pts from [0, 3 s]  |  shaded = extrapolation",
             fontsize=12, fontweight="bold")

ax1.plot(t_eval, th_true, color=COLORS["True"], lw=1, ls="--", label="Ground truth", zorder=5)
for name in ["LNN", "VanillaNN"]:
    p = preds[name]
    th_plot = np.clip(p["theta"], -15, 15)
    ax1.plot(t_eval, th_plot, color=COLORS[name], lw=1.8, ls="--",
             label=f"{name}  (extrap RMSE={p['extrap']:.4f})")
ax1.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
ax1.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax1.set_ylabel("θ (rad)", fontsize=11)
ax1.set_title("Trajectory rollout", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

for name in ["LNN", "VanillaNN"]:
    err = np.abs(np.clip(preds[name]["theta"], -50, 50) - th_true)
    ax2.semilogy(t_eval, err + 1e-6, color=COLORS[name], lw=1.6, label=name)
ax2.axvline(split, color="gray", ls=":", lw=1.4)
ax2.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax2.set_ylabel("|error|  (log scale)", fontsize=11)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "lnn_vs_vanilla.png"), dpi=150, bbox_inches="tight")
print(f"Saved: {os.path.join(RESULTS_DIR, 'lnn_vs_vanilla.png')}")
plt.show()

# ---------------------------------------------------------------------------
# Individual figures per model
# ---------------------------------------------------------------------------

for _name in ["LNN", "VanillaNN"]:
    _p = preds[_name]
    _fi, (_a1, _a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    _fi.suptitle(f"{_name} — Simple Pendulum  (θ₀=2.5 rad, near-separatrix)\n"
                 f"{N_TRAIN} training pts from [0, {T_TRAIN[1]} s]  |  shaded = extrapolation",
                 fontsize=12, fontweight="bold")
    _a1.plot(t_eval, th_true, color=COLORS["True"], lw=1, ls="--", label="Ground truth", zorder=5)
    _a1.plot(t_eval, np.clip(_p["theta"], -15, 15), color=COLORS[_name], lw=1.8,
             label=f"{_name}  (extrap RMSE={_p['extrap']:.4f})")
    _a1.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
    _a1.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
    _a1.set_ylabel("θ (rad)", fontsize=11); _a1.legend(fontsize=9); _a1.grid(True, alpha=0.3)
    _err = np.abs(np.clip(_p["theta"], -50, 50) - th_true)
    _a2.semilogy(t_eval, _err + 1e-6, color=COLORS[_name], lw=1.6, label=_name)
    _a2.axvline(split, color="gray", ls=":", lw=1.4)
    _a2.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
    _a2.set_ylabel("|error|  (log scale)", fontsize=11)
    _a2.set_xlabel("Time (s)", fontsize=11)
    _a2.legend(fontsize=9); _a2.grid(True, alpha=0.3)
    plt.tight_layout()
    _fp = os.path.join(RESULTS_DIR, f"lnn_vs_vanilla_{_name}.png")
    _fi.savefig(_fp, dpi=150, bbox_inches="tight")
    print(f"Saved: {_fp}")
    plt.close(_fi)

# ---------------------------------------------------------------------------
# Animation — pendulum physical view
# ---------------------------------------------------------------------------
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

_L = pend.L
_s = max(1, len(t_eval) // 200)
_ta   = t_eval[::_s]
_thd  = {"True": th_true[::_s],
         "LNN":       preds["LNN"]["theta"][::_s],
         "VanillaNN": preds["VanillaNN"]["theta"][::_s]}
_nf = len(_ta)
_pnames = ["True", "LNN", "VanillaNN"]

fig_a, (ax_ph, ax_tr) = plt.subplots(1, 2, figsize=(13, 6))
fig_a.suptitle("LNN vs VanillaNN — simple pendulum  (True / LNN / VanillaNN)", fontsize=11)

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
_leg_ax = [ax_ph.plot([], [], '-', color=COLORS[nm], label=nm)[0] for nm in _pnames]
ax_ph.legend(handles=_leg_ax, fontsize=8, loc="lower right")
_ttx = ax_ph.text(0.02, 0.98, "", transform=ax_ph.transAxes,
                  ha="left", va="top", fontsize=9, color="gray")

_thlo = th_true.min() - 0.2; _thhi = th_true.max() + 0.2
ax_tr.set_xlim(t_eval[0], t_eval[-1])
ax_tr.set_ylim(np.clip(min(p["theta"].min() for p in preds.values())-0.3, -10, _thlo),
               np.clip(max(p["theta"].max() for p in preds.values())+0.3, _thhi, 10))
ax_tr.set_xlabel("t (s)"); ax_tr.set_ylabel("θ (rad)"); ax_tr.grid(True, alpha=0.3)
ax_tr.axvline(split, color="gray", ls=":", lw=1.2)
ax_tr.plot(t_eval, th_true, color=COLORS["True"], lw=1, ls="--", label="True")
for nm in ["LNN", "VanillaNN"]:
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
_ap = os.path.join(RESULTS_DIR, "lnn_vs_vanilla_anim.mp4")
try:
    _anim.save(_ap, writer=FFMpegWriter(fps=25, bitrate=1800))
except Exception:
    _ap = _ap.replace(".mp4", ".gif"); _anim.save(_ap, writer=PillowWriter(fps=20))
print(f"Saved: {_ap}")
plt.show()
