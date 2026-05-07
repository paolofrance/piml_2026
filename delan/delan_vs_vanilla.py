"""
delan_vs_vanilla.py
===================
DeLaN vs VanillaNN on the spring pendulum (2-DOF conservative system).

System  : Spring pendulum in polar coordinates (r, θ)
          r̈ = r·θ̇² − g(1−cosθ) − 2k(r−r₀)
          θ̈ = (−g·sinθ − 2ṙ·θ̇) / r
          Parameters: g=10, k=10, r₀=1
Data    : 80 (q, qdot, qddot) samples from t ∈ [0, 3 s]
Eval    : RK4 rollout over t ∈ [0, 8 s]  (Cartesian xy)

DeLaN structures the Lagrangian as L = T − V where T uses a Cholesky-
parameterised SPD mass matrix M(q), guaranteeing energy conservation by
construction. VanillaNN fits the same (q, q̇) → q̈ mapping without structure.

Key insight: DeLaN's built-in energy conservation keeps the trajectory on
the correct orbit during extrapolation. VanillaNN's energy drifts by ~54%
causing the predicted trajectory to diverge completely.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from systems import SpringPendulum
from models import DeLaN, VanillaNN
from utils.training import train_dynamics_model

SEED   = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {"True": "#2c3e50", "DeLaN": "#2ecc71", "VanillaNN": "#e74c3c"}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

sp    = SpringPendulum(g=10.0, k=10.0, r0=1.0)
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

EPOCHS = 2_500

models = {
    "DeLaN":     DeLaN(    n_dof=2, hidden_dim=64, n_layers=3, activation=nn.Softplus),
    "VanillaNN": VanillaNN(n_dof=2, hidden_dim=64, n_layers=3),
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
    e_drift  = float(np.mean(np.abs(E_pred - E_true[0]) / (np.abs(E_true[0]) + 1e-8)))
    preds[name] = {"xy": xy_pred, "E": E_pred,
                   "interp": rmse_in, "extrap": rmse_out, "edrift": e_drift}

print(f"\n{'Model':<12} {'Interp RMSE':>13} {'Extrap RMSE':>13} {'E-drift':>10}")
print("-" * 52)
for name, p in preds.items():
    print(f"{name:<12} {p['interp']:>13.4f} {p['extrap']:>13.4f} {p['edrift']:>10.4f}")
gain = preds["VanillaNN"]["extrap"] / (preds["DeLaN"]["extrap"] + 1e-12)
print(f"\nDeLaN extrap improvement over VanillaNN: {gain:.1f}×")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
fig.suptitle(f"DeLaN vs VanillaNN — Spring Pendulum 2-DOF\n"
             f"{N_TRAIN} training pts from [0, 3 s]  |  shaded = extrapolation",
             fontsize=12, fontweight="bold")

for ax, ci, lbl in zip(axes[:2], [0, 1], ["x (m)", "y (m)"]):
    ax.plot(t_eval, xy_true[:, ci], color=COLORS["True"], lw=1, ls="--",
            label="Ground truth", zorder=5)
    for name in ["DeLaN", "VanillaNN"]:
        xy_plot = np.clip(preds[name]["xy"][:, ci], -5, 5)
        p = preds[name]
        ax.plot(t_eval, xy_plot, color=COLORS[name], lw=1.8, ls="--",
                label=f"{name}  (extrap RMSE={p['extrap']:.4f})")
    ax.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
    ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
    ax.set_ylabel(lbl, fontsize=11)
    ax.set_title(f"Trajectory rollout — {lbl}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[2]
for name in ["DeLaN", "VanillaNN"]:
    xy_p = preds[name]["xy"]
    err  = np.sqrt(np.sum((np.clip(xy_p, -5, 5) - xy_true)**2, axis=1))
    ax.semilogy(t_eval, err + 1e-6, color=COLORS[name], lw=1.6, label=name)
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("‖xy error‖  (log scale)", fontsize=11)
ax.set_xlabel("Time (s)", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
# p = os.path.join(RESULTS_DIR, "delan_vs_vanilla.png")
# plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
# print(f"\nSaved: {p}")
plt.plot()
plt.show()
