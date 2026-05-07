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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
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
# p = os.path.join(RESULTS_DIR, "lnn_vs_vanilla.png")
# plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
# print(f"\nSaved: {p}")
plt.plot()
plt.show()
