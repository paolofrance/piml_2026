"""
pinn_2dof_spring_damper.py
==========================
PINN vs plain NN on the 2-DOF mass-spring-damper system.

System  :   M ẍ + C ẋ + K x = 0

          [ m1   0 ] ẍ + [ c1+c2  -c2 ] ẋ + [ k1+k2  -k2 ] x = 0
          [  0  m2 ]     [ -c2     c2 ]     [ -k2   k2+k3]

          m1=m2=1,  k1=6, k2=4, k3=6,  c1=0.4, c2=0.2
          → K eigenvalues: 6 and 14  (ω₁≈2.45, ω₂≈3.74 rad/s)
          → lightly damped: two coupled decaying oscillation modes

Data    : 15 observations of (x₁, x₂) from t ∈ [0, 2.5 s]
          (roughly one period of the slower mode)
Eval    : t ∈ [0, 10.0 s]

PINN: trajectory model  t → [x₁(t), x₂(t)].
      Physics residuals for both DOFs enforced via autograd over [0, 10].
      Residual normalised by max(eigenvalues of K) = 14 so both loss terms
      stay O(1) during training.
NN  : same architecture, data fit only — no ODE knowledge.

Two-phase training (same strategy as pinn_vs_nn.py):
  Phase 1 (warm-up) : data loss only   → network learns rough trajectory shape
  Phase 2 (physics) : data + physics   → ODE enforced across full eval domain
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

SEED   = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {"True": "#2c3e50", "PINN": "#3498db", "NN": "#e74c3c"}

# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

M_SYS = np.array([[1.0, 0.0],
                   [0.0, 1.0]])
K_SYS = np.array([[10.0, -4.0],
                   [-4.0, 10.0]])    # eigenvalues 6, 14
C_SYS = np.array([[ 0.6, -0.2],
                   [-0.2,  0.2]])

K_NORM = float(np.linalg.eigvalsh(K_SYS).max())   # 14.0 — residual normaliser

def true_solution(t_eval, x0, xdot0):
    """Integrate M ẍ + C ẋ + K x = 0 with scipy."""
    def ode(t, s):
        x, xd = s[:2], s[2:]
        xdd = np.linalg.solve(M_SYS, -C_SYS @ xd - K_SYS @ x)
        return np.concatenate([xd, xdd])
    sol = solve_ivp(ode, (t_eval[0], t_eval[-1]),
                    np.concatenate([x0, xdot0]),
                    t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12)
    return sol.y[:2].T   # (N, 2)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

X0    = np.array([1.0, 0.5])
XDOT0 = np.array([0.0, 0.0])
T_TRAIN = (0.0, 2.5)
T_EVAL  = (0.0, 10.0)
N_TRAIN, N_EVAL = 15, 500

t_train = np.linspace(*T_TRAIN, N_TRAIN)
t_eval  = np.linspace(*T_EVAL,  N_EVAL)

x_train = true_solution(t_train, X0, XDOT0)   # (N_TRAIN, 2)
x_true  = true_solution(t_eval,  X0, XDOT0)   # (N_EVAL, 2)

t_tr = torch.tensor(t_train[:, None], dtype=torch.float32, device=DEVICE)
x_tr = torch.tensor(x_train,          dtype=torch.float32, device=DEVICE)
t_ev = torch.tensor(t_eval[:,  None], dtype=torch.float32, device=DEVICE)

M_t = torch.tensor(M_SYS, dtype=torch.float32, device=DEVICE)
C_t = torch.tensor(C_SYS, dtype=torch.float32, device=DEVICE)
K_t = torch.tensor(K_SYS, dtype=torch.float32, device=DEVICE)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PINN2DOF(nn.Module):
    """
    PINN for the 2-DOF mass-spring-damper.
    Input : t  (B, 1)
    Output: x  (B, 2)  — both displacements simultaneously.
    """

    def __init__(self, M, C, K, k_norm, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.M, self.C, self.K = M, C, K
        self.k_norm  = k_norm
        self.t_scale = t_scale

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t / self.t_scale)   # (B, 2)

    def residual(self, t_c):
        """
        Physics residual  r = M ẍ + C ẋ + K x,  normalised by k_norm.
        Compute ẋ and ẍ per DOF via scalar autograd on t.
        Returns: (B, 2)
        """
        t_r = t_c.detach().requires_grad_(True)   # (B, 1)
        x   = self.forward(t_r)                   # (B, 2)

        xdot_cols, xddot_cols = [], []
        for i in range(2):
            dxi   = torch.autograd.grad(x[:, i].sum(),   t_r, create_graph=True)[0]  # (B,1)
            ddxi  = torch.autograd.grad(dxi.sum(),        t_r, create_graph=True)[0]  # (B,1)
            xdot_cols.append(dxi)
            xddot_cols.append(ddxi)

        xdot  = torch.cat(xdot_cols,  dim=-1)   # (B, 2)
        xddot = torch.cat(xddot_cols, dim=-1)   # (B, 2)

        # r = (M ẍ + C ẋ + K x) row-wise:  r[b] = M @ xddot[b] + ...
        res = xddot @ self.M.T + xdot @ self.C.T + x @ self.K.T   # (B, 2)
        return res / self.k_norm

    def total_loss(self, t_obs, x_obs, t_c, w_data=1.0, w_phys=1.0):
        ld = ((self.forward(t_obs) - x_obs)**2).mean()
        lp = (self.residual(t_c)**2).mean()
        return w_data*ld + w_phys*lp, ld, lp


class TrajNN2DOF(nn.Module):
    """Plain trajectory NN: t → [x₁(t), x₂(t)], no physics."""

    def __init__(self, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.t_scale = t_scale
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t / self.t_scale)

# ---------------------------------------------------------------------------
# Train PINN  (two-phase: data warm-up → data + physics)
# ---------------------------------------------------------------------------

EPOCHS_PINN, WARMUP = 30_000, 5_000

pinn = PINN2DOF(M_t, C_t, K_t, K_NORM, t_scale=T_EVAL[1]).to(DEVICE)
opt  = torch.optim.Adam(pinn.parameters(), lr=1e-3)
sch  = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=[int(EPOCHS_PINN*.5), int(EPOCHS_PINN*.8)], gamma=0.5)

print("Training PINN ...")
for ep in range(1, EPOCHS_PINN + 1):
    opt.zero_grad()
    if ep <= WARMUP:
        loss = ((pinn(t_tr) - x_tr)**2).mean()
        ld, lp = loss, torch.tensor(0.0)
    else:
        t_c = torch.rand(500, 1, device=DEVICE) * T_EVAL[1]
        loss, ld, lp = pinn.total_loss(t_tr, x_tr, t_c, w_data=1.0, w_phys=1.0)
    loss.backward(); opt.step(); sch.step()
    if ep % 6000 == 0:
        print(f"  ep {ep:5d}  data={ld.item():.4e}  phys={lp.item():.4e}")

# ---------------------------------------------------------------------------
# Train plain NN  (data only)
# ---------------------------------------------------------------------------

EPOCHS_NN = 5_000

nn_model = TrajNN2DOF(t_scale=T_TRAIN[1]).to(DEVICE)
opt_nn   = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
sch_nn   = torch.optim.lr_scheduler.MultiStepLR(
    opt_nn, milestones=[int(EPOCHS_NN*.5), int(EPOCHS_NN*.8)], gamma=0.5)

print("\nTraining NN ...")
for ep in range(1, EPOCHS_NN + 1):
    opt_nn.zero_grad()
    loss_nn = ((nn_model(t_tr) - x_tr)**2).mean()
    loss_nn.backward(); opt_nn.step(); sch_nn.step()
print(f"  final loss={loss_nn.item():.4e}")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

with torch.no_grad():
    x_pinn = pinn(t_ev).cpu().numpy()      # (N_EVAL, 2)
    x_nn   = nn_model(t_ev).cpu().numpy()  # (N_EVAL, 2)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

split = T_TRAIN[1]
in_m, out_m = t_eval <= split, t_eval > split

def rmse(pred, true, mask):
    return float(np.sqrt(np.mean((pred[mask] - true[mask])**2)))

metrics = {
    "PINN": {
        "interp": rmse(x_pinn, x_true, in_m),
        "extrap": rmse(x_pinn, x_true, out_m),
    },
    "NN": {
        "interp": rmse(x_nn, x_true, in_m),
        "extrap": rmse(x_nn, x_true, out_m),
    },
}

print(f"\n{'Model':<8} {'Interp RMSE':>13} {'Extrap RMSE':>13}")
print("-" * 36)
for name, m in metrics.items():
    print(f"{name:<8} {m['interp']:>13.4f} {m['extrap']:>13.4f}")
gain = metrics["NN"]["extrap"] / (metrics["PINN"]["extrap"] + 1e-12)
print(f"\nPINN extrap improvement over NN: {gain:.1f}×")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
fig.suptitle(
    "PINN vs plain NN — 2-DOF Mass-Spring-Damper\n"
    f"k₁={6}, k₂={4}, k₃={6}  |  c₁={0.4}, c₂={0.2}  |  "
    f"{N_TRAIN} observations from [0, {T_TRAIN[1]} s]  |  shaded = extrapolation",
    fontsize=12, fontweight="bold",
)

labels_pinn = [f"PINN  (extrap RMSE={metrics['PINN']['extrap']:.4f})",
               "PINN"]
labels_nn   = [f"NN    (extrap RMSE={metrics['NN']['extrap']:.4f})",
               "NN"]

for ax, ci, dof_label in zip(axes[:2], [0, 1], ["x₁ (m)", "x₂ (m)"]):
    ax.plot(t_eval, x_true[:, ci], color=COLORS["True"], lw=1, ls="--",
            label="Ground truth", zorder=5)
    ax.plot(t_eval, x_pinn[:, ci], color=COLORS["PINN"], lw=1.8,
            label=labels_pinn[ci])
    ax.plot(t_eval, x_nn[:, ci],   color=COLORS["NN"],   lw=1.8,
            label=labels_nn[ci])
    ax.scatter(t_train, x_train[:, ci], s=50, color=COLORS["True"], zorder=10,
               edgecolors="white", linewidths=0.5,
               label=f"{N_TRAIN} obs" if ci == 0 else None)
    ax.axvline(split, color="gray", ls=":", lw=1.4, label="train end" if ci == 0 else None)
    ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
    ax.set_ylabel(dof_label, fontsize=11)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[2]
for name, pred, c in [("PINN", x_pinn, COLORS["PINN"]), ("NN", x_nn, COLORS["NN"])]:
    err = np.sqrt(np.sum((pred - x_true)**2, axis=1))
    ax.semilogy(t_eval, err + 1e-6, color=c, lw=1.6, label=name)
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("‖[x₁,x₂] error‖  (log scale)", fontsize=11)
ax.set_xlabel("Time (s)", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
