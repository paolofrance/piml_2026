"""
pinn_vs_nn.py
=============
PINN vs plain NN on the damped harmonic oscillator.

System  : ẍ + 2δẋ + ω₀²x = 0   (δ=2, ω₀=20, underdamped)
Data    : 10 observations from t ∈ [0, 0.36]  (first ~1 oscillation)
Eval    : t ∈ [0, 1.0]           (extrapolation to t=1)

PINN: trajectory model  t → x(t),  physics residual enforced over [0, 1].
NN  : same architecture, data fit only — no ODE knowledge.

Key insight: PINN enforces the ODE everywhere in the eval domain, so it
extrapolates the decaying oscillation correctly. The plain NN memorises
the 10 training points and collapses outside the training window.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from systems import HarmonicOscillator

SEED   = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {"True": "#2c3e50", "PINN": "#3498db", "NN": "#e74c3c"}

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """Trajectory PINN: t → x(t).  Residual: ẍ + 2δẋ + ω₀²x = 0."""

    def __init__(self, delta, omega0, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.delta, self.omega0, self.t_scale = delta, omega0, t_scale
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t / self.t_scale)

    def residual(self, t_c):
        t_r = t_c.detach().requires_grad_(True)
        x   = self.forward(t_r)
        xd  = torch.autograd.grad(x.sum(),  t_r, create_graph=True)[0]
        xdd = torch.autograd.grad(xd.sum(), t_r, create_graph=True)[0]
        # Normalise by ω₀² so the residual is O(1) regardless of frequency
        return (xdd + 2*self.delta*xd + self.omega0**2 * x) / self.omega0**2

    def total_loss(self, t_obs, x_obs, t_c, w_data=1.0, w_phys=1.0):
        ld = ((self.forward(t_obs) - x_obs)**2).mean()
        lp = (self.residual(t_c)**2).mean()
        return w_data*ld + w_phys*lp, ld, lp


class TrajNN(nn.Module):
    """Plain trajectory NN: t → x(t), no physics."""

    def __init__(self, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.t_scale = t_scale
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t / self.t_scale)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

HO      = HarmonicOscillator(delta=2.0, omega0=20.0)
T_TRAIN = (0.0, 0.36)
T_EVAL  = (0.0, 1.0)
N_TRAIN, N_EVAL = 10, 500

t_train = np.linspace(*T_TRAIN, N_TRAIN)
x_train = HO.solution(t_train)
t_eval  = np.linspace(*T_EVAL, N_EVAL)
x_true  = HO.solution(t_eval)

t_tr = torch.tensor(t_train[:, None], dtype=torch.float32, device=DEVICE)
x_tr = torch.tensor(x_train[:, None], dtype=torch.float32, device=DEVICE)
t_ev = torch.tensor(t_eval[:,  None], dtype=torch.float32, device=DEVICE)

# ---------------------------------------------------------------------------
# Train PINN  (two-phase: data warm-up → data + physics over full domain)
# ---------------------------------------------------------------------------

EPOCHS_PINN, WARMUP = 20_000, 3_000

pinn = PINN(HO.delta, HO.omega0, t_scale=T_EVAL[1]).to(DEVICE)
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
    if ep % 4000 == 0:
        print(f"  ep {ep:5d}  data={ld.item():.4e}  phys={lp.item():.4e}")

# ---------------------------------------------------------------------------
# Train plain NN  (data only)
# ---------------------------------------------------------------------------

EPOCHS_NN = 3_000

nn_model = TrajNN(t_scale=T_TRAIN[1]).to(DEVICE)
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
    x_pinn = pinn(t_ev).cpu().numpy().squeeze()
    x_nn   = nn_model(t_ev).cpu().numpy().squeeze()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

split = T_TRAIN[1]
in_m, out_m = t_eval <= split, t_eval > split

metrics = {
    "PINN": {"interp": float(np.sqrt(np.mean((x_pinn[in_m]  - x_true[in_m]) **2))),
             "extrap": float(np.sqrt(np.mean((x_pinn[out_m] - x_true[out_m])**2)))},
    "NN":   {"interp": float(np.sqrt(np.mean((x_nn[in_m]    - x_true[in_m]) **2))),
             "extrap": float(np.sqrt(np.mean((x_nn[out_m]   - x_true[out_m])**2)))},
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("PINN vs plain NN — Damped Harmonic Oscillator\n"
             "10 observations from [0, 0.36]  |  shaded = extrapolation",
             fontsize=12, fontweight="bold")

ax1.plot(t_eval, x_true,  color=COLORS["True"], lw=1, ls="--", label="Ground truth", zorder=5)
ax1.plot(t_eval, x_pinn,  color=COLORS["PINN"], lw=1.8, ls="--", label=f"PINN  (extrap RMSE={metrics['PINN']['extrap']:.4f})")
ax1.plot(t_eval, x_nn,    color=COLORS["NN"],   lw=1.8, ls="--", label=f"NN    (extrap RMSE={metrics['NN']['extrap']:.4f})")
ax1.scatter(t_train, x_train, s=60, color=COLORS["True"], zorder=10,
            label=f"{N_TRAIN} training obs", edgecolors="white", linewidths=0.5)
ax1.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
ax1.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax1.set_ylabel("x(t)", fontsize=11)
ax1.set_title("Trajectory rollout", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

for name, pred, c in [("PINN", x_pinn, COLORS["PINN"]), ("NN", x_nn, COLORS["NN"])]:
    err = np.abs(pred - x_true)
    ax2.semilogy(t_eval, err + 1e-6, color=c, lw=1.6, label=name)
ax2.axvline(split, color="gray", ls=":", lw=1.4)
ax2.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax2.set_ylabel("|error|  (log scale)", fontsize=11)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

plt.tight_layout()
# p = os.path.join(RESULTS_DIR, "pinn_vs_nn.png")
# plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
# print(f"\nSaved: {p}")
plt.plot()
plt.show()