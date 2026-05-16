"""
pinn_identification.py
======================
PINN as a system identifier: learning the damping coefficient c from data.

System  : m·ẍ + c·ẋ + k·x = 0   (m=1 kg, k=400 N/m known;  c=4 N·s/m unknown)
Data    : 10 observations from t ∈ [0, 0.36 s]  (same window as pinn_vs_nn_mck.py)
Eval    : t ∈ [0, 1.0 s]

Two PINN-ID variants are compared:

  PINN-ID (raw)      — c is a plain nn.Parameter, no positivity constraint.
                       Gradient descent can drive c negative (unphysical).

  PINN-ID (softplus) — c = softplus(raw_c) = log(1 + exp(raw_c)) ≥ 0.
                       Strictly positive by construction; gradient flows
                       smoothly; no hard clipping needed.

Plain NN: same architecture, data fit only — no physics, no identification.

Training schedule:
  - Warm-up (first ~15% of epochs): data loss only.
    During this phase c receives no gradient — observe that it stays flat.
  - Physics phase: full loss (data + physics).
    c begins converging once the ODE residual is active.
This makes visible that the ODE residual is the *sole* identification signal.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from systems import MassSpringDamper

SEED   = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {"True": "#2c3e50", "raw": "#e67e22", "softplus": "#9b59b6", "NN": "#e74c3c"}

# ---------------------------------------------------------------------------
# True system
# ---------------------------------------------------------------------------

M       = 1.0        # kg    (known)
K       = 400.0      # N/m   (known)
C_TRUE  = 4.0        # N·s/m (unknown — to be identified)
C_INIT  = 1.0        # deliberate wrong initial guess

mck = MassSpringDamper(delta=C_TRUE / (2*M), omega0=np.sqrt(K / M))

T_TRAIN = (0.0, 0.36)
T_EVAL  = (0.0, 1.0)
N_TRAIN, N_EVAL = 10, 500

t_train = np.linspace(*T_TRAIN, N_TRAIN)
x_train = mck.solution(t_train)
t_eval  = np.linspace(*T_EVAL,  N_EVAL)
x_true  = mck.solution(t_eval)

t_tr = torch.tensor(t_train[:, None], dtype=torch.float32, device=DEVICE)
x_tr = torch.tensor(x_train[:, None], dtype=torch.float32, device=DEVICE)
t_ev = torch.tensor(t_eval[:,  None], dtype=torch.float32, device=DEVICE)

# ---------------------------------------------------------------------------
# Shared network builder
# ---------------------------------------------------------------------------

def _make_net(hidden_dim=64, n_layers=4):
    layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, 1))
    return nn.Sequential(*layers)

# ---------------------------------------------------------------------------
# PINN-ID (raw) — unconstrained c
# ---------------------------------------------------------------------------

class PINN_ID_Raw(nn.Module):
    """
    c is a plain nn.Parameter.
    No positivity constraint: gradient descent can drive c negative.
    """

    def __init__(self, m, k, c_init=1.0, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.m, self.k = m, k
        self.t_scale   = t_scale
        self.c         = nn.Parameter(torch.tensor(float(c_init)))
        self.net       = _make_net(hidden_dim, n_layers)

    def forward(self, t):
        return self.net(t / self.t_scale)

    def residual(self, t_c):
        t_r = t_c.detach().requires_grad_(True)
        x   = self.forward(t_r)
        xd  = torch.autograd.grad(x.sum(),  t_r, create_graph=True)[0]
        xdd = torch.autograd.grad(xd.sum(), t_r, create_graph=True)[0]
        return (self.m * xdd + self.c * xd + self.k * x) / self.k

    def total_loss(self, t_obs, x_obs, t_c, w_data=1.0, w_phys=1.0):
        ld = ((self.forward(t_obs) - x_obs)**2).mean()
        lp = (self.residual(t_c)**2).mean()
        return w_data*ld + w_phys*lp, ld, lp

    def c_value(self):
        return self.c.item()


# ---------------------------------------------------------------------------
# PINN-ID (softplus) — positivity-constrained c
# ---------------------------------------------------------------------------

class PINN_ID_Softplus(nn.Module):
    """
    c = softplus(raw_c) = log(1 + exp(raw_c))  ≥  0  always.
    The unconstrained parameter is raw_c; the physical parameter is c.
    """

    def __init__(self, m, k, c_init=1.0, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.m, self.k = m, k
        self.t_scale   = t_scale
        # Invert softplus to initialise raw_c so that softplus(raw_c) = c_init
        raw_init = float(np.log(np.exp(c_init) - 1.0))
        self.raw_c = nn.Parameter(torch.tensor(raw_init))
        self.net   = _make_net(hidden_dim, n_layers)

    @property
    def c(self):
        return F.softplus(self.raw_c)   # always ≥ 0

    def forward(self, t):
        return self.net(t / self.t_scale)

    def residual(self, t_c):
        t_r = t_c.detach().requires_grad_(True)
        x   = self.forward(t_r)
        xd  = torch.autograd.grad(x.sum(),  t_r, create_graph=True)[0]
        xdd = torch.autograd.grad(xd.sum(), t_r, create_graph=True)[0]
        return (self.m * xdd + self.c * xd + self.k * x) / self.k

    def total_loss(self, t_obs, x_obs, t_c, w_data=1.0, w_phys=1.0):
        ld = ((self.forward(t_obs) - x_obs)**2).mean()
        lp = (self.residual(t_c)**2).mean()
        return w_data*ld + w_phys*lp, ld, lp

    def c_value(self):
        return self.c.item()


# ---------------------------------------------------------------------------
# Plain NN
# ---------------------------------------------------------------------------

class TrajNN(nn.Module):
    """Plain trajectory NN: t → x(t), no physics."""

    def __init__(self, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.t_scale = t_scale
        self.net     = _make_net(hidden_dim, n_layers)

    def forward(self, t):
        return self.net(t / self.t_scale)


# ---------------------------------------------------------------------------
# Generic training helper
# ---------------------------------------------------------------------------

def train_pinn_id(model, epochs, warmup, label):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(epochs*.5), int(epochs*.8)], gamma=0.5)
    c_hist = []
    print(f"Training {label}  (c_true={C_TRUE}, c_init={C_INIT}) ...")
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        if ep <= warmup:
            loss = ((model(t_tr) - x_tr)**2).mean()
            ld, lp = loss, torch.tensor(0.0)
        else:
            t_c = torch.rand(500, 1, device=DEVICE) * T_EVAL[1]
            loss, ld, lp = model.total_loss(t_tr, x_tr, t_c)
        loss.backward(); opt.step(); sch.step()
        c_hist.append(model.c_value())
        if ep % 4000 == 0:
            print(f"  ep {ep:5d}  data={ld.item():.4e}  phys={lp.item():.4e}"
                  f"  c={model.c_value():.4f}")
    c_id = model.c_value()
    print(f"  → c identified = {c_id:.4f}  (true = {C_TRUE}, "
          f"error = {abs(c_id - C_TRUE)/C_TRUE*100:.2f}%)\n")
    return c_hist


# ---------------------------------------------------------------------------
# Train all models
# ---------------------------------------------------------------------------

EPOCHS_PINN, WARMUP = 20_000, 3_000
EPOCHS_NN           = 3_000

pinn_raw = PINN_ID_Raw(M, K, c_init=C_INIT, t_scale=T_EVAL[1]).to(DEVICE)
c_hist_raw = train_pinn_id(pinn_raw, EPOCHS_PINN, WARMUP, "PINN-ID (raw)")

pinn_sp = PINN_ID_Softplus(M, K, c_init=C_INIT, t_scale=T_EVAL[1]).to(DEVICE)
c_hist_sp = train_pinn_id(pinn_sp, EPOCHS_PINN, WARMUP, "PINN-ID (softplus)")

nn_model = TrajNN(t_scale=T_TRAIN[1]).to(DEVICE)
opt_nn   = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
sch_nn   = torch.optim.lr_scheduler.MultiStepLR(
    opt_nn, milestones=[int(EPOCHS_NN*.5), int(EPOCHS_NN*.8)], gamma=0.5)
print("Training NN ...")
for ep in range(1, EPOCHS_NN + 1):
    opt_nn.zero_grad()
    loss_nn = ((nn_model(t_tr) - x_tr)**2).mean()
    loss_nn.backward(); opt_nn.step(); sch_nn.step()
print(f"  final loss={loss_nn.item():.4e}\n")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

with torch.no_grad():
    x_raw = pinn_raw(t_ev).cpu().numpy().squeeze()
    x_sp  = pinn_sp(t_ev).cpu().numpy().squeeze()
    x_nn  = nn_model(t_ev).cpu().numpy().squeeze()

c_raw_id = pinn_raw.c_value()
c_sp_id  = pinn_sp.c_value()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

split = T_TRAIN[1]
in_m, out_m = t_eval <= split, t_eval > split

def rmse(pred, mask):
    return float(np.sqrt(np.mean((pred[mask] - x_true[mask])**2)))

print(f"\n{'Model':<20} {'Interp RMSE':>13} {'Extrap RMSE':>13} {'c identified':>14}")
print("-" * 64)
for name, pred, c_id in [("PINN-ID (raw)",      x_raw, c_raw_id),
                          ("PINN-ID (softplus)",  x_sp,  c_sp_id),
                          ("NN",                  x_nn,  None)]:
    c_str = f"{c_id:.4f}" if c_id is not None else "      —"
    print(f"{name:<20} {rmse(pred, in_m):>13.4f} {rmse(pred, out_m):>13.4f} {c_str:>14}")

# ---------------------------------------------------------------------------
# Plot — 3 panels: trajectory, error, c convergence
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(13, 9))
fig.suptitle(
    f"PINN System Identification — m·ẍ + c·ẋ + k·x = 0\n"
    f"m={M} kg,  k={K} N/m  known  |  c unknown  "
    f"(true={C_TRUE}, init={C_INIT})  |  10 obs from [0, 0.36 s]",
    fontsize=12, fontweight="bold",
)
gs  = fig.add_gridspec(3, 1, hspace=0.45)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2])

# --- trajectory ---
ax1.plot(t_eval, x_true, color=COLORS["True"], lw=1, ls="--", label="Ground truth", zorder=5)
for name, pred, col, c_id in [("PINN-ID (raw)",     x_raw, COLORS["raw"],      c_raw_id),
                                ("PINN-ID (softplus)", x_sp,  COLORS["softplus"], c_sp_id),
                                ("NN",                 x_nn,  COLORS["NN"],       None)]:
    c_str = f"c={c_id:.3f}" if c_id is not None else ""
    ax1.plot(t_eval, pred, color=col, lw=1.8,
             label=f"{name}  {c_str}  extrap RMSE={rmse(pred, out_m):.4f}")
ax1.scatter(t_train, x_train, s=60, color=COLORS["True"], zorder=10,
            edgecolors="white", linewidths=0.5, label=f"{N_TRAIN} obs")
ax1.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
ax1.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax1.set_ylabel("x(t)  [m]", fontsize=11)
ax1.set_title("Trajectory rollout", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8.5); ax1.grid(True, alpha=0.3)

# --- absolute error ---
for name, pred, col in [("PINN-ID (raw)",     x_raw, COLORS["raw"]),
                         ("PINN-ID (softplus)", x_sp,  COLORS["softplus"]),
                         ("NN",                 x_nn,  COLORS["NN"])]:
    ax2.semilogy(t_eval, np.abs(pred - x_true) + 1e-6, color=col, lw=1.6, label=name)
ax2.axvline(split, color="gray", ls=":", lw=1.4)
ax2.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax2.set_ylabel("|error|  (log scale)", fontsize=11)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.legend(fontsize=8.5); ax2.grid(True, alpha=0.3)

# --- c convergence ---
epochs = np.arange(1, EPOCHS_PINN + 1)
ax3.plot(epochs, c_hist_raw, color=COLORS["raw"],      lw=1.5, label="raw c")
ax3.plot(epochs, c_hist_sp,  color=COLORS["softplus"], lw=1.5, label="softplus c")
ax3.axhline(C_TRUE, color=COLORS["True"], lw=1.2, ls="--", label=f"c true = {C_TRUE}")
ax3.axhline(C_INIT, color="gray",         lw=1.0, ls=":",  label=f"c init = {C_INIT}")
ax3.axvline(WARMUP, color="gray", ls="--", lw=1.0, label="warm-up end")
ax3.set_xlabel("Epoch", fontsize=11)
ax3.set_ylabel("c  [N·s/m]", fontsize=11)
ax3.set_title("Damping coefficient identification — raw vs softplus", fontsize=11, fontweight="bold")
ax3.legend(fontsize=8.5); ax3.grid(True, alpha=0.3)

plt.tight_layout()
_fp = os.path.join(RESULTS_DIR, "pinn_identification.png")
fig.savefig(_fp, dpi=150, bbox_inches="tight")
print(f"\nSaved: {_fp}")
plt.show()
