"""
pinn_identification.py
======================
PINN as a system identifier: learning the damping coefficient from data.

System  : ẍ + 2δẋ + ω₀²x = 0   (ω₀=20 known, δ=2 unknown)
Data    : 10 observations from t ∈ [0, 0.36]  (same as pinn_vs_nn.py)
Eval    : t ∈ [0, 1.0]

PINN-ID : treats δ as a learnable nn.Parameter alongside the network weights.
          The ODE residual drives δ towards its true value; no extra supervision.
Plain NN: same architecture, data fit only — no physics, no identification.

Key insight: the physics constraint acts as a self-supervised signal that
simultaneously regularises the trajectory AND identifies the hidden parameter.
Even from 10 observations PINN-ID recovers δ ≈ 2.0 and extrapolates correctly.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from systems import MassSpringDamper

SEED   = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {"True": "#2c3e50", "PINN-ID": "#9b59b6", "NN": "#e74c3c"}

# ---------------------------------------------------------------------------
# True system
# ---------------------------------------------------------------------------

DELTA_TRUE = 2.0
OMEGA0     = 20.0
DELTA_INIT = 0.5          # intentionally wrong starting guess

mck      = MassSpringDamper(delta=DELTA_TRUE, omega0=OMEGA0)
T_TRAIN = (0.0, 0.36)
T_EVAL  = (0.0, 1.0)
N_TRAIN, N_EVAL = 10, 500

t_train = np.linspace(*T_TRAIN, N_TRAIN)
x_train = mck.solution(t_train)
t_eval  = np.linspace(*T_EVAL, N_EVAL)
x_true  = mck.solution(t_eval)

t_tr = torch.tensor(t_train[:, None], dtype=torch.float32, device=DEVICE)
x_tr = torch.tensor(x_train[:, None], dtype=torch.float32, device=DEVICE)
t_ev = torch.tensor(t_eval[:,  None], dtype=torch.float32, device=DEVICE)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PINN_ID(nn.Module):
    """
    Trajectory PINN with learnable damping coefficient δ.

    ω₀ is assumed known; δ is a free nn.Parameter initialised far from truth.
    The ODE residual propagates gradient back into δ, identifying it from data.
    """

    def __init__(self, omega0, delta_init=0.5, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.omega0   = omega0
        self.t_scale  = t_scale
        # learnable δ — constrained positive via softplus in the residual
        self.log_delta = nn.Parameter(torch.tensor(np.log(delta_init), dtype=torch.float32))

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    @property
    def delta(self):
        return torch.exp(self.log_delta)   # always positive

    def forward(self, t):
        return self.net(t / self.t_scale)

    def residual(self, t_c):
        t_r = t_c.detach().requires_grad_(True)
        x   = self.forward(t_r)
        xd  = torch.autograd.grad(x.sum(),  t_r, create_graph=True)[0]
        xdd = torch.autograd.grad(xd.sum(), t_r, create_graph=True)[0]
        # normalise by ω₀² so residual is O(1) regardless of frequency
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
# Train PINN-ID  (two-phase: data warm-up → data + physics)
# ---------------------------------------------------------------------------

EPOCHS_PINN, WARMUP = 20_000, 3_000

pinn_id = PINN_ID(OMEGA0, delta_init=DELTA_INIT, t_scale=T_EVAL[1]).to(DEVICE)
opt     = torch.optim.Adam(pinn_id.parameters(), lr=1e-3)
sch     = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=[int(EPOCHS_PINN*.5), int(EPOCHS_PINN*.8)], gamma=0.5)

delta_history = []   # track identified δ over epochs

print(f"Training PINN-ID  (δ_true={DELTA_TRUE}, δ_init={DELTA_INIT}) ...")
for ep in range(1, EPOCHS_PINN + 1):
    opt.zero_grad()
    if ep <= WARMUP:
        loss = ((pinn_id(t_tr) - x_tr)**2).mean()
        ld, lp = loss, torch.tensor(0.0)
    else:
        t_c = torch.rand(500, 1, device=DEVICE) * T_EVAL[1]
        loss, ld, lp = pinn_id.total_loss(t_tr, x_tr, t_c, w_data=1.0, w_phys=1.0)
    loss.backward(); opt.step(); sch.step()
    delta_history.append(pinn_id.delta.item())
    if ep % 4000 == 0:
        print(f"  ep {ep:5d}  data={ld.item():.4e}  phys={lp.item():.4e}"
              f"  δ_id={pinn_id.delta.item():.4f}")

delta_identified = pinn_id.delta.item()
print(f"\nIdentified δ = {delta_identified:.4f}  (true = {DELTA_TRUE})")
print(f"Relative error: {abs(delta_identified - DELTA_TRUE)/DELTA_TRUE*100:.2f}%")

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
    x_pinn_id = pinn_id(t_ev).cpu().numpy().squeeze()
    x_nn      = nn_model(t_ev).cpu().numpy().squeeze()

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

split = T_TRAIN[1]
in_m, out_m = t_eval <= split, t_eval > split

metrics = {
    "PINN-ID": {
        "interp": float(np.sqrt(np.mean((x_pinn_id[in_m] - x_true[in_m])**2))),
        "extrap": float(np.sqrt(np.mean((x_pinn_id[out_m] - x_true[out_m])**2))),
        "delta":  delta_identified,
    },
    "NN": {
        "interp": float(np.sqrt(np.mean((x_nn[in_m] - x_true[in_m])**2))),
        "extrap": float(np.sqrt(np.mean((x_nn[out_m] - x_true[out_m])**2))),
        "delta":  None,
    },
}

print(f"\n{'Model':<10} {'Interp RMSE':>13} {'Extrap RMSE':>13} {'δ identified':>14}")
print("-" * 54)
for name, m in metrics.items():
    d_str = f"{m['delta']:.4f}" if m["delta"] is not None else "      —"
    print(f"{name:<10} {m['interp']:>13.4f} {m['extrap']:>13.4f} {d_str:>14}")
gain = metrics["NN"]["extrap"] / (metrics["PINN-ID"]["extrap"] + 1e-12)
print(f"\nPINN-ID extrap improvement over NN: {gain:.1f}×")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(12, 9))
fig.suptitle(
    f"PINN System Identification — Damped Mass-Spring-Damper (MCK)\n"
    f"ω₀={OMEGA0} known,  δ unknown  (true={DELTA_TRUE}, init={DELTA_INIT})  |  "
    f"10 observations from [0, 0.36]  |  shaded = extrapolation",
    fontsize=12, fontweight="bold",
)

gs = fig.add_gridspec(3, 1, hspace=0.45)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2])   # δ convergence — independent x-axis

# --- trajectory ---
ax1.plot(t_eval, x_true,     color=COLORS["True"],    lw=1,   ls="--", label="Ground truth", zorder=5)
ax1.plot(t_eval, x_pinn_id,  color=COLORS["PINN-ID"], lw=1.8, ls="-",
         label=f"PINN-ID  (δ_id={delta_identified:.3f},  extrap RMSE={metrics['PINN-ID']['extrap']:.4f})")
ax1.plot(t_eval, x_nn,       color=COLORS["NN"],      lw=1.8, ls="-",
         label=f"NN        (extrap RMSE={metrics['NN']['extrap']:.4f})")
ax1.scatter(t_train, x_train, s=60, color=COLORS["True"], zorder=10,
            label=f"{N_TRAIN} training obs", edgecolors="white", linewidths=0.5)
ax1.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
ax1.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax1.set_ylabel("x(t)", fontsize=11)
ax1.set_title("Trajectory rollout", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# --- absolute error ---
for name, pred, c in [("PINN-ID", x_pinn_id, COLORS["PINN-ID"]), ("NN", x_nn, COLORS["NN"])]:
    err = np.abs(pred - x_true)
    ax2.semilogy(t_eval, err + 1e-6, color=c, lw=1.6, label=name)
ax2.axvline(split, color="gray", ls=":", lw=1.4)
ax2.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax2.set_ylabel("|error|  (log scale)", fontsize=11)
ax2.set_xlabel("Time (s)", fontsize=11)
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

# --- δ convergence ---
epochs = np.arange(1, EPOCHS_PINN + 1)
ax3.plot(epochs, delta_history, color=COLORS["PINN-ID"], lw=1.5, label="δ identified")
ax3.axhline(DELTA_TRUE, color=COLORS["True"], lw=1.2, ls="--", label=f"δ true = {DELTA_TRUE}")
ax3.axhline(DELTA_INIT, color="gray",         lw=1.0, ls=":",  label=f"δ init = {DELTA_INIT}")
ax3.axvline(WARMUP, color="gray", ls="--", lw=1.0, label="warm-up end")
ax3.set_xlabel("Epoch", fontsize=11)
ax3.set_ylabel("δ", fontsize=11)
ax3.set_title("Damping coefficient identification", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Animation — mass-spring physical view
# ---------------------------------------------------------------------------
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

_s = max(1, len(t_eval) // 200)
_ta = t_eval[::_s]
_xd = {"True": x_true[::_s], "PINN-ID": x_pinn_id[::_s], "NN": x_nn[::_s]}
_nf = len(_ta)
_rows = {"True": 2.0, "PINN-ID": 1.0, "NN": 0.0}
_WX, _R = -1.7, 0.10

def _sp1d(x0, x1, y, n=8, a=0.04):
    N = n*2+2; xs = np.linspace(x0, x1, N); ys = np.full(N, y, dtype=float)
    ys[1:-1:2] += a; ys[2:-1:2] -= a; return xs, ys

fig_a, (ax_ph, ax_tr) = plt.subplots(1, 2, figsize=(13, 5))
fig_a.suptitle(
    f"PINN-ID vs NN — MCK  (δ_true={DELTA_TRUE}, δ_id={delta_identified:.3f})",
    fontsize=11)
ax_ph.set_xlim(-2.1, 1.6); ax_ph.set_ylim(-0.5, 2.6)
ax_ph.set_aspect("equal"); ax_ph.axis("off")
ax_ph.vlines(_WX, -0.3, 2.6, colors="k", lw=4)
for nm, y in _rows.items():
    ax_ph.text(_WX-0.07, y, nm, ha="right", va="center",
               color=COLORS[nm], fontsize=9, fontweight="bold")
_spL, _bob = {}, {}
for nm, y in _rows.items():
    _spL[nm], = ax_ph.plot(*_sp1d(_WX, 0, y), color=COLORS[nm], lw=1.5)
    _bob[nm], = ax_ph.plot([], [], 'o', color=COLORS[nm], ms=14, zorder=5)
_ttx = ax_ph.text(0.98, 0.02, "", transform=ax_ph.transAxes,
                  ha="right", va="bottom", fontsize=9, color="gray")

_ylo = min(x_true.min(), x_pinn_id.min(), x_nn.min()) - 0.05
_yhi = max(x_true.max(), x_pinn_id.max(), x_nn.max()) + 0.05
ax_tr.set_xlim(t_eval[0], t_eval[-1]); ax_tr.set_ylim(_ylo, _yhi)
ax_tr.set_xlabel("t (s)"); ax_tr.set_ylabel("x(t)"); ax_tr.grid(True, alpha=0.3)
ax_tr.axvline(T_TRAIN[1], color="gray", ls=":", lw=1.2)
ax_tr.plot(t_eval, x_true,    color=COLORS["True"],    lw=1,   ls="--", label="True")
ax_tr.plot(t_eval, x_pinn_id, color=COLORS["PINN-ID"], lw=1.5, label="PINN-ID")
ax_tr.plot(t_eval, x_nn,      color=COLORS["NN"],      lw=1.5, label="NN")
ax_tr.legend(fontsize=8)
_cur, = ax_tr.plot([], [], color="k", lw=1.2, zorder=10)

def _upd(i):
    for nm, y in _rows.items():
        xv = float(np.clip(_xd[nm][i], -1.6, 1.5))
        _spL[nm].set_data(*_sp1d(_WX, xv-_R, y))
        _bob[nm].set_data([xv], [y])
    _cur.set_data([_ta[i], _ta[i]], [_ylo, _yhi])
    _ttx.set_text(f"t={_ta[i]:.3f} s")

_anim = FuncAnimation(fig_a, _upd, frames=_nf, interval=40, blit=False)
_ap = os.path.join(RESULTS_DIR, "pinn_identification_anim.mp4")
try:
    _anim.save(_ap, writer=FFMpegWriter(fps=25, bitrate=1800))
except Exception:
    _ap = _ap.replace(".mp4", ".gif"); _anim.save(_ap, writer=PillowWriter(fps=20))
print(f"Saved: {_ap}")
plt.show()
