"""
pinn_spring_pendulum.py
=======================
PINN vs plain NN on the spring pendulum (2-DOF conservative system).

System  : r̈ = r θ̇² − g(1−cosθ) − 2k(r−r₀)
          θ̈ = (−g sinθ − 2ṙ θ̇) / r
          g=10, k=10, r₀=1
Data    : 20 observations of (t, r, θ) from t ∈ [0, 3 s]
          — position only, no velocities or accelerations needed
Eval    : t ∈ [0, 8 s]  (Cartesian xy for comparison with LNN/DeLaN)

PINN: trajectory model  t → [r(t), θ(t)].
      Physics residuals for both ODEs enforced via autograd over [0, 8 s].
NN  : same architecture, data fit only.

Key contrast with LNN/DeLaN (examples/lnn/, examples/delan/):
  - LNN/DeLaN learn a dynamics model (q, q̇) → q̈  and need (q, q̇, q̈) data
  - PINN learns the solution trajectory t → q and needs only position data
  - LNN/DeLaN get energy conservation by construction; PINN does not
  The energy panel below makes this trade-off visible: PINN may extrapolate
  the trajectory shape but will show energy drift, unlike the Lagrangian models.
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from systems import SpringPendulum

SEED   = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {"True": "#2c3e50", "PINN": "#3498db", "NN": "#e74c3c"}

# ---------------------------------------------------------------------------
# System + data
# ---------------------------------------------------------------------------

sp    = SpringPendulum(g=10.0, k=10.0, r0=1.0)
G, K, R0 = sp.g, sp.k, sp.r0

Q0    = np.array([1.1, 0.5])
QDOT0 = np.array([0.0, 0.0])
T_TRAIN = (0.0, 3.0)
T_EVAL  = (0.0, 8.0)
N_TRAIN, N_EVAL = 20, 800

traj_train = sp.rollout(Q0, QDOT0, T_TRAIN, N_TRAIN)
traj_eval  = sp.rollout(Q0, QDOT0, T_EVAL,  N_EVAL)

# Training: only (t, q) — no velocities or accelerations
t_obs  = traj_train["t"]                       # (N_TRAIN,)
q_obs  = traj_train["q"]                       # (N_TRAIN, 2) — [r, θ]

t_eval  = traj_eval["t"]
q_true  = traj_eval["q"]
qd_true = traj_eval["qdot"]
xy_true = traj_eval["xy"]
E_true  = sp.total_energy(q_true, qd_true)

t_tr = torch.tensor(t_obs[:, None],  dtype=torch.float32, device=DEVICE)
q_tr = torch.tensor(q_obs,           dtype=torch.float32, device=DEVICE)
t_ev = torch.tensor(t_eval[:, None], dtype=torch.float32, device=DEVICE)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _mlp(hidden_dim, n_layers, out_dim, act=nn.Tanh):
    layers = [nn.Linear(1, hidden_dim), act()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), act()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class SpringPendulumPINN(nn.Module):
    """Trajectory PINN: t → [r(t), θ(t)].  Residuals from the EL equations."""

    def __init__(self, g, k, r0, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.g, self.k, self.r0 = g, k, r0
        self.t_scale = t_scale
        self.net = _mlp(hidden_dim, n_layers, out_dim=2)

    def forward(self, t):
        return self.net(t / self.t_scale)          # (N, 2): col0=r, col1=θ

    def residual(self, t_c):
        t_r = t_c.detach().requires_grad_(True)
        q   = self.forward(t_r)                    # (N, 2)
        r   = q[:, 0:1];  th  = q[:, 1:2]

        dr   = autograd.grad(r.sum(),  t_r, create_graph=True)[0]
        dth  = autograd.grad(th.sum(), t_r, create_graph=True)[0]
        ddr  = autograd.grad(dr.sum(), t_r, create_graph=True)[0]
        ddth = autograd.grad(dth.sum(), t_r, create_graph=True)[0]

        # r̈ − r θ̇² + g(1−cosθ) + 2k(r−r₀) = 0    normalised by 2k
        R1 = (ddr - r*dth**2 + self.g*(1 - torch.cos(th)) + 2*self.k*(r - self.r0)) \
             / (2*self.k)
        # θ̈ + (g sinθ + 2ṙ θ̇)/r = 0              normalised by g/r₀
        R2 = (ddth + (self.g*torch.sin(th) + 2*dr*dth) / (r.abs() + 1e-4)) \
             / (self.g / self.r0)

        return R1, R2

    def total_loss(self, t_tr, q_tr, t_c, w_data=1.0, w_phys=1.0):
        ld      = ((self.forward(t_tr) - q_tr)**2).mean()
        R1, R2  = self.residual(t_c)
        lp      = (R1**2).mean() + (R2**2).mean()
        return w_data*ld + w_phys*lp, ld, lp


class TrajNN(nn.Module):
    """Plain trajectory NN: t → [r(t), θ(t)], no physics."""

    def __init__(self, hidden_dim=64, n_layers=4, t_scale=1.0):
        super().__init__()
        self.t_scale = t_scale
        self.net = _mlp(hidden_dim, n_layers, out_dim=2)

    def forward(self, t):
        return self.net(t / self.t_scale)

# ---------------------------------------------------------------------------
# Train PINN  (warm-up on data → data + physics)
# ---------------------------------------------------------------------------

EPOCHS_PINN, WARMUP = 50_000, 0

W_DATA = 1.0
W_PHYS = 0.10

pinn = SpringPendulumPINN(G, K, R0, t_scale=T_EVAL[1]).to(DEVICE)
opt  = torch.optim.Adam(pinn.parameters(), lr=1e-3)
sch  = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=[int(EPOCHS_PINN*.5), int(EPOCHS_PINN*.8)], gamma=0.5)

print("Training PINN ...")
for ep in range(1, EPOCHS_PINN + 1):
    opt.zero_grad()
    if ep <= WARMUP:
        loss = ((pinn(t_tr) - q_tr)**2).mean()
    else:
        t_c  = torch.rand(1000, 1, device=DEVICE) * T_EVAL[1]
        loss, ld, lp = pinn.total_loss(t_tr, q_tr, t_c,
                                       w_data=W_DATA,w_phys=W_PHYS)
    loss.backward(); opt.step(); sch.step()
    if ep % 5000 == 0:
        t_c = torch.rand(1000, 1, device=DEVICE) * T_EVAL[1]
        _, ld, lp = pinn.total_loss(t_tr, q_tr, t_c,
                                    w_data=W_DATA,w_phys=W_PHYS)
        print(f"  ep {ep:5d}  data={ld.item():.3e}  phys={lp.item():.3e}")

# ---------------------------------------------------------------------------
# Train NN  (data only)
# ---------------------------------------------------------------------------

EPOCHS_NN = 5_000

nn_model = TrajNN(t_scale=T_TRAIN[1]).to(DEVICE)
opt_nn   = torch.optim.Adam(nn_model.parameters(), lr=1e-3)
sch_nn   = torch.optim.lr_scheduler.MultiStepLR(
    opt_nn, milestones=[int(EPOCHS_NN*.5), int(EPOCHS_NN*.8)], gamma=0.5)

print("\nTraining NN ...")
for ep in range(1, EPOCHS_NN + 1):
    opt_nn.zero_grad()
    loss_nn = ((nn_model(t_tr) - q_tr)**2).mean()
    loss_nn.backward(); opt_nn.step(); sch_nn.step()
print(f"  final loss = {loss_nn.item():.4e}")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

with torch.no_grad():
    q_pinn = pinn(t_ev).cpu().numpy()       # (N_EVAL, 2)
    q_nn   = nn_model(t_ev).cpu().numpy()   # (N_EVAL, 2)

xy_pinn = SpringPendulum.polar_to_xy(q_pinn)
xy_nn   = SpringPendulum.polar_to_xy(q_nn)

# energy: need velocities — estimate via finite differences
qd_pinn = np.gradient(q_pinn, t_eval, axis=0)
qd_nn   = np.gradient(q_nn,   t_eval, axis=0)
E_pinn  = sp.total_energy(q_pinn, qd_pinn)
E_nn    = sp.total_energy(q_nn,   qd_nn)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

split = T_TRAIN[1]
in_m, out_m = t_eval <= split, t_eval > split

preds = {
    "PINN": {"xy": xy_pinn, "E": E_pinn},
    "NN":   {"xy": xy_nn,   "E": E_nn},
}
for name, p in preds.items():
    xy = p["xy"]
    p["interp"] = float(np.sqrt(np.mean((xy[in_m]  - xy_true[in_m])**2)))
    p["extrap"] = float(np.sqrt(np.mean((xy[out_m] - xy_true[out_m])**2)))
    p["edrift"] = float(np.mean(np.abs(p["E"] - E_true[0]) / (np.abs(E_true[0]) + 1e-8)))

print(f"\n{'Model':<8} {'Interp RMSE':>13} {'Extrap RMSE':>13} {'E-drift':>10}")
print("-" * 48)
for name, p in preds.items():
    print(f"{name:<8} {p['interp']:>13.4f} {p['extrap']:>13.4f} {p['edrift']:>10.4f}")
gain = preds["NN"]["extrap"] / (preds["PINN"]["extrap"] + 1e-12)
print(f"\nPINN extrap improvement over NN: {gain:.1f}×")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)
fig.suptitle(
    f"PINN vs NN — Spring Pendulum 2-DOF\n"
    f"{N_TRAIN} position observations from [0, 3 s]  |  shaded = extrapolation",
    fontsize=12, fontweight="bold")

for ax, ci, lbl in zip(axes[:2], [0, 1], ["x (m)", "y (m)"]):
    ax.plot(t_eval, xy_true[:, ci], color=COLORS["True"], lw=1, ls="--",
            label="Ground truth", zorder=5)
    for name, p in preds.items():
        xy_pl = np.clip(p["xy"][:, ci], -5, 5)
        ax.plot(t_eval, xy_pl, color=COLORS[name], lw=1.8,
                label=f"{name}  (extrap={p['extrap']:.4f})")
    ax.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
    ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
    ax.set_ylabel(lbl, fontsize=11); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[2]
for name, p in preds.items():
    err = np.sqrt(np.sum((np.clip(p["xy"], -5, 5) - xy_true)**2, axis=1))
    ax.semilogy(t_eval, err + 1e-6, color=COLORS[name], lw=1.6, label=name)
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("‖xy error‖  (log)", fontsize=11); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[3]
ax.plot(t_eval, E_true, color=COLORS["True"], lw=1, ls="--", label="True (conserved)")
for name, p in preds.items():
    ax.plot(t_eval, p["E"], color=COLORS[name], lw=1.6,
            label=f"{name}  (drift={p['edrift']:.3f})")
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, t_eval[-1], alpha=0.07, color="gray")
ax.set_ylabel("Total energy", fontsize=11); ax.set_xlabel("Time (s)", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_title("Energy — PINN has no conservation guarantee (cf. LNN/DeLaN)",
             fontsize=10, style="italic")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "pinn_spring_pendulum.png"),
            dpi=130, bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------------
# Animation — spring-pendulum orbital view
# ---------------------------------------------------------------------------

_s = max(1, N_EVAL // 200)
_ta  = t_eval[::_s]
_nf  = len(_ta)
_pnames = ["True", "PINN", "NN"]
_xyd = {"True": xy_true[::_s],
        "PINN": np.clip(xy_pinn[::_s], -5, 5),
        "NN":   np.clip(xy_nn[::_s],   -5, 5)}
_TRAIL = 30


def _sp2d(ox, oy, bx, by, n=7, af=0.09):
    dx, dy = bx-ox, by-oy
    L  = np.sqrt(dx**2 + dy**2) + 1e-10
    amp = af * L;  px, py = -dy/L, dx/L
    N  = n*2+2;  t = np.linspace(0, 1, N)
    xs = ox + dx*t;  ys = oy + dy*t
    off = np.zeros(N)
    off[1:-1:2] = amp;  off[2:-1:2] = -amp
    return xs + off*px, ys + off*py


fig_a, (ax_ph, ax_tr) = plt.subplots(1, 2, figsize=(13, 6))
fig_a.suptitle("PINN vs NN — spring pendulum orbital motion", fontsize=11)

_lim = 1.8
ax_ph.set_xlim(-_lim, _lim); ax_ph.set_ylim(-_lim*1.2, 0.3)
ax_ph.set_aspect("equal"); ax_ph.axis("off")
ax_ph.plot(0, 0, 'k+', ms=10, mew=2, zorder=10)

_springs, _bobs, _trails = {}, {}, {}
for nm in _pnames:
    sp_line, = ax_ph.plot([], [], '-', color=COLORS[nm], lw=1.4)
    bob, = ax_ph.plot([], [], 'o', color=COLORS[nm],
                      ms=(11 if nm == "True" else 9),
                      markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    trail, = ax_ph.plot([], [], '-', color=COLORS[nm], lw=0.8, alpha=0.4, zorder=2)
    _springs[nm] = sp_line; _bobs[nm] = bob; _trails[nm] = trail

_leg_h = [ax_ph.plot([], [], '-', color=COLORS[nm], label=nm)[0] for nm in _pnames]
ax_ph.legend(handles=_leg_h, fontsize=8, loc="lower right")
_ttx = ax_ph.text(0.02, 0.98, "", transform=ax_ph.transAxes,
                  ha="left", va="top", fontsize=9, color="gray")

ax_tr.set_xlim(t_eval[0], t_eval[-1]); ax_tr.grid(True, alpha=0.3)
ax_tr.set_xlabel("t (s)"); ax_tr.set_ylabel("x (m)")
ax_tr.axvline(split, color="gray", ls=":", lw=1.2)
ax_tr.plot(t_eval, xy_true[:, 0], color=COLORS["True"], lw=1, ls="--", label="True")
ax_tr.plot(t_eval, np.clip(xy_pinn[:, 0], -5, 5), color=COLORS["PINN"], lw=1.5, label="PINN")
ax_tr.plot(t_eval, np.clip(xy_nn[:,   0], -5, 5), color=COLORS["NN"],   lw=1.5, label="NN")
ax_tr.legend(fontsize=8)
_cur, = ax_tr.plot([], [], color="k", lw=1.2, zorder=10)


def _upd(i):
    t0 = max(0, i - _TRAIL)
    for nm in _pnames:
        bx, by = _xyd[nm][i, 0], _xyd[nm][i, 1]
        _springs[nm].set_data(*_sp2d(0, 0, bx, by))
        _bobs[nm].set_data([bx], [by])
        _trails[nm].set_data(_xyd[nm][t0:i+1, 0], _xyd[nm][t0:i+1, 1])
    _cur.set_data([_ta[i], _ta[i]], ax_tr.get_ylim())
    _ttx.set_text(f"t={_ta[i]:.2f} s")


_anim = FuncAnimation(fig_a, _upd, frames=_nf, interval=40, blit=False)
_ap = os.path.join(RESULTS_DIR, "pinn_spring_pendulum_anim.mp4")
try:
    _anim.save(_ap, writer=FFMpegWriter(fps=25, bitrate=1800))
except Exception:
    _ap = _ap.replace(".mp4", ".gif"); _anim.save(_ap, writer=PillowWriter(fps=20))
print(f"Saved: {_ap}")
plt.show()
