"""
pinn_cooling.py
===============
PINN for Newton's Law of Cooling — faithful to the notebook approach.

ODE     : dT/dt = R (T_env - T)      ←  cooling_law residual form
Solution: T(t) = T_env + (T0-T_env) exp(-R t)

Parameters: T_env=25, T0=100, R=0.005
Data      : 10 noisy observations from t ∈ [0, 300]  (noise std=2)
Eval      : t ∈ [0, 1000]

Three models (all following the notebook architecture/training choices):
  NN       — data fit only                                (lr=1e-5, 20k epochs)
  PINN     — data + physics residual (R known)            (lr=1e-5, 30k epochs)
  PINN-ID  — data + physics residual (R learnable, r₀=0) (lr=5e-6, 40k epochs)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

SEED   = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {"True": "#2c3e50", "NN": "#e74c3c", "PINN": "#3498db", "PINN-ID": "#9b59b6"}

# ---------------------------------------------------------------------------
# System  (same as notebook)
# ---------------------------------------------------------------------------

T_ENV = 25.0
T0    = 100.0
R     = 0.005

def cooling_law(t):
    return T_ENV + (T0 - T_ENV) * np.exp(-R * t)

np.random.seed(10)
times  = np.linspace(0, 1000, 1000)
temps  = cooling_law(times)
t_data = np.linspace(0, 300, 10)
T_data = cooling_law(t_data) + 2.0 * np.random.randn(10)

# ---------------------------------------------------------------------------
# Helpers  (from notebook)
# ---------------------------------------------------------------------------

def np_to_th(x):
    return torch.from_numpy(x).float().to(DEVICE).reshape(len(x), -1)

def grad(outputs, inputs):
    """dOutputs/dInputs via autograd (notebook helper)."""
    return torch.autograd.grad(
        outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
    )

# ---------------------------------------------------------------------------
# Network  (faithful to notebook: ReLU, 100 units, 4 hidden layers)
# ---------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, input_dim, output_dim, n_units=100,
                 epochs=1000, lr=1e-3, loss2=None, loss2_weight=0.1):
        super().__init__()
        self.epochs       = epochs
        self.loss2        = loss2
        self.loss2_weight = loss2_weight
        self.lr           = lr

        self.layers = nn.Sequential(
            nn.Linear(input_dim,  n_units), nn.ReLU(),
            nn.Linear(n_units,    n_units), nn.ReLU(),
            nn.Linear(n_units,    n_units), nn.ReLU(),
            nn.Linear(n_units,    n_units), nn.ReLU(),
        )
        self.out = nn.Linear(n_units, output_dim)

    def forward(self, x):
        return self.out(self.layers(x))

    def fit(self, t_np, T_np, log_every=5000):
        Xt = np_to_th(t_np)
        yt = np_to_th(T_np)
        opt = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        losses = []
        self.train()
        for ep in range(1, self.epochs + 1):
            opt.zero_grad()
            loss = criterion(self.forward(Xt), yt)
            if self.loss2 is not None:
                loss = loss + self.loss2_weight * self.loss2(self)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if ep % log_every == 0:
                print(f"  ep {ep:6d}/{self.epochs}  loss={losses[-1]:.4e}")
        return losses

    def predict(self, t_np):
        self.eval()
        with torch.no_grad():
            return self.forward(np_to_th(t_np)).cpu().numpy().squeeze()


class NetDiscovery(Net):
    """Same as Net but with a learnable cooling rate r (initialised at 0)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.r = nn.Parameter(torch.tensor([0.0]))   # r₀=0, same as notebook

    def fit(self, t_np, T_np, log_every=5000):
        losses = []
        Xt = np_to_th(t_np)
        yt = np_to_th(T_np)
        opt = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        r_history = []
        self.train()
        for ep in range(1, self.epochs + 1):
            opt.zero_grad()
            loss = criterion(self.forward(Xt), yt)
            if self.loss2 is not None:
                loss = loss + self.loss2_weight * self.loss2(self)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            r_history.append(self.r.item())
            if ep % log_every == 0:
                print(f"  ep {ep:6d}/{self.epochs}  loss={losses[-1]:.4e}"
                      f"  r={self.r.item():.5f}")
        return losses, r_history

# ---------------------------------------------------------------------------
# Physics losses  (from notebook)
# ---------------------------------------------------------------------------

def physics_loss(model):
    ts = (torch.linspace(0, 1000, steps=1000)
          .view(-1, 1).requires_grad_(True).to(DEVICE))
    T_pred = model(ts)
    dT     = grad(T_pred, ts)[0]
    pde    = R * (T_ENV - T_pred) - dT
    return torch.mean(pde**2)


def physics_loss_discovery(model):
    ts = (torch.linspace(0, 1000, steps=1000)
          .view(-1, 1).requires_grad_(True).to(DEVICE))
    T_pred = model(ts)
    dT     = grad(T_pred, ts)[0]
    pde    = model.r * (T_ENV - T_pred) - dT
    return torch.mean(pde**2)

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

print("Training NN (data only) ...")
net = Net(1, 1, loss2=None, epochs=20_000, lr=1e-5).to(DEVICE)
net.fit(t_data, T_data)

print("\nTraining PINN (known R) ...")
pinn = Net(1, 1, loss2=physics_loss, loss2_weight=1, epochs=30_000, lr=1e-5).to(DEVICE)
pinn.fit(t_data, T_data)

print("\nTraining PINN-ID (R learnable, r₀=0) ...")
pinn_id = NetDiscovery(1, 1, loss2=physics_loss_discovery,
                       loss2_weight=1, epochs=40_000, lr=5e-6).to(DEVICE)
_, r_history = pinn_id.fit(t_data, T_data)

R_id = pinn_id.r.item()
print(f"\nIdentified R = {R_id:.5f}  (true = {R:.5f})")
print(f"Relative error: {abs(R_id - R)/R * 100:.2f}%")

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

T_nn   = net.predict(times)
T_pinn = pinn.predict(times)
T_id   = pinn_id.predict(times)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

split = 300.0
in_m  = times <= split
out_m = times >  split

def rmse(pred, mask):
    return float(np.sqrt(np.mean((pred[mask] - temps[mask])**2)))

for name, pred in [("NN", T_nn), ("PINN", T_pinn), ("PINN-ID", T_id)]:
    print(f"{name:<10}  interp={rmse(pred, in_m):.3f}  extrap={rmse(pred, out_m):.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(11, 11))
fig.suptitle(
    f"PINN — Newton's Law of Cooling\n"
    f"T_env={T_ENV} °C,  T₀={T0} °C,  R={R} s⁻¹  |  "
    f"10 noisy obs from [0, 300 s]  |  shaded = extrapolation",
    fontsize=12, fontweight="bold",
)

# --- trajectories ---
ax = axes[0]
ax.plot(times, temps, color=COLORS["True"], lw=1, ls="--",
        label="Ground truth", zorder=5)
ax.plot(times, T_nn,   color=COLORS["NN"],      lw=1.8, label="NN (data only)")
ax.plot(times, T_pinn, color=COLORS["PINN"],    lw=1.8, label=f"PINN (R known={R})")
ax.plot(times, T_id,   color=COLORS["PINN-ID"], lw=1.8,
        label=f"PINN-ID (R_id={R_id:.4f})")
ax.scatter(t_data, T_data, s=60, color=COLORS["True"], zorder=10,
           edgecolors="white", linewidths=0.5, label="10 noisy obs (σ=2 °C)")
ax.axvline(split, color="gray", ls=":", lw=1.4, label="train end")
ax.axvspan(split, times[-1], alpha=0.07, color="gray")
ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.set_title("Trajectory", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# --- absolute error ---
ax = axes[1]
for name, pred, c in [("NN", T_nn, COLORS["NN"]),
                       ("PINN", T_pinn, COLORS["PINN"]),
                       ("PINN-ID", T_id, COLORS["PINN-ID"])]:
    ax.semilogy(times, np.abs(pred - temps) + 1e-3, color=c, lw=1.6, label=name)
ax.axvline(split, color="gray", ls=":", lw=1.4)
ax.axvspan(split, times[-1], alpha=0.07, color="gray")
ax.set_ylabel("|T error| °C  (log)", fontsize=11)
ax.set_xlabel("Time (s)", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# --- R convergence ---
ax = axes[2]
epochs_id = np.arange(1, len(r_history) + 1)
ax.plot(epochs_id, r_history, color=COLORS["PINN-ID"], lw=1.5, label="R identified")
ax.axhline(R,   color=COLORS["True"], lw=1.2, ls="--", label=f"R true = {R}")
ax.axhline(0.0, color="gray",         lw=1.0, ls=":",  label="R init = 0")
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("R (s⁻¹)", fontsize=11)
ax.set_title("Cooling rate identification", fontsize=11, fontweight="bold")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Animation — thermometer bars + growing temperature curves
# ---------------------------------------------------------------------------
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

_s = max(1, len(times) // 200)
_ta   = times[::_s]
_Td   = {"True": temps[::_s], "NN": T_nn[::_s],
         "PINN": T_pinn[::_s], "PINN-ID": T_id[::_s]}
_nf   = len(_ta)
_names = ["True", "NN", "PINN", "PINN-ID"]
_xs   = [0, 1, 2, 3]   # bar x positions

fig_a, (ax_th, ax_tr) = plt.subplots(1, 2, figsize=(13, 5))
fig_a.suptitle("Newton's Law of Cooling — temperature comparison", fontsize=11)

# --- thermometer panel ---
ax_th.set_xlim(-0.6, 3.6); ax_th.set_ylim(T_ENV - 5, T0 + 5)
ax_th.set_xticks(_xs); ax_th.set_xticklabels(_names, fontsize=9)
ax_th.set_ylabel("Temperature (°C)"); ax_th.grid(True, alpha=0.2, axis="y")
ax_th.axhline(T_ENV, color="gray", ls="--", lw=1, label=f"T_env={T_ENV}°C")
ax_th.legend(fontsize=8)
_bars = {}
for xi, nm in zip(_xs, _names):
    bar, = ax_th.plot([xi, xi], [T_ENV, _Td[nm][0]],
                      color=COLORS[nm], lw=18, solid_capstyle="butt")
    _bars[nm] = bar
    ax_th.text(xi, T_ENV - 3, nm, ha="center", va="top",
               color=COLORS[nm], fontsize=8, fontweight="bold")
_ttx = ax_th.text(0.98, 0.98, "", transform=ax_th.transAxes,
                  ha="right", va="top", fontsize=9, color="gray")

# --- trajectory panel: curves grow over time ---
ax_tr.set_xlim(times[0], times[-1]); ax_tr.set_ylim(T_ENV - 3, T0 + 5)
ax_tr.set_xlabel("t (s)"); ax_tr.set_ylabel("T(t) (°C)"); ax_tr.grid(True, alpha=0.3)
ax_tr.axvline(300, color="gray", ls=":", lw=1.2, label="train end")
ax_tr.scatter(t_data, T_data, s=50, color=COLORS["True"], zorder=10,
              edgecolors="white", linewidths=0.5, label="obs")
for nm in _names:
    ax_tr.plot(times, _Td[nm]*(_s if False else 1), color=COLORS[nm],
               lw=1, ls="--", alpha=0.3)
_growL = {}
for nm in _names:
    ln, = ax_tr.plot([], [], color=COLORS[nm], lw=2, label=nm)
    _growL[nm] = ln
ax_tr.legend(fontsize=8)
_cur, = ax_tr.plot([], [], color="k", lw=1.2, zorder=10)

def _upd(i):
    t_i = _ta[i]
    mask = times <= t_i
    for nm in _names:
        _bars[nm].set_ydata([T_ENV, float(np.clip(_Td[nm][i], T_ENV, T0 + 5))])
        _growL[nm].set_data(times[mask], temps[mask] if nm == "True" else
                            (T_nn[mask] if nm == "NN" else
                             (T_pinn[mask] if nm == "PINN" else T_id[mask])))
    _cur.set_data([t_i, t_i], [T_ENV - 3, T0 + 5])
    _ttx.set_text(f"t={t_i:.0f} s")

_anim = FuncAnimation(fig_a, _upd, frames=_nf, interval=40, blit=False)
_ap = os.path.join(RESULTS_DIR, "pinn_cooling_anim.mp4")
try:
    _anim.save(_ap, writer=FFMpegWriter(fps=25, bitrate=1800))
except Exception:
    _ap = _ap.replace(".mp4", ".gif"); _anim.save(_ap, writer=PillowWriter(fps=20))
print(f"Saved: {_ap}")
plt.show()
