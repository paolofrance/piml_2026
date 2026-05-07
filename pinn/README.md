# Physics-Informed Neural Networks (PINNs)

## Method

**Reference:** Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems*, Journal of Computational Physics, 2019.

### Core idea

A PINN is a trajectory network that maps time directly to the system state:

```
f_θ : t ──► x(t)
```

Instead of learning the dynamics `ẋ = f(x)` (as LNN/DeLaN do), it learns the solution `x(t)` directly, and enforces the governing ODE as a soft constraint via collocation points sampled over the evaluation domain.

### Loss function

```
L = w_data · L_data  +  w_phys · L_phys

L_data = (1/N) Σ ‖f_θ(tᵢ) − xᵢ‖²          (fit observations)
L_phys = (1/M) Σ ‖R(t_c; θ)‖²              (satisfy ODE at M collocation points)
```

where `R(t; θ)` is the ODE residual evaluated by differentiating the network output with `torch.autograd`.

### Why collocation points must span the full eval domain

Collocation points must be sampled over the entire evaluation window `[0, T_eval]`, not just the training window. If the ODE is only enforced where data exists, the network has no reason to obey the physics in the extrapolation region.

### Residual normalisation

For ODEs with large characteristic frequencies (e.g. the damped harmonic oscillator with ω₀ = 20), the raw residual is O(ω₀²) ≈ 400. Without normalisation the physics term dominates the data term by orders of magnitude. The fix:

```python
residual = (ẍ + 2δẋ + ω₀²x) / ω₀²   # O(1) regardless of ω₀
```

### Two-phase training

Training in two phases prevents the cold-start problem where physics loss dominates before the network has any useful representation:

| Phase | Epochs | Loss |
|---|---|---|
| Warm-up | first ~15 % | data only |
| Physics | remaining | data + physics |

### Parameter identification

Unknown ODE parameters can be made learnable `nn.Parameter`s. The ODE residual provides a self-supervised gradient signal that drives the parameter towards its true value — no extra supervision needed.

**Positivity constraint:** use log-parameterisation to keep parameters positive:
```python
self.log_delta = nn.Parameter(torch.tensor(np.log(delta_init)))

@property
def delta(self):
    return torch.exp(self.log_delta)  # always > 0
```

---

## Experiments

### 1. `pinn_vs_nn.py` — Damped harmonic oscillator (known parameters)

**System:** `ẍ + 2δẋ + ω₀²x = 0`  (δ=2, ω₀=20, underdamped)

**Setup:** 10 noiseless observations from `t ∈ [0, 0.36]`, evaluation on `[0, 1.0]`. The training window covers roughly one oscillation period; the evaluation window is ~3×.

**Models:**
- `PINN` — trajectory model with normalised ODE residual over `[0, 1.0]`
- `NN`   — same architecture, data fit only

**Key result:** NN memorises the 10 training points and collapses outside the window. PINN enforces the ODE everywhere and extrapolates the decaying oscillation correctly (~300× lower extrap RMSE).

---

### 2. `pinn_identification.py` — Damped harmonic oscillator (unknown δ)

**System:** same ODE, but δ is treated as unknown.

**Setup:** same 10 observations. δ initialised at 0.5 (4× below the true value of 2.0).

**Models:**
- `PINN-ID` — PINN with δ as a learnable parameter (`log_delta` parameterisation)
- `NN`       — data fit only baseline

**Key result:** PINN-ID recovers δ ≈ 2.0 from 10 data points and extrapolates correctly. The convergence plot shows δ staying flat during the warm-up phase and converging once physics is switched on.

---

### 3. `pinn_cooling.py` — Newton's law of cooling

**System:** `dT/dt = R (T_env − T)`  (T_env=25, T₀=100, R=0.005 s⁻¹)

**Setup:** 10 noisy observations (σ=2 °C) from `t ∈ [0, 300 s]`, evaluation on `[0, 1000 s]`. Faithful to the notebook approach: ReLU activations, 100 units, constant low learning rate, no warm-up phase.

**Models:**
- `NN`      — data fit only (lr=1e-5, 20k epochs)
- `PINN`    — known R, physics residual always on (lr=1e-5, 30k epochs)
- `PINN-ID` — R learnable (initialised at 0), identification (lr=5e-6, 40k epochs)

**Key result:** NN overfits the noisy observations and diverges. PINN extrapolates to equilibrium. PINN-ID simultaneously fits the trajectory and identifies R from noisy data.

---

### 4. `pinn_2dof_spring_damper.py` — 2-DOF mass-spring-damper

**System:**
```
M ẍ + C ẋ + K x = 0
M = I,  K = [[10,-4],[-4,10]],  C = [[0.6,-0.2],[-0.2,0.2]]
```
Two coupled modes: ω₁ ≈ 2.45 rad/s, ω₂ ≈ 3.74 rad/s.

**Setup:** 15 observations of `(x₁, x₂)` from `t ∈ [0, 2.5 s]`, evaluation on `[0, 10 s]`.

**Architecture extension:** the network output is 2-dimensional. Time derivatives for each DOF are computed independently via autograd:
```python
for i in range(2):
    dxi  = autograd.grad(x[:, i].sum(), t_r, create_graph=True)[0]
    ddxi = autograd.grad(dxi.sum(),     t_r, create_graph=True)[0]
```
Residual: `xddot @ M.T + xdot @ C.T + x @ K.T`, normalised by max eigenvalue of K.

**Key result:** NN fits the training window but cannot reproduce the beat-like coupled oscillation during extrapolation. PINN enforces both ODE equations simultaneously and tracks both modes correctly.

---

## How to run

All scripts are standalone. From the repo root:

```bash
python pinn/pinn_vs_nn.py
python pinn/pinn_identification.py
python pinn/pinn_cooling.py
python pinn/pinn_2dof_spring_damper.py
```

Dependencies: `torch`, `numpy`, `matplotlib`, `scipy`.
