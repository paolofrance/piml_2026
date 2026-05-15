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

For ODEs with large characteristic frequencies (e.g. the damped mass-spring-damper (MCK) with ω₀ = 20), the raw residual is O(ω₀²) ≈ 400. Without normalisation the physics term dominates the data term by orders of magnitude. The fix:

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

## Examples

### 1. `pinn_cooling.py` — Newton's law of cooling

**System:** `dT/dt = R (T_env − T)`  (T_env=25, T₀=100, R=0.005 s⁻¹)

**Setup:** 10 noisy observations (σ=2 °C) from `t ∈ [0, 300 s]`, evaluation on `[0, 1000 s]`.

**Models:**
- `NN`      — data fit only
- `PINN`    — known R, ODE residual as additional loss
- `PINN-ID` — R treated as a learnable parameter

**Key result:** NN overfits the noisy observations and diverges far from equilibrium. PINN extrapolates correctly to T_env. PINN-ID simultaneously fits the trajectory and identifies R from noisy data — the physics loss provides the identification signal at no extra supervision cost.

*Start here: the 1st-order ODE is the simplest possible setting to introduce the residual loss concept.*

---

### 2. `pinn_vs_nn.py` — Damped mass-spring-damper (MCK)

**System:** `ẍ + 2δẋ + ω₀²x = 0`  (δ=2, ω₀=20, underdamped)

**Setup:** 10 noiseless observations from `t ∈ [0, 0.36]` (≈1 oscillation), evaluation on `[0, 1.0]` (≈3×).

**Models:**
- `PINN` — trajectory model with normalised ODE residual over the full eval domain
- `NN`   — same architecture, data fit only

**Key result:** NN memorises the 10 training points and collapses outside the window. PINN enforces the ODE everywhere and extrapolates the decaying oscillation correctly.

Two important implementation details highlighted here:
- **Residual normalisation** — dividing by ω₀² keeps the physics term O(1) so it does not swamp the data loss.
- **Two-phase training** — warm-up on data only before switching on the physics loss avoids the cold-start problem.

---

## How to run

From the repo root:

```bash
python examples/pinn/pinn_cooling.py
python examples/pinn/pinn_vs_nn.py
```

Dependencies: `torch`, `numpy`, `matplotlib`, `scipy`.

## Outputs

All figures and animations are saved to `examples/pinn/results/`.

| Script | Saved files |
|---|---|
| `pinn_cooling.py` | `pinn_cooling.png` (3-panel comparison), `pinn_cooling_NN.png`, `pinn_cooling_PINN.png`, `pinn_cooling_PINN_ID.png`, `pinn_cooling_anim.mp4` |
| `pinn_vs_nn.py` | `pinn_vs_nn.png` (2-panel comparison), `pinn_vs_nn_PINN.png`, `pinn_vs_nn_NN.png`, `pinn_vs_nn_anim.mp4` |

## Exercises

The following scripts extend these ideas — see `excercises/pinn/`:

| Exercise | Script | Extension |
|---|---|---|
| 1 | `pinn_identification.py` | Unknown system parameters as learnable variables |
| 2 | `pinn_2dof_spring_damper.py` | Multi-DOF coupled ODE system |
| 3 | `pinn_multi_traj.py` | Generalising across initial conditions |
