# Deep Lagrangian Networks (DeLaN)

## Method

**Reference:** Lutter, Ritter & Peters, *Deep Lagrangian Networks: Using Physics as Model Prior for Deep Neural Networks*, ICLR 2019.

### Core idea

DeLaN is a strictly more structured variant of LNN. Rather than learning an arbitrary scalar `L_θ(q, q̇)`, it explicitly decomposes the Lagrangian into physically meaningful components:

```
L(q, q̇) = T(q, q̇) − V(q)

T(q, q̇) = ½ q̇ᵀ M(q) q̇        (kinetic energy)
V(q)    = V_net(q) ≥ 0          (potential energy)
```

This decomposition enforces two additional physical constraints that LNN cannot guarantee:

| Property | LNN | DeLaN |
|---|---|---|
| Energy conservation | ✓ (emergent) | ✓ (emergent) |
| M(q) symmetric positive-definite | ✗ (approximate) | ✓ (by construction) |
| V(q) ≥ 0 | ✗ | ✓ (by construction) |

### SPD mass matrix via Cholesky factorisation

The mass network outputs the entries of a lower-triangular matrix `L(q)`. The mass matrix is then:

```
M(q) = L(q) L(q)ᵀ + ε I
```

`ε I` is a small regularisation that guarantees strict positive-definiteness and invertibility. The diagonal of `L(q)` is exponentiated to keep it positive.

### Non-negative potential via Softplus

The potential network output is passed through `softplus`:

```
V(q) = softplus(V_net(q)) ≥ 0
```

This ensures the total energy `E = T + V ≥ 0`, which is required for physical consistency.

### Equations of motion

The Euler-Lagrange equations are applied to the structured Lagrangian using the same row-by-row autograd approach as LNN:

```
M(q) q̈ = ∂L/∂q − [∂/∂q(∂L/∂q̇)] q̇
```

The linear system is solved with `torch.linalg.solve` (stable since M is SPD by design).

### Extension to dissipative systems: DeLaN-F

A **Rayleigh dissipation matrix** `B(q)` is added as a third learned component. It uses the same Cholesky parameterisation as the mass matrix to guarantee PSD:

```
B(q) = L_B(q) L_B(q)ᵀ + ε_B I     (PSD by construction)
```

Modified equations of motion:

```
M(q) q̈ = ∂L/∂q − [∂/∂q(∂L/∂q̇)] q̇ − B(q) q̇
```

`B(q)` PSD implies `dE/dt = −q̇ᵀ B(q) q̇ ≤ 0`: energy is guaranteed to be non-increasing along any trajectory, which is the correct physics for dissipative systems.

### Training

Same supervised protocol as LNN: minimise MSE on `(q, q̇, q̈)` tuples. Rollout via RK4.

---

## Experiments

### 1. `delan_vs_vanilla.py` — Spring pendulum (2-DOF, conservative)

**System:** spring pendulum in polar coordinates `(r, θ)`:
```
r̈ = r θ̇² − g(1−cosθ) − 2k(r−r₀)
θ̈ = (−g sinθ − 2ṙ θ̇) / r
```
(g=10, k=10, r₀=1)

**Setup:** 80 samples from `t ∈ [0, 3 s]`. Evaluation rollout over `[0, 8 s]` in Cartesian `(x, y)` coordinates.

**Models:**
- `DeLaN`     — structured L=T−V with SPD mass matrix
- `VanillaNN` — unstructured `(q, q̇) → q̈` regression

**Key result:** with 80 training points on a 2-DOF nonlinear system, VanillaNN fails to generalise — its energy drifts by ~54% causing the trajectory to diverge. DeLaN's energy conservation structure keeps the rollout on the correct orbit.

---

### 2. `delan_friction.py` — Spring pendulum (2-DOF, dissipative)

**System:** spring pendulum with viscous friction:
```
r̈ = r θ̇² − g(1−cosθ) − 2k(r−r₀) − b_r ṙ
θ̈ = (−g sinθ − 2ṙ θ̇) / r − b_θ θ̇
```
(b_r=0.1, b_θ=0.3)

**Setup:** 80 samples from `t ∈ [0, 3 s]`, evaluation over `[0, 8 s]`.

**Models:**
- `DeLaN-F`   — DeLaN + learnable Rayleigh dissipation matrix B(q) (PSD via Cholesky)
- `DeLaN`     — standard conservative DeLaN (wrong model for this system)
- `VanillaNN` — unstructured baseline

**Key result:** conservative DeLaN correctly captures Lagrangian structure but assumes energy conservation — it predicts undamped oscillations when the ground truth is decaying. DeLaN-F's B(q) matrix correctly absorbs the dissipation, tracking both the trajectory shape and the monotone energy decrease. The energy panel is the decisive diagnostic.

---

### 3. `delan_friction_pendulum.py` — Simple pendulum (1-DOF, dissipative)

**System:** `θ̈ = −(g/L) sin(θ) − b θ̇`  (b=0.3)

**Setup:** 20 data-scarce samples from `t ∈ [0, 3 s]` with θ₀=2.5 rad. Evaluation over `[0, 12 s]`.

This experiment uses DeLaN (not LNN) for a 1-DOF system, demonstrating that the more structured Cholesky mass matrix and softplus potential are valuable even at low dimensionality. For 1-DOF the Cholesky matrix reduces to a single positive scalar, so the structure adds no computational overhead.

**Models:**
- `DeLaN-F`   — DeLaN + friction (n_dof=1, B is a scalar > 0)
- `DeLaN`     — conservative DeLaN (wrong model)
- `VanillaNN` — unstructured baseline

**Key result:** mirrors the 2-DOF result at lower dimensionality. Conservative DeLaN maintains flat energy while the ground truth decays. DeLaN-F correctly tracks the dissipation. VanillaNN fits training data but diverges in extrapolation.

---

## How to run

```bash
python delan/delan_vs_vanilla.py
python delan/delan_friction.py
python delan/delan_friction_pendulum.py
```

Dependencies: `torch`, `numpy`, `matplotlib`, `scipy`.
The scripts import from `models/`, `systems/`, and `utils/` at the repo root.
