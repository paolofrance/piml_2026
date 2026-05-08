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

## Example

### `delan_vs_vanilla.py` — Spring pendulum (2-DOF, conservative)

**System:** spring pendulum in polar coordinates `(r, θ)`:
```
r̈ = r θ̇² − g(1−cosθ) − 2k(r−r₀)
θ̈ = (−g sinθ − 2ṙ θ̇) / r
```
(g=10, k=10, r₀=1)

**Setup:** 80 samples from `t ∈ [0, 3 s]`. Evaluation rollout over `[0, 8 s]` in Cartesian `(x, y)` coordinates.

**Models:**
- `DeLaN`     — structured L=T−V with SPD mass matrix (Cholesky) and non-negative potential (softplus)
- `VanillaNN` — unstructured `(q, q̇) → q̈` regression

**Key result:** with 80 training points on a 2-DOF nonlinear system, VanillaNN fails to generalise — its energy drifts by ~54% causing the trajectory to diverge from its orbit entirely. DeLaN's guaranteed energy conservation keeps the rollout on the correct orbit throughout the evaluation window.

*Use after the LNN example: same Lagrangian idea, but now with explicit physical structure in the mass matrix and potential — stronger guarantees, multi-DOF system.*

---

## How to run

From the repo root:

```bash
python examples/delan/delan_vs_vanilla.py
```

Dependencies: `torch`, `numpy`, `matplotlib`, `scipy`.
Scripts import from `models/`, `systems/`, and `utils/` at the repo root.

## Exercises

The following scripts extend this example — see `excercises/delan/`:

| Exercise | Script | Extension |
|---|---|---|
| 1 | `delan_friction_pendulum.py` | DeLaN-F on the simple pendulum (1-DOF) — familiar system, dissipation |
| 2 | `delan_friction.py` | DeLaN-F on the spring pendulum (2-DOF) — full complexity |
