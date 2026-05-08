# DeLaN Exercises

Two exercises extending `examples/delan/delan_vs_vanilla.py` to dissipative systems.

---

## Exercise 1 — `delan_friction_pendulum.py` — Simple pendulum, dissipative (1-DOF)

**System:** `θ̈ = −(g/L) sin(θ) − b θ̇`  (b=0.3)

**Setup:** 20 data-scarce samples from `t ∈ [0, 3 s]`, θ₀=2.5 rad. Evaluation over `[0, 12 s]`.

**Extension:** add a Rayleigh dissipation matrix `B(q)` to DeLaN. For 1-DOF, `B` reduces to a single positive scalar. The modified equations of motion become:

```
M(q) q̈ = ∂L/∂q − [∂/∂q(∂L/∂q̇)] q̇ − B(q) q̇
```

`B(q) > 0` is enforced via Cholesky (same pattern as M), guaranteeing `dE/dt ≤ 0`.

**Models:**
- `DeLaN-F` — DeLaN + learnable Rayleigh dissipation
- `DeLaN`   — conservative (structurally wrong)
- `VanillaNN` — unstructured baseline

**What to observe:** the energy panel is the decisive diagnostic. Conservative DeLaN keeps energy flat; DeLaN-F follows the monotone decay. Start with this exercise before the 2-DOF version — it uses a familiar system and the 1-DOF Cholesky reduces to a scalar.

---

## Exercise 2 — `delan_friction.py` — Spring pendulum, dissipative (2-DOF)

**System:** spring pendulum with viscous friction on both DOFs:
```
r̈ = r θ̇² − g(1−cosθ) − 2k(r−r₀) − b_r ṙ
θ̈ = (−g sinθ − 2ṙ θ̇) / r − b_θ θ̇
```
(b_r=0.1, b_θ=0.3)

**Setup:** 80 samples from `t ∈ [0, 3 s]`, evaluation over `[0, 8 s]`.

**Extension:** same DeLaN-F structure as Exercise 1, now with a 2×2 Rayleigh matrix `B(q)` (PSD via 2×2 Cholesky).

**Models:**
- `DeLaN-F`   — DeLaN + PSD Rayleigh matrix
- `DeLaN`     — conservative (wrong model)
- `VanillaNN` — unstructured baseline

**What to observe:** compare the trajectory and energy plots for all three models. The 2-DOF case shows that the Cholesky-parameterised `B(q)` correctly captures cross-DOF dissipation coupling.

---

## How to run

```bash
python excercises/delan/delan_friction_pendulum.py
python excercises/delan/delan_friction.py
```
