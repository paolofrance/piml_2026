# LNN Exercise

One exercise extending `examples/lnn/lnn_vs_vanilla.py`.

---

## Exercise 1 — `lnn_friction.py` — Dissipative system

**Extension:** the simple pendulum now has viscous friction `θ̈ = −(g/L) sin(θ) − b θ̇`. The conservative LNN is structurally wrong for this system — it will maintain constant energy while the ground truth decays. The exercise adds a learnable friction term.

**System:** `θ̈ = −(g/L) sin(θ) − b θ̇`  (b=0.3)

**Approach (Ragusano et al., Politecnico di Milano 2024):** rather than constraining the friction network to be positive (Rayleigh), train an unconstrained DNN `F̂(θ, θ̇)` jointly with the LNN using two losses:

```
l_dyn  = MSE(q̈_LNN + F̂/M,  q̈_true)           # full dynamics
l_fric = MSE(F̂,  M · (q̈_true − q̈_LNN))        # friction residual
```

The second loss trains the friction network against the *implied* residual that the conservative LNN cannot explain. Gradients from `l_fric` flow only to the friction network (LNN outputs are detached).

**What to observe:**
- The conservative LNN rolls out with flat energy (physically incorrect).
- LNN-F tracks the monotone energy decay.
- The fifth plot panel shows the learned friction profile `F̂(θ, θ̇)` vs the ground truth `−b θ̇`.

**Training tip:** the joint training creates a moving-target problem for the friction network. A warm-up phase (LNN only) followed by joint fine-tuning stabilises convergence.

---

## How to run

```bash
python excercises/lnn/lnn_friction.py
```
