# PINN Exercises

Four exercises extending `examples/pinn/pinn_vs_nn_mck.py`.

---

## Exercise 1 — `pinn_identification.py` — Parameter identification

**Extension:** the damping coefficient c is unknown. Two PINN-ID variants are compared against a plain NN:

| Variant | Parameterisation | Positivity guaranteed? |
|---|---|---|
| PINN-ID (raw) | `self.c = nn.Parameter(c_init)` | No — can go negative |
| PINN-ID (softplus) | `self.c = softplus(raw_c)` | Yes — `c ≥ 0` always |

**System:** `m·ẍ + c·ẋ + k·x = 0`  (m=1 kg, k=400 N/m known; c=4 N·s/m unknown, init=1)

**Softplus parameterisation:**
```python
self.raw_c = nn.Parameter(torch.tensor(np.log(np.exp(c_init) - 1.0)))

@property
def c(self):
    return F.softplus(self.raw_c)   # log(1 + exp(raw_c)) ≥ 0
```

Residual normalised by k: `(m·ẍ + c·ẋ + k·x) / k`.

**What to observe:**
- During warm-up (data loss only) c receives no gradient and stays flat — the ODE residual is the sole identification signal.
- The raw variant can drift to negative c if initialised poorly or if the warm-up is too short.
- Softplus keeps c non-negative throughout, giving a smoother convergence curve.

---

## Exercise 2 — `pinn_2dof_spring_damper.py` — Multi-DOF system

**Extension:** the PINN output becomes 2-dimensional `t → (x₁(t), x₂(t))` for a coupled 2-DOF system.

**System:**
```
M ẍ + C ẋ + K x = 0
M = I,  K = [[10,−4],[−4,10]],  C = [[0.6,−0.2],[−0.2,0.2]]
```
Two coupled oscillation modes with beat-like interference.

**Key implementation detail:** compute time derivatives independently for each output DOF:
```python
for i in range(2):
    dxi  = autograd.grad(x[:, i].sum(), t_r, create_graph=True)[0]
    ddxi = autograd.grad(dxi.sum(),     t_r, create_graph=True)[0]
```
Residual: `ẍ + C ẋ + K x`, normalised by the largest eigenvalue of K.

**What to observe:** NN fits the training window but misses the beat pattern in extrapolation. PINN enforces both equations simultaneously and tracks both modes.

---

## Exercise 3 — `pinn_multi_traj.py` — Generalising across initial conditions

**Extension:** the network takes `(t, x₀, ẋ₀)` as input and learns to predict `x(t; x₀, ẋ₀)` across a distribution of initial conditions, not just a single trajectory.

**Key idea:** collocation points are sampled from the training IC set over the full eval domain. The IC loss `x(0; x₀, ẋ₀) = x₀` and `ẋ(0; x₀, ẋ₀) = ẋ₀` is enforced via autograd.

**CLI parameter:** `--n_train N` controls how many training trajectories are used.

```bash
python excercises/pinn/pinn_multi_traj.py --n_train 5
python excercises/pinn/pinn_multi_traj.py --n_train 20 --seed 42
```

**What to observe:** with few training trajectories, the PINN generalises to unseen ICs because it has internalised the ODE structure. The plain NN can only interpolate between trajectories it has seen.

---

---

## Exercise 4 — `pinn_spring_pendulum.py` — 2-DOF spring pendulum

**Extension:** apply a PINN to the same spring pendulum used in the LNN/DeLaN examples — the 2-DOF nonlinear system `(r, θ)`. This lets students directly compare the two approaches on identical data.

**System:** `r̈ = r θ̇² − g(1−cosθ) − 2k(r−r₀)`,  `θ̈ = (−g sinθ − 2ṙ θ̇) / r`  (g=10, k=10, r₀=1)

**Key contrast with LNN/DeLaN:**
- PINN takes *position-only* observations (no velocity or acceleration needed).
- PINN outputs the full trajectory `t → (r(t), θ(t))` from a single initial condition.
- PINN has **no energy conservation guarantee** — unlike LNN/DeLaN, energy drift is expected during extrapolation.

**Two ODE residuals (normalised for O(1) scale):**
```
R₁ = (r̈ − r θ̇² + g(1−cosθ) + 2k(r−r₀)) / (2k)
R₂ = (θ̈ + (g sinθ + 2ṙ θ̇)/r) / (g/r₀)
```

**What to observe:** compare the energy panel of the PINN rollout to the DeLaN rollout on the same system. The PINN trajectory may be qualitatively reasonable within the training window but drifts in energy during extrapolation — illustrating what is lost when the Lagrangian structure is replaced by a soft physics penalty.

---

## How to run

```bash
python excercises/pinn/pinn_identification.py
python excercises/pinn/pinn_2dof_spring_damper.py
python excercises/pinn/pinn_multi_traj.py --n_train 10
python excercises/pinn/pinn_spring_pendulum.py
```
