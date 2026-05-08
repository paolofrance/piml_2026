# PINN Exercises

Three exercises extending `examples/pinn/pinn_vs_nn.py`.

---

## Exercise 1 — `pinn_identification.py` — Parameter identification

**Extension:** the damping coefficient δ is unknown. Make it a learnable `nn.Parameter` and let the ODE residual drive its identification — no extra supervision needed.

**System:** `ẍ + 2δẋ + ω₀²x = 0`  (δ=2 unknown, ω₀=20)

**Key implementation detail:** use log-parameterisation to keep δ strictly positive:
```python
self.log_delta = nn.Parameter(torch.tensor(np.log(delta_init)))

@property
def delta(self):
    return torch.exp(self.log_delta)
```

**What to observe:** δ stays near its initial value during the warm-up phase (data loss only has no gradient signal for it), then converges once the physics loss is switched on. This shows that the ODE residual is the sole identification signal.

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

## How to run

```bash
python excercises/pinn/pinn_identification.py
python excercises/pinn/pinn_2dof_spring_damper.py
python excercises/pinn/pinn_multi_traj.py --n_train 10
```
