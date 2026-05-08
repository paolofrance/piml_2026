# Exercises

Student exercises extending the worked examples from the two lectures.
Each script builds directly on an example — read the corresponding example README before starting.

## Lecture 1 — PINN exercises

| # | Script | Builds on | What to implement |
|---|---|---|---|
| 1 | `pinn/pinn_identification.py` | `examples/pinn/pinn_vs_nn.py` | Make system parameter δ a learnable variable driven by the physics loss |
| 2 | `pinn/pinn_2dof_spring_damper.py` | `examples/pinn/pinn_vs_nn.py` | Extend to a 2-DOF coupled ODE with a 2-output network |
| 3 | `pinn/pinn_multi_traj.py` | `examples/pinn/pinn_vs_nn.py` | Train on multiple trajectories; generalise to unseen initial conditions |

## Lecture 2 — Lagrangian network exercises

| # | Script | Builds on | What to implement |
|---|---|---|---|
| 1 | `lnn/lnn_friction.py` | `examples/lnn/lnn_vs_vanilla.py` | Add a Rayleigh dissipation network to LNN; learn friction from data |
| 2 | `delan/delan_friction_pendulum.py` | `examples/delan/delan_vs_vanilla.py` | Apply DeLaN-F to the 1-DOF pendulum; compare with LNN-F |
| 3 | `delan/delan_friction.py` | `examples/delan/delan_vs_vanilla.py` | Apply DeLaN-F to the full 2-DOF spring pendulum |

## Running all exercises

```bash
# PINN
python excercises/pinn/pinn_identification.py
python excercises/pinn/pinn_2dof_spring_damper.py
python excercises/pinn/pinn_multi_traj.py --n_train 10

# Lagrangian
python excercises/lnn/lnn_friction.py
python excercises/delan/delan_friction_pendulum.py
python excercises/delan/delan_friction.py
```
