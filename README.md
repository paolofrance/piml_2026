# Physics-Informed & Structure-Preserving Neural Networks

Lecture materials comparing three families of physics-aware neural networks against a plain VanillaNN baseline, across a range of mechanical systems in data-scarce regimes.

## Repository structure

```
pinn_2026/
├── examples/           Worked examples — presented during the lecture
│   ├── pinn/
│   │   ├── README.md
│   │   ├── pinn_cooling.py            Newton's law of cooling (1st-order ODE)
│   │   └── pinn_vs_nn.py              damped mass-spring-damper (2nd-order ODE)
│   ├── lnn/
│   │   ├── README.md
│   │   ├── lnn_vs_vanilla.py          simple pendulum, conservative
│   │   └── lnn_vs_delan.py            spring pendulum 2-DOF: LNN vs DeLaN vs VanillaNN
│   └── delan/
│       ├── README.md
│       └── delan_vs_vanilla.py        spring pendulum 2-DOF, conservative
│
├── excercises/         Student exercises — completed after the lecture
│   ├── README.md
│   ├── pinn/
│   │   ├── pinn_identification.py     PINN parameter identification (raw vs softplus c)
│   │   ├── pinn_2dof_spring_damper.py PINN on a 2-DOF coupled system
│   │   ├── pinn_multi_traj.py         PINN generalising across initial conditions
│   │   └── pinn_spring_pendulum.py    PINN on the 2-DOF spring pendulum (contrast with LNN/DeLaN)
│   ├── lnn/
│   │   └── lnn_friction.py            LNN extended to dissipative systems
│   └── delan/
│       ├── delan_friction_pendulum.py DeLaN-F on the simple pendulum (1-DOF)
│       └── delan_friction.py          DeLaN-F on the spring pendulum (2-DOF)
│
├── models/             Reusable model classes
│   ├── lnn.py                         LNN (arbitrary n_dof)
│   ├── delan.py                       DeLaN (arbitrary n_dof, SPD mass + softplus V)
│   └── vanilla_nn.py                  VanillaNN baseline
│
├── systems/            Ground-truth dynamical systems
│   ├── mass_spring_damper.py          damped MCK — analytic solution
│   ├── pendulum.py                    simple pendulum — scipy RK45
│   └── spring_pendulum.py             spring pendulum 2-DOF — scipy RK45
│
└── utils/
    └── training.py                    train_dynamics_model() for LNN/DeLaN/VanillaNN
```

## Lecture structure

**Lecture 1 — Physics-Informed Neural Networks (PINN)**

| | Script | Topic |
|---|---|---|
| Example 1 | `examples/pinn/pinn_cooling.py` | 1st-order ODE, core concept with minimal setup |
| Example 2 | `examples/pinn/pinn_vs_nn_mck.py` | 2nd-order MCK ODE, extrapolation vs plain NN |
| Exercise 1 | `excercises/pinn/pinn_identification.py` | Identify unknown c; compare raw vs softplus parameterisation |
| Exercise 2 | `excercises/pinn/pinn_2dof_spring_damper.py` | Scale to a multi-DOF system |
| Exercise 3 | `excercises/pinn/pinn_multi_traj.py` | Generalise across initial conditions |
| Exercise 4 | `excercises/pinn/pinn_spring_pendulum.py` | PINN on 2-DOF system — contrast with LNN/DeLaN |

**Lecture 2 — Structure-preserving networks (Lagrangian)**

| | Script | Topic |
|---|---|---|
| Example A | `examples/lnn/lnn_vs_vanilla.py` | LNN: scalar Lagrangian + Euler-Lagrange autograd |
| Example B | `examples/delan/delan_vs_vanilla.py` | DeLaN: structured mass matrix, multi-DOF |
| Example C | `examples/lnn/lnn_vs_delan.py` | LNN vs DeLaN vs VanillaNN — 3-way comparison |
| Exercise 1 | `excercises/lnn/lnn_friction.py` | Extend LNN to non-conservative systems |
| Exercise 2 | `excercises/delan/delan_friction_pendulum.py` | DeLaN-F on a familiar 1-DOF system |
| Exercise 3 | `excercises/delan/delan_friction.py` | DeLaN-F on the full 2-DOF system |

## Methods at a glance

| Method | What is learned | Physics constraint | Handles dissipation |
|---|---|---|---|
| **PINN** | trajectory `t → x(t)` | ODE residual as soft loss | yes (ODE encodes it) |
| **LNN** | scalar Lagrangian `L(q,q̇)` | Euler-Lagrange (hard) | via LNN-F extension |
| **DeLaN** | structured `L = T−V`, SPD M(q) | Euler-Lagrange (hard) | via DeLaN-F extension |
| **VanillaNN** | acceleration `(q,q̇) → q̈` | none | n/a |

## Key design choices

**PINN vs LNN/DeLaN:**
PINNs are trajectory models (input: time, output: state) suited for problems where the ODE form is fully known. They require specifying the ODE explicitly and are sensitive to residual normalisation and collocation domain. LNN/DeLaN are dynamics models (input: state, output: acceleration) that learn the physics from `(q, q̇, q̈)` data; they require no ODE specification but assume a Lagrangian structure exists.

**LNN vs DeLaN:**
LNN learns an unstructured scalar Lagrangian — energy conservation emerges but the mass matrix is not guaranteed SPD. DeLaN enforces `M(q) = L(q)L(q)ᵀ + εI` (SPD by Cholesky) and `V(q) ≥ 0` (softplus), giving stronger physical guarantees at negligible extra cost.

**Conservative vs dissipative extensions:**
Both LNN and DeLaN assume a conservative Lagrangian by default. Adding a Rayleigh dissipation network `B(q)` (PSD via Cholesky) extends them to dissipative systems while preserving the guarantee that `dE/dt ≤ 0`.

## Running the examples

```bash
# Lecture 1 — PINN
python examples/pinn/pinn_cooling.py
python examples/pinn/pinn_vs_nn.py

# Lecture 2 — Lagrangian networks
python examples/lnn/lnn_vs_vanilla.py
python examples/delan/delan_vs_vanilla.py
python examples/lnn/lnn_vs_delan.py
```

Each example saves static figures and an animation to its own `results/` subfolder.

## Running the exercises

```bash
python excercises/pinn/pinn_identification.py
python excercises/pinn/pinn_2dof_spring_damper.py
python excercises/pinn/pinn_multi_traj.py --n_train 10

python excercises/lnn/lnn_friction.py

python excercises/delan/delan_friction_pendulum.py
python excercises/delan/delan_friction.py
```

**Dependencies:** `torch`, `numpy`, `matplotlib`, `scipy`
