# Physics-Informed & Structure-Preserving Neural Networks

Lecture materials comparing three families of physics-aware neural networks against a plain VanillaNN baseline, across a range of mechanical systems in data-scarce regimes.

## Repository structure

```
pinn_2026/
├── pinn/               Physics-Informed Neural Networks
│   ├── README.md
│   ├── pinn_vs_nn.py              damped mass-spring-damper (MCK) (known params)
│   ├── pinn_identification.py     damped MCK — unknown δ (system ID)
│   ├── pinn_cooling.py            Newton's law of cooling (known R + ID)
│   └── pinn_2dof_spring_damper.py 2-DOF mass-spring-damper
│
├── lnn/                Lagrangian Neural Networks
│   ├── README.md
│   ├── lnn_vs_vanilla.py          simple pendulum, conservative
│   └── lnn_friction.py            simple pendulum, dissipative (LNN-F)
│
├── delan/              Deep Lagrangian Networks
│   ├── README.md
│   ├── delan_vs_vanilla.py        spring pendulum 2-DOF, conservative
│   ├── delan_friction.py          spring pendulum 2-DOF, dissipative (DeLaN-F)
│   └── delan_friction_pendulum.py simple pendulum 1-DOF, dissipative (DeLaN-F)
│
├── models/             Reusable model classes
│   ├── lnn.py                     LNN (arbitrary n_dof)
│   ├── delan.py                   DeLaN (arbitrary n_dof, SPD mass + softplus V)
│   └── vanilla_nn.py              VanillaNN baseline
│
├── systems/            Ground-truth dynamical systems
│   ├── mass_spring_damper.py     damped MCK — analytic solution
│   ├── pendulum.py                simple pendulum — scipy RK45
│   └── spring_pendulum.py         spring pendulum 2-DOF — scipy RK45
│
└── utils/
    └── training.py                train_dynamics_model() for LNN/DeLaN/VanillaNN
```

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

## Running the experiments

```bash
# PINN
python pinn/pinn_vs_nn.py
python pinn/pinn_identification.py
python pinn/pinn_cooling.py
python pinn/pinn_2dof_spring_damper.py

# LNN
python lnn/lnn_vs_vanilla.py
python lnn/lnn_friction.py

# DeLaN
python delan/delan_vs_vanilla.py
python delan/delan_friction.py
python delan/delan_friction_pendulum.py
```

**Dependencies:** `torch`, `numpy`, `matplotlib`, `scipy`
