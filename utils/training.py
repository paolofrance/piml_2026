"""
Training utilities for dynamics models and PINNs.

Two training protocols:

1. train_dynamics_model  - for VanillaNN, LNN, DeLaN
   Supervised regression on (theta, theta_dot) -> theta_ddot
   Loss: MSE(predicted_accel, true_accel)

2. train_pinn  - for PINN
   Self-supervised with optional data.
   Loss: w_data * MSE(theta_pred, theta_obs) + w_phys * MSE(residual, 0)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------------
# Dynamics models (VanillaNN, LNN, DeLaN)
# ---------------------------------------------------------------------------

def train_dynamics_model(
    model: nn.Module,
    dataset: dict,
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: str = "cpu",
    verbose: bool = True,
    log_every: int = 200,
) -> dict:
    """
    Train a dynamics model (VanillaNN / LNN / DeLaN) on acceleration data.

    Args:
        model  : nn.Module with predict_acceleration(q, qdot) method
        dataset: dict with keys 'theta'/'q', 'theta_dot'/'qdot', 'theta_ddot'/'qddot'
                 1-DOF arrays are shape (N,); multi-DOF arrays are shape (N, n_dof).
    Returns:
        history dict with 'train_loss' list and 'wall_time'
    """
    model = model.to(device)
    model.train()

    # Accept both 1-DOF naming (theta/theta_dot/theta_ddot) and
    # n-DOF naming (q/qdot/qddot) so the same function works everywhere.
    def _to_2d(arr):
        t = torch.tensor(arr, dtype=torch.float32)
        return t if t.dim() == 2 else t.unsqueeze(-1)

    q          = _to_2d(dataset.get("q",     dataset.get("theta")))
    qdot       = _to_2d(dataset.get("qdot",  dataset.get("theta_dot")))
    qddot_true = _to_2d(dataset.get("qddot", dataset.get("theta_ddot")))

    ds = TensorDataset(q, qdot, qddot_true)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "wall_time": 0.0}
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for q_b, qdot_b, qddot_b in loader:
            q_b = q_b.to(device)
            qdot_b = qdot_b.to(device)
            qddot_b = qddot_b.to(device)

            optimizer.zero_grad()
            qddot_pred = model.predict_acceleration(q_b, qdot_b)
            loss = criterion(qddot_pred, qddot_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            epoch_loss += loss.item() * q_b.shape[0]

        epoch_loss /= len(ds)
        history["train_loss"].append(epoch_loss)
        scheduler.step()

        if verbose and epoch % log_every == 0:
            print(f"  Epoch {epoch:5d}/{n_epochs}  loss={epoch_loss:.6f}")

    history["wall_time"] = time.time() - t0
    return history


# ---------------------------------------------------------------------------
# PINN
# ---------------------------------------------------------------------------

def train_pinn(
    model,
    t_obs: np.ndarray,
    theta_obs: np.ndarray,
    t_span: tuple,
    n_epochs: int = 1000,
    n_colloc: int = 200,
    lr: float = 1e-3,
    w_data: float = 1.0,
    w_phys: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
    log_every: int = 500,
) -> dict:
    """
    Train a PINN on a single trajectory.

    Args:
        model    : PINN instance
        t_obs    : (N,) observed time points
        theta_obs: (N,) observed angles
        t_span   : (t0, t1) domain for collocation points
        n_colloc : number of physics collocation points sampled per epoch
        w_data   : weight for data loss
        w_phys   : weight for physics residual loss
    Returns:
        history dict
    """
    model = model.to(device)
    model.train()

    t_obs_t = torch.tensor(t_obs, dtype=torch.float32).unsqueeze(-1).to(device)
    th_obs_t = torch.tensor(theta_obs, dtype=torch.float32).unsqueeze(-1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Step decay: halve lr at 50% and 80% — gentler than cosine for PINNs,
    # which need a sustained lr to escape the flat physics-loss landscape.
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(n_epochs * 0.5), int(n_epochs * 0.8)], gamma=0.5
    )

    history = {"train_loss": [], "data_loss": [], "phys_loss": [], "wall_time": 0.0}
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        # Sample fresh collocation points every epoch
        t_c = torch.rand(n_colloc, 1, device=device) * (t_span[1] - t_span[0]) + t_span[0]

        optimizer.zero_grad()
        total, ld, lp = model.total_loss(t_obs_t, th_obs_t, t_c, w_data, w_phys)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        history["train_loss"].append(total.item())
        history["data_loss"].append(ld.item())
        history["phys_loss"].append(lp.item())

        if verbose and epoch % log_every == 0:
            print(
                f"  Epoch {epoch:5d}/{n_epochs}  total={total.item():.6f}"
                f"  data={ld.item():.6f}  phys={lp.item():.6f}"
            )

    history["wall_time"] = time.time() - t0
    return history
