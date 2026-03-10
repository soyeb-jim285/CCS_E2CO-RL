"""Utility functions: seeding, metrics, denormalization."""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 1010):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# --------------- Metrics ---------------

def compute_rmse(pred, true):
    pred_np = _to_numpy(pred)
    true_np = _to_numpy(true)
    return np.sqrt(np.mean((pred_np - true_np) ** 2))


def compute_mae(pred, true):
    pred_np = _to_numpy(pred)
    true_np = _to_numpy(true)
    return np.mean(np.abs(pred_np - true_np))


def compute_r2(pred, true):
    """Compute R-squared (coefficient of determination)."""
    pred_np = _to_numpy(pred).flatten()
    true_np = _to_numpy(true).flatten()
    ss_res = np.sum((true_np - pred_np) ** 2)
    ss_tot = np.sum((true_np - np.mean(true_np)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# --------------- Denormalization ---------------

def denormalize_pressure(p_norm, p_min=2200.0, p_max=4069.2):
    return p_norm * (p_max - p_min) + p_min


def denormalize_saturation(s_norm, s_min=0.0, s_max=1.0):
    return s_norm * (s_max - s_min) + s_min


def denormalize_rates(yobs, num_prod=5, num_inj=4,
                      Q_max_w=3151.0, Q_max_g=1.2e6,
                      p_min=2200.0, p_max=4069.2):
    out = yobs.clone() if isinstance(yobs, torch.Tensor) else yobs.copy()
    out[..., :num_prod] = out[..., :num_prod] * Q_max_w
    out[..., num_prod:2 * num_prod] = out[..., num_prod:2 * num_prod] * Q_max_g
    out[..., 2 * num_prod:] = out[..., 2 * num_prod:] * (p_max - p_min) + p_min
    return out
