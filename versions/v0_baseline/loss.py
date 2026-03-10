"""V0 Baseline loss — 5 data-only losses, no physics, no adaptive weights."""

import torch
import torch.nn as nn


def _reconstruction_loss(x, x_rec):
    v = 0.1
    diff = x.reshape(x.size(0), -1) - x_rec.reshape(x_rec.size(0), -1)
    return torch.mean(torch.sum(diff * diff * (0.5 / v), dim=-1))


def _l2_reg_loss(z):
    return torch.mean(torch.sum(0.5 * z * z, dim=-1))


class BaselineLoss(nn.Module):
    """Data-only loss for V0 baseline.

    5 loss terms:
      1. rec_t0:   reconstruction at t=0
      2. rec_t1:   reconstruction at t=1..K
      3. l2_reg:   0.5 * ||z0||^2
      4. trans:    0.5 * ||z_true - z_pred||^2
      5. yobs:     observation match
    """

    def __init__(self, cfg):
        super().__init__()
        self._lambda_keys = ['rec_t0', 'rec_t1', 'l2_reg', 'trans', 'yobs']

        # Fixed lambdas (all 1.0 for baseline)
        lambda_vals = [
            cfg.lambda_rec_t0, cfg.lambda_rec_t1, cfg.lambda_l2_reg,
            cfg.lambda_trans, cfg.lambda_yobs,
        ]
        self.register_buffer('_lambdas', torch.tensor(lambda_vals, dtype=torch.float32))

        # No adaptive weights for baseline
        self.adaptive = None

    def forward(self, pred):
        """Compute all 5 loss terms. Returns (total_loss, losses_stack).

        losses_stack is a 1D tensor with 5 elements.
        """
        x0 = pred['x0']
        x0_rec = pred['x0_rec']
        z0 = pred['z0']
        X_next_pred = pred['X_next_pred']
        X_next = pred['X_next']
        Z_next_pred = pred['Z_next_pred']
        Z_next = pred['Z_next']
        Y_next_pred = pred['Y_next_pred']
        Y = pred['Y']

        # 1. Reconstruction at t=0
        loss_rec_t0 = _reconstruction_loss(x0, x0_rec)

        # 2. Reconstruction at t=1..K
        loss_rec_t1 = _reconstruction_loss(X_next[0], X_next_pred[0])
        for i in range(1, len(X_next)):
            loss_rec_t1 = loss_rec_t1 + _reconstruction_loss(X_next[i], X_next_pred[i])

        # 3. L2 reg
        loss_l2_reg = _l2_reg_loss(z0)

        # 4. Transition
        loss_trans = _l2_reg_loss(Z_next[0] - Z_next_pred[0])
        for i in range(1, len(Z_next)):
            loss_trans = loss_trans + _l2_reg_loss(Z_next[i] - Z_next_pred[i])

        # 5. Observation
        loss_yobs = _l2_reg_loss(Y[0] - Y_next_pred[0])
        for i in range(1, len(Y)):
            loss_yobs = loss_yobs + _l2_reg_loss(Y[i] - Y_next_pred[i])

        # Stack for fast weighted sum
        losses_stack = torch.stack([
            loss_rec_t0, loss_rec_t1, loss_l2_reg, loss_trans, loss_yobs,
        ])

        total_loss = (self._lambdas * losses_stack).sum()

        return total_loss, losses_stack

    def losses_to_dict(self, losses_stack):
        """Convert losses stack to dict with .item() calls — only for logging."""
        vals = losses_stack.detach()
        return {name: vals[i].item() for i, name in enumerate(self._lambda_keys)}

    @property
    def num_loss_terms(self):
        return 5

    def get_adaptive_state(self):
        return None

    def load_adaptive_state(self, state):
        pass
