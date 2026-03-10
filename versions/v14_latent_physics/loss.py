"""V14 Structured ODE loss — data-only (5 terms) + eigenvalue regularization.

Physics is encoded in the dynamics structure (A=-LLᵀ), not in the loss.
Optional regularization penalizes eigenvalues of A that are too negative.
"""

from collections import OrderedDict

import torch
import torch.nn as nn


def _reconstruction_loss(x, x_rec):
    v = 0.1
    diff = x.reshape(x.size(0), -1) - x_rec.reshape(x_rec.size(0), -1)
    return torch.mean(torch.sum(diff * diff * (0.5 / v), dim=-1))


def _l2_reg_loss(z):
    return torch.mean(torch.sum(0.5 * z * z, dim=-1))


class AdaptiveLossWeights(nn.Module):
    def __init__(self, num_terms=6):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(num_terms))

    def forward(self, losses_dict):
        total = torch.zeros(1, device=self.log_sigmas.device)
        for i, (name, loss_val) in enumerate(losses_dict.items()):
            log_s = self.log_sigmas[i]
            precision = torch.exp(-2 * log_s)
            total = total + (0.5 * precision * loss_val + log_s)
        return total.squeeze(0)

    def get_sigmas(self):
        return torch.exp(self.log_sigmas).detach().cpu().numpy()


class StructuredODELoss(nn.Module):
    """Data-only loss + eigenvalue regularization for structured ODE.

    6 loss terms: rec_t0, rec_t1, l2_reg, trans, yobs, eigenvalue_reg
    """

    _lambda_keys = [
        'rec_t0', 'rec_t1', 'l2_reg', 'trans', 'yobs', 'eigenvalue_reg',
    ]

    def __init__(self, cfg, perm_log, device, model=None, lambda_eigen_reg=0.001):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.model = model  # reference to access L matrix
        self.lambda_eigen_reg = lambda_eigen_reg

        # Keep perm for compute_physics_residuals visualization
        perm_linear = torch.pow(10.0, perm_log.to(device))
        self.register_buffer('perm', perm_linear)

        self._phi_ct_over_dt = cfg.porosity * cfg.total_compressibility / cfg.dt_physical
        self._phi_over_dt = cfg.porosity / cfg.dt_physical

        self.use_adaptive = cfg.use_adaptive_weights
        if self.use_adaptive:
            self.adaptive = AdaptiveLossWeights(num_terms=6)
        else:
            self.adaptive = None

        lambda_vals = [
            cfg.lambda_rec_t0, cfg.lambda_rec_t1, cfg.lambda_l2_reg,
            cfg.lambda_trans, cfg.lambda_yobs, lambda_eigen_reg,
        ]
        self.register_buffer('_lambdas', torch.tensor(lambda_vals, dtype=torch.float32))

    def forward(self, pred):
        x0 = pred['x0']
        x0_rec = pred['x0_rec']
        z0 = pred['z0']
        X_next_pred = pred['X_next_pred']
        X_next = pred['X_next']
        Z_next_pred = pred['Z_next_pred']
        Z_next = pred['Z_next']
        Y_next_pred = pred['Y_next_pred']
        Y = pred['Y']

        loss_rec_t0 = _reconstruction_loss(x0, x0_rec)

        loss_rec_t1 = _reconstruction_loss(X_next[0], X_next_pred[0])
        for i in range(1, len(X_next)):
            loss_rec_t1 = loss_rec_t1 + _reconstruction_loss(X_next[i], X_next_pred[i])

        loss_l2_reg = _l2_reg_loss(z0)

        loss_trans = _l2_reg_loss(Z_next[0] - Z_next_pred[0])
        for i in range(1, len(Z_next)):
            loss_trans = loss_trans + _l2_reg_loss(Z_next[i] - Z_next_pred[i])

        loss_yobs = _l2_reg_loss(Y[0] - Y_next_pred[0])
        for i in range(1, len(Y)):
            loss_yobs = loss_yobs + _l2_reg_loss(Y[i] - Y_next_pred[i])

        # Eigenvalue regularization: penalize eigenvalues of A that are too negative
        loss_eigen = torch.tensor(0.0, device=self.device)
        if self.model is not None:
            # Get L matrix from the model
            ode_func = None
            model = self.model
            if hasattr(model, '_orig_mod'):
                model = model._orig_mod
            if hasattr(model, 'transition') and hasattr(model.transition, 'ode_func'):
                ode_func = model.transition.ode_func
            if ode_func is not None and hasattr(ode_func, 'L'):
                L = ode_func.L
                A = -L @ L.T
                eigenvalues = torch.linalg.eigvalsh(A)
                # Penalize eigenvalues more negative than -10
                loss_eigen = torch.mean(torch.relu(-eigenvalues - 10.0) ** 2)

        losses_stack = torch.stack([
            loss_rec_t0, loss_rec_t1, loss_l2_reg, loss_trans,
            loss_yobs, loss_eigen,
        ])

        if self.use_adaptive and self.adaptive is not None:
            losses_dict = OrderedDict(zip(self._lambda_keys, losses_stack))
            total_loss = self.adaptive(losses_dict)
        else:
            total_loss = (self._lambdas * losses_stack).sum()

        return total_loss, losses_stack

    def losses_to_dict(self, losses_stack):
        vals = losses_stack.detach()
        return {name: vals[i].item() for i, name in enumerate(self._lambda_keys)}

    @property
    def num_loss_terms(self):
        return 6

    def get_adaptive_state(self):
        if self.adaptive is not None:
            return self.adaptive.state_dict()
        return None

    def load_adaptive_state(self, state):
        if self.adaptive is not None and state is not None:
            self.adaptive.load_state_dict(state)

    def compute_physics_residuals(self, x_t, x_t1):
        tran_x = 2.0 / (1.0 / self.perm[:, 1:, :] + 1.0 / self.perm[:, :-1, :])
        tran_y = 2.0 / (1.0 / self.perm[:, :, 1:] + 1.0 / self.perm[:, :, :-1])
        p_t = x_t[:, -1:, :, :]
        p_t1 = x_t1[:, -1:, :, :]
        dp_dt = self._phi_ct_over_dt * (p_t1 - p_t)
        flux_x = tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        flux_y = tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        div_x = flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]
        div_y = flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]
        pres_residual = dp_dt[:, :, 1:-1, 1:-1] - (div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :])
        c_t = x_t[:, 0:1, :, :]
        c_t1 = x_t1[:, 0:1, :, :]
        accum = self._phi_over_dt * (c_t1 - c_t)
        vx = -tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        vy = -tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        cx_avg = 0.5 * (c_t1[:, :, 1:, :] + c_t1[:, :, :-1, :])
        cy_avg = 0.5 * (c_t1[:, :, :, 1:] + c_t1[:, :, :, :-1])
        mdiv_x = (cx_avg * vx)[:, :, 1:, :] - (cx_avg * vx)[:, :, :-1, :]
        mdiv_y = (cy_avg * vy)[:, :, :, 1:] - (cy_avg * vy)[:, :, :, :-1]
        mass_residual = accum[:, :, 1:-1, 1:-1] + (mdiv_x[:, :, :, 1:-1] + mdiv_y[:, :, 1:-1, :])
        return pres_residual, mass_residual
