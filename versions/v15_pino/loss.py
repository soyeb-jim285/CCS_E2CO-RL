"""V15 PINO loss — spectral + autograd pressure physics + consistency + curriculum."""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLossWeights(nn.Module):
    def __init__(self, num_terms=8):
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


def _reconstruction_loss(x, x_rec):
    v = 0.1
    diff = x.reshape(x.size(0), -1) - x_rec.reshape(x_rec.size(0), -1)
    return torch.mean(torch.sum(diff * diff * (0.5 / v), dim=-1))


def _l2_reg_loss(z):
    return torch.mean(torch.sum(0.5 * z * z, dim=-1))


class PINOLoss(nn.Module):
    """Combined loss with spectral + autograd physics + consistency + curriculum.

    8 loss terms: rec_t0, rec_t1, l2_reg, trans, yobs,
                  pressure_pde_spectral, pressure_pde_autograd, consistency
    """

    _lambda_keys = [
        'rec_t0', 'rec_t1', 'l2_reg', 'trans', 'yobs',
        'pressure_pde_spectral', 'pressure_pde_autograd', 'consistency',
    ]

    def __init__(self, cfg, perm_log, device, lambda_consistency=0.1,
                 curriculum_start=60, curriculum_end=120):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.lambda_consistency = lambda_consistency
        self.curriculum_start = curriculum_start
        self.curriculum_end = curriculum_end
        self._current_epoch = 0

        perm_linear = torch.pow(10.0, perm_log.to(device))
        self.register_buffer('perm', perm_linear)
        self.register_buffer('perm_4d', perm_linear.unsqueeze(0))

        self._phi_ct_over_dt = cfg.porosity * cfg.total_compressibility / cfg.dt_physical
        self._phi_over_dt = cfg.porosity / cfg.dt_physical

        # Wavenumber grids for spectral derivatives
        Nx, Ny = cfg.Nx, cfg.Ny
        kx = torch.fft.fftfreq(Nx, d=1.0).to(device) * 2 * np.pi
        ky = torch.fft.rfftfreq(Ny, d=1.0).to(device) * 2 * np.pi
        kx_grid = kx.unsqueeze(-1).expand(Nx, Ny // 2 + 1)
        ky_grid = ky.unsqueeze(0).expand(Nx, Ny // 2 + 1)
        self.register_buffer('kx_grid', kx_grid)
        self.register_buffer('ky_grid', ky_grid)

        self.use_adaptive = cfg.use_adaptive_weights
        if self.use_adaptive:
            self.adaptive = AdaptiveLossWeights(num_terms=8)
        else:
            self.adaptive = None

        lambda_vals = [
            cfg.lambda_rec_t0, cfg.lambda_rec_t1, cfg.lambda_l2_reg,
            cfg.lambda_trans, cfg.lambda_yobs,
            cfg.lambda_pressure_pde, cfg.lambda_pressure_pde,
            lambda_consistency,
        ]
        self.register_buffer('_lambdas', torch.tensor(lambda_vals, dtype=torch.float32))

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    def _get_physics_scale(self):
        e = self._current_epoch
        if e < self.curriculum_start:
            return 0.0
        elif e < self.curriculum_end:
            return (e - self.curriculum_start) / (self.curriculum_end - self.curriculum_start)
        else:
            return 1.0

    @torch.amp.autocast('cuda', enabled=False)
    def _spectral_laplacian(self, field):
        field = field.float()
        f_hat = torch.fft.rfft2(field)
        lap_hat = -(self.kx_grid ** 2 + self.ky_grid ** 2) * f_hat
        return torch.fft.irfft2(lap_hat, s=(field.shape[-2], field.shape[-1]))

    def _pressure_pde_spectral(self, x_t, x_t1):
        p_t = x_t[:, -1:, :, :]
        p_t1 = x_t1[:, -1:, :, :]
        dp_dt = self._phi_ct_over_dt * (p_t1 - p_t)
        k_lap_p = self.perm * self._spectral_laplacian(p_t1)
        residual = dp_dt - k_lap_p
        return torch.mean(residual * residual)

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
        pinn_preds = pred['pinn_pred']
        collocation_coords = pred['collocation_coords']

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

        # Spectral pressure PDE on FNO output
        x_ref = x0
        loss_pde_spectral = self._pressure_pde_spectral(x_ref, X_next_pred[0])
        for i in range(1, len(X_next_pred)):
            loss_pde_spectral = loss_pde_spectral + self._pressure_pde_spectral(
                X_next[i - 1], X_next_pred[i])

        # Autograd pressure PDE at collocation points
        loss_pde_autograd = torch.tensor(0.0, device=self.device)
        can_autograd = torch.is_grad_enabled() and len(pinn_preds) > 0 and pinn_preds[0].requires_grad
        if can_autograd:
            for i_step in range(len(pinn_preds)):
                pinn_out = pinn_preds[i_step]
                coords = collocation_coords[i_step]
                pde_res = self._autograd_pressure_only(pinn_out, coords)
                loss_pde_autograd = loss_pde_autograd + pde_res

        # Consistency: FNO vs PINN at collocation points
        loss_consistency = torch.tensor(0.0, device=self.device)
        for i_step in range(len(pinn_preds)):
            pinn_out = pinn_preds[i_step]
            coords = collocation_coords[i_step]
            fno_out = X_next_pred[i_step]
            grid = coords.detach().unsqueeze(2)
            fno_sampled = F.grid_sample(
                fno_out, grid, mode='bilinear', padding_mode='border',
                align_corners=True
            ).squeeze(-1).permute(0, 2, 1)
            loss_consistency = loss_consistency + F.mse_loss(pinn_out, fno_sampled.detach())

        # Apply curriculum
        scale = self._get_physics_scale()
        loss_pde_spectral = loss_pde_spectral * scale
        loss_pde_autograd = loss_pde_autograd * scale
        loss_consistency = loss_consistency * scale

        losses_stack = torch.stack([
            loss_rec_t0, loss_rec_t1, loss_l2_reg, loss_trans, loss_yobs,
            loss_pde_spectral, loss_pde_autograd, loss_consistency,
        ])

        if self.use_adaptive and self.adaptive is not None:
            losses_dict = OrderedDict(zip(self._lambda_keys, losses_stack))
            total_loss = self.adaptive(losses_dict)
        else:
            total_loss = (self._lambdas * losses_stack).sum()

        return total_loss, losses_stack

    @torch.amp.autocast('cuda', enabled=False)
    def _autograd_pressure_only(self, pinn_out, coords):
        B, N, _ = pinn_out.shape
        pressure = pinn_out[..., 1]

        dp = torch.autograd.grad(
            pressure.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        dp_dx = dp[..., 0]
        dp_dy = dp[..., 1]

        grid = coords.unsqueeze(2)
        perm_4d = self.perm_4d.expand(B, -1, -1, -1)
        perm_at_pts = F.grid_sample(
            perm_4d, grid, mode='bilinear', padding_mode='border',
            align_corners=True
        ).squeeze(-1).squeeze(1)
        k = perm_at_pts

        d2p_dx2 = torch.autograd.grad(
            dp_dx.sum(), coords, create_graph=True, retain_graph=True
        )[0][..., 0]
        d2p_dy2 = torch.autograd.grad(
            dp_dy.sum(), coords, create_graph=True, retain_graph=True
        )[0][..., 1]

        pde_residual = k * (d2p_dx2 + d2p_dy2)
        return torch.mean(pde_residual ** 2)

    def losses_to_dict(self, losses_stack):
        vals = losses_stack.detach()
        return {name: vals[i].item() for i, name in enumerate(self._lambda_keys)}

    @property
    def num_loss_terms(self):
        return 8

    def get_adaptive_state(self):
        if self.adaptive is not None:
            return self.adaptive.state_dict()
        return None

    def load_adaptive_state(self, state):
        if self.adaptive is not None and state is not None:
            self.adaptive.load_state_dict(state)

    def compute_physics_residuals(self, x_t, x_t1):
        p_t = x_t[:, -1:, :, :]
        p_t1 = x_t1[:, -1:, :, :]
        dp_dt = self._phi_ct_over_dt * (p_t1 - p_t)
        k_lap_p = self.perm * self._spectral_laplacian(p_t1)
        pres_residual = dp_dt - k_lap_p
        c_t = x_t[:, 0:1, :, :]
        c_t1 = x_t1[:, 0:1, :, :]
        accum = self._phi_over_dt * (c_t1 - c_t)
        # Simplified mass residual using spectral laplacian on saturation
        mass_residual = accum
        return pres_residual, mass_residual
