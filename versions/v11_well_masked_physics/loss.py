"""V11 Well-Masked Physics loss — V1's FD physics with well proximity mask."""

from collections import OrderedDict

import torch
import torch.nn as nn


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


class WellMaskedPhysicsLoss(nn.Module):
    """V1 physics loss with well proximity mask on physics residuals.

    8 loss terms: rec_t0, rec_t1, l2_reg, trans, yobs,
                  pressure_pde, mass_conservation, darcy_flux
    """

    def __init__(self, cfg, perm_log, device, well_mask_radius=3):
        super().__init__()
        self.cfg = cfg
        self.device = device

        perm_linear = torch.pow(10.0, perm_log.to(device))
        self.register_buffer('perm', perm_linear)

        self.register_buffer('tran_x',
            2.0 / (1.0 / self.perm[:, 1:, :] + 1.0 / self.perm[:, :-1, :]))
        self.register_buffer('tran_y',
            2.0 / (1.0 / self.perm[:, :, 1:] + 1.0 / self.perm[:, :, :-1]))
        self.register_buffer('darcy_tran_x',
            1.0 / (1.0 / self.perm[:, 1:, :] + 1.0 / self.perm[:, :-1, :]))
        self.register_buffer('darcy_tran_y',
            1.0 / (1.0 / self.perm[:, :, 1:] + 1.0 / self.perm[:, :, :-1]))

        self._phi_ct_over_dt = cfg.porosity * cfg.total_compressibility / cfg.dt_physical
        self._phi_over_dt = cfg.porosity / cfg.dt_physical

        # Build well mask: 0 near wells, 1 elsewhere
        Nx, Ny = cfg.Nx, cfg.Ny
        mask = torch.ones(1, 1, Nx, Ny, device=device)
        for loc in cfg.prod_loc:
            r, c = loc
            for di in range(-well_mask_radius, well_mask_radius + 1):
                for dj in range(-well_mask_radius, well_mask_radius + 1):
                    ri, cj = r + di, c + dj
                    if 0 <= ri < Nx and 0 <= cj < Ny:
                        dist = (di ** 2 + dj ** 2) ** 0.5
                        if dist <= well_mask_radius:
                            mask[0, 0, ri, cj] = 0.0
        self.register_buffer('well_mask', mask)
        # Interior mask for FD residuals (62x62)
        self.register_buffer('well_mask_interior', mask[:, :, 1:-1, 1:-1])

        self.use_adaptive = cfg.use_adaptive_weights
        if self.use_adaptive:
            self.adaptive = AdaptiveLossWeights(num_terms=8)
        else:
            self.adaptive = None

        self._lambda_keys = [
            'rec_t0', 'rec_t1', 'l2_reg', 'trans',
            'yobs', 'pressure_pde', 'mass_conservation', 'darcy_flux',
        ]
        lambda_vals = [
            cfg.lambda_rec_t0, cfg.lambda_rec_t1, cfg.lambda_l2_reg,
            cfg.lambda_trans, cfg.lambda_yobs, cfg.lambda_pressure_pde,
            cfg.lambda_mass_conservation, cfg.lambda_darcy_flux,
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

        # Physics losses with well mask
        x_ref = x0
        loss_pde = self._pressure_pde_residual(x_ref, X_next_pred[0])
        for i in range(1, len(X_next_pred)):
            loss_pde = loss_pde + self._pressure_pde_residual(X_next[i - 1], X_next_pred[i])

        x_ref = x0
        loss_mass = self._mass_conservation_residual(x_ref, X_next_pred[0])
        for i in range(1, len(X_next_pred)):
            loss_mass = loss_mass + self._mass_conservation_residual(X_next_pred[i - 1], X_next_pred[i])

        loss_darcy = self._darcy_flux_loss(X_next[0], X_next_pred[0])
        for i in range(1, len(X_next)):
            loss_darcy = loss_darcy + self._darcy_flux_loss(X_next[i], X_next_pred[i])

        losses_stack = torch.stack([
            loss_rec_t0, loss_rec_t1, loss_l2_reg, loss_trans,
            loss_yobs, loss_pde, loss_mass, loss_darcy,
        ])

        if self.use_adaptive and self.adaptive is not None:
            losses_dict = OrderedDict(zip(self._lambda_keys, losses_stack))
            total_loss = self.adaptive(losses_dict)
        else:
            total_loss = (self._lambdas * losses_stack).sum()

        return total_loss, losses_stack

    def _pressure_pde_residual(self, x_t, x_t1):
        p_t = x_t[:, -1:, :, :]
        p_t1 = x_t1[:, -1:, :, :]
        dp_dt = self._phi_ct_over_dt * (p_t1 - p_t)
        flux_x = self.tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        div_x = flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]
        flux_y = self.tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        div_y = flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]
        residual = dp_dt[:, :, 1:-1, 1:-1] - (div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :])
        # Apply well mask
        residual = residual * self.well_mask_interior
        return torch.mean(residual * residual)

    def _mass_conservation_residual(self, x_t, x_t1):
        c_t = x_t[:, 0:1, :, :]
        c_t1 = x_t1[:, 0:1, :, :]
        p_t1 = x_t1[:, -1:, :, :]
        accum = self._phi_over_dt * (c_t1 - c_t)
        vx = -self.tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        vy = -self.tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        cx_avg = 0.5 * (c_t1[:, :, 1:, :] + c_t1[:, :, :-1, :])
        cy_avg = 0.5 * (c_t1[:, :, :, 1:] + c_t1[:, :, :, :-1])
        flux_x = cx_avg * vx
        flux_y = cy_avg * vy
        div_x = flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]
        div_y = flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]
        residual = accum[:, :, 1:-1, 1:-1] + (div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :])
        # Apply well mask
        residual = residual * self.well_mask_interior
        return torch.mean(residual * residual)

    def _darcy_flux_loss(self, x_true, x_pred):
        p = x_true[:, -1:, :, :]
        p_pred = x_pred[:, -1:, :, :]
        tx = self.darcy_tran_x
        ty = self.darcy_tran_y
        flux_x = (p[:, :, 1:, :] - p[:, :, :-1, :]) * tx
        flux_y = (p[:, :, :, 1:] - p[:, :, :, :-1]) * ty
        flux_x_pred = (p_pred[:, :, 1:, :] - p_pred[:, :, :-1, :]) * tx
        flux_y_pred = (p_pred[:, :, :, 1:] - p_pred[:, :, :, :-1]) * ty
        # Apply well mask (crop to match flux dimensions)
        mask_x = self.well_mask[:, :, :, :-1] * self.well_mask[:, :, :, 1:]  # conservative
        mask_y = self.well_mask[:, :, :-1, :] * self.well_mask[:, :, 1:, :]
        # Use min of adjacent cells' masks
        diff_x = torch.abs(flux_x - flux_x_pred)
        diff_y = torch.abs(flux_y - flux_y_pred)
        # Mask shape may differ, crop to match
        Nx_f = diff_x.shape[2]
        Ny_f = diff_x.shape[3]
        mask_x_crop = mask_x[:, :, :Nx_f, :Ny_f]
        mask_y_crop = mask_y[:, :, :diff_y.shape[2], :diff_y.shape[3]]
        loss_x = torch.sum((diff_x * mask_x_crop).reshape(p.size(0), -1), dim=-1)
        loss_y = torch.sum((diff_y * mask_y_crop).reshape(p.size(0), -1), dim=-1)
        return torch.mean(loss_x + loss_y)

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
        flux_x = self.tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        flux_y = self.tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        div_x = flux_x[:, :, 1:, :] - flux_x[:, :, :-1, :]
        div_y = flux_y[:, :, :, 1:] - flux_y[:, :, :, :-1]
        pres_residual = dp_dt[:, :, 1:-1, 1:-1] - (div_x[:, :, :, 1:-1] + div_y[:, :, 1:-1, :])
        c_t = x_t[:, 0:1, :, :]
        c_t1 = x_t1[:, 0:1, :, :]
        accum = self._phi_over_dt * (c_t1 - c_t)
        vx = -self.tran_x * (p_t1[:, :, 1:, :] - p_t1[:, :, :-1, :])
        vy = -self.tran_y * (p_t1[:, :, :, 1:] - p_t1[:, :, :, :-1])
        cx_avg = 0.5 * (c_t1[:, :, 1:, :] + c_t1[:, :, :-1, :])
        cy_avg = 0.5 * (c_t1[:, :, :, 1:] + c_t1[:, :, :, :-1])
        mdiv_x = (cx_avg * vx)[:, :, 1:, :] - (cx_avg * vx)[:, :, :-1, :]
        mdiv_y = (cy_avg * vy)[:, :, :, 1:] - (cy_avg * vy)[:, :, :, :-1]
        mass_residual = accum[:, :, 1:-1, 1:-1] + (mdiv_x[:, :, :, 1:-1] + mdiv_y[:, :, 1:-1, :])
        return pres_residual, mass_residual
