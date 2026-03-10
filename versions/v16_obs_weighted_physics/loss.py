"""V16 Observation-Weighted Physics loss — distance-weighted physics near wells."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLossWeights(nn.Module):
    def __init__(self, num_terms=9):
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


class ObsWeightedPhysicsLoss(nn.Module):
    """V5 loss with distance-weighted physics near wells.

    Cells close to producer wells get reduced physics weight,
    letting data losses dominate there.

    9 loss terms: rec_t0, rec_t1, l2_reg, trans, yobs,
                  pressure_pde_autograd, mass_conservation_autograd,
                  darcy_flux_autograd, consistency
    """

    _lambda_keys = [
        'rec_t0', 'rec_t1', 'l2_reg', 'trans', 'yobs',
        'pressure_pde_autograd', 'mass_conservation_autograd',
        'darcy_flux_autograd', 'consistency',
    ]

    def __init__(self, cfg, perm_log, device, lambda_consistency=0.1,
                 well_influence_radius=5):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.lambda_consistency = lambda_consistency

        perm_linear = torch.pow(10.0, perm_log.to(device))
        self.register_buffer('perm', perm_linear)
        self.register_buffer('perm_4d', perm_linear.unsqueeze(0))

        self._phi_ct_over_dt = cfg.porosity * cfg.total_compressibility / cfg.dt_physical
        self._phi_over_dt = cfg.porosity / cfg.dt_physical

        # Build distance-based physics weight map
        Nx, Ny = cfg.Nx, cfg.Ny
        weight_map = torch.ones(1, 1, Nx, Ny, device=device)
        for i in range(Nx):
            for j in range(Ny):
                min_dist = float('inf')
                for loc in cfg.prod_loc:
                    dist = ((i - loc[0]) ** 2 + (j - loc[1]) ** 2) ** 0.5
                    min_dist = min(min_dist, dist)
                weight_map[0, 0, i, j] = min(1.0, min_dist / well_influence_radius)
        self.register_buffer('physics_weight_map', weight_map)

        # Normalized [-1,1] weight map for grid_sample at collocation points
        # physics_weight_map is already (1,1,Nx,Ny), usable with grid_sample

        self.use_adaptive = cfg.use_adaptive_weights
        if self.use_adaptive:
            self.adaptive = AdaptiveLossWeights(num_terms=9)
        else:
            self.adaptive = None

        lambda_vals = [
            cfg.lambda_rec_t0, cfg.lambda_rec_t1, cfg.lambda_l2_reg,
            cfg.lambda_trans, cfg.lambda_yobs,
            cfg.lambda_pressure_pde, cfg.lambda_mass_conservation,
            cfg.lambda_darcy_flux, lambda_consistency,
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

        # Weighted autograd physics
        loss_pde_ag = torch.tensor(0.0, device=self.device)
        loss_mass_ag = torch.tensor(0.0, device=self.device)
        loss_darcy_ag = torch.tensor(0.0, device=self.device)

        can_autograd = torch.is_grad_enabled() and len(pinn_preds) > 0 and pinn_preds[0].requires_grad
        if can_autograd:
            for i_step in range(len(pinn_preds)):
                pinn_out = pinn_preds[i_step]
                coords = collocation_coords[i_step]
                pde_res, mass_res, darcy_res = self._weighted_autograd_physics(
                    pinn_out, coords)
                loss_pde_ag = loss_pde_ag + pde_res
                loss_mass_ag = loss_mass_ag + mass_res
                loss_darcy_ag = loss_darcy_ag + darcy_res

        # Consistency loss
        loss_consistency = torch.tensor(0.0, device=self.device)
        for i_step in range(len(pinn_preds)):
            pinn_out = pinn_preds[i_step]
            coords = collocation_coords[i_step]
            cnn_out = X_next_pred[i_step]
            grid = coords.detach().unsqueeze(2)
            cnn_sampled = F.grid_sample(
                cnn_out, grid, mode='bilinear', padding_mode='border',
                align_corners=True
            ).squeeze(-1).permute(0, 2, 1)
            loss_consistency = loss_consistency + F.mse_loss(pinn_out, cnn_sampled.detach())

        losses_stack = torch.stack([
            loss_rec_t0, loss_rec_t1, loss_l2_reg, loss_trans, loss_yobs,
            loss_pde_ag, loss_mass_ag, loss_darcy_ag, loss_consistency,
        ])

        if self.use_adaptive and self.adaptive is not None:
            losses_dict = OrderedDict(zip(self._lambda_keys, losses_stack))
            total_loss = self.adaptive(losses_dict)
        else:
            total_loss = (self._lambdas * losses_stack).sum()

        return total_loss, losses_stack

    @torch.amp.autocast('cuda', enabled=False)
    def _weighted_autograd_physics(self, pinn_out, coords):
        """Autograd physics with distance-based weighting at collocation points."""
        B, N, _ = pinn_out.shape
        saturation = pinn_out[..., 0]
        pressure = pinn_out[..., 1]

        dp = torch.autograd.grad(
            pressure.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        dp_dx = dp[..., 0]
        dp_dy = dp[..., 1]

        ds = torch.autograd.grad(
            saturation.sum(), coords, create_graph=True, retain_graph=True
        )[0]
        ds_dx = ds[..., 0]
        ds_dy = ds[..., 1]

        grid = coords.unsqueeze(2)
        perm_4d = self.perm_4d.expand(B, -1, -1, -1)
        perm_at_pts = F.grid_sample(
            perm_4d, grid, mode='bilinear', padding_mode='border',
            align_corners=True
        ).squeeze(-1).squeeze(1)
        k = perm_at_pts

        # Get physics weight at collocation points
        weight_map_expanded = self.physics_weight_map.expand(B, -1, -1, -1)
        weights_at_pts = F.grid_sample(
            weight_map_expanded, grid, mode='bilinear', padding_mode='border',
            align_corners=True
        ).squeeze(-1).squeeze(1)  # (B, N)

        dp_dx_sum = dp_dx.sum()
        dp_dy_sum = dp_dy.sum()

        d2p_dx2 = torch.autograd.grad(
            dp_dx_sum, coords, create_graph=True, retain_graph=True
        )[0][..., 0]
        d2p_dy2 = torch.autograd.grad(
            dp_dy_sum, coords, create_graph=True, retain_graph=True
        )[0][..., 1]

        # Weighted pressure PDE
        pde_residual = k * (d2p_dx2 + d2p_dy2)
        loss_pde = torch.mean(weights_at_pts * pde_residual ** 2)

        # Weighted mass conservation
        flux_div = (ds_dx * (-k * dp_dx) + ds_dy * (-k * dp_dy)
                    + saturation * (-k * (d2p_dx2 + d2p_dy2)))
        loss_mass = torch.mean(weights_at_pts * flux_div ** 2)

        # Weighted Darcy flux
        darcy_residual = k * (dp_dx ** 2 + dp_dy ** 2)
        darcy_mean = darcy_residual.mean(dim=-1, keepdim=True)
        loss_darcy = torch.mean(weights_at_pts * (darcy_residual - darcy_mean) ** 2)

        return loss_pde, loss_mass, loss_darcy

    def losses_to_dict(self, losses_stack):
        vals = losses_stack.detach()
        return {name: vals[i].item() for i, name in enumerate(self._lambda_keys)}

    @property
    def num_loss_terms(self):
        return 9

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
