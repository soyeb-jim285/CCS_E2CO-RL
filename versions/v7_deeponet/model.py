"""V7 DeepONet E2CO — replaces CNN decoder with DeepONet decoder."""

import os
import glob
import torch
import torch.nn as nn

from versions.shared.layers import (
    conv_bn_relu, ResidualConv, dconv_bn_nolinear, fc_bn_relu
)


def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)


def fc_bn_relu_2arg(input_dim, output_dim):
    """Two-argument version matching MSE2C.py:203."""
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU()
    )


def create_trans_encoder(total_input_dim):
    return nn.Sequential(
        fc_bn_relu_2arg(total_input_dim, 200),
        fc_bn_relu_2arg(200, 200),
        fc_bn_relu_2arg(200, total_input_dim - 1)
    )


class Encoder(nn.Module):
    def __init__(self, latent_dim, input_shape, sigma=0.0):
        super().__init__()
        self.sigma = sigma
        self.fc_layers = nn.Sequential(
            conv_bn_relu(input_shape[0], 16, 3, 3, stride=2),
            conv_bn_relu(16, 32, 3, 3, stride=1),
            conv_bn_relu(32, 64, 3, 3, stride=2),
            conv_bn_relu(64, 128, 3, 3, stride=1)
        )
        self.fc_layers.apply(weights_init)

        self.res_layers = nn.Sequential(
            ResidualConv(128, 128, 3, 3, stride=1),
            ResidualConv(128, 128, 3, 3, stride=1),
            ResidualConv(128, 128, 3, 3, stride=1)
        )
        self.res_layers.apply(weights_init)

        self.flatten = nn.Flatten()
        self.fc_mean = nn.Linear(
            128 * int(input_shape[1] / 4) * int(input_shape[2] / 4),
            latent_dim
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = self.res_layers(x)
        x = self.flatten(x)
        xi_mean = self.fc_mean(x)
        return xi_mean


class DeepONetDecoder(nn.Module):
    """DeepONet decoder: branch(z) * trunk(x,y) -> field values.

    branch_net: z -> (n_basis * n_channels) features
    trunk_net: (x, y) -> (n_basis * n_channels) basis functions
    Output: dot product reshaped to (B, n_channels, Nx, Ny)
    """

    def __init__(self, latent_dim, input_shape, n_basis=64, trunk_dim=128):
        super().__init__()
        self.n_basis = n_basis
        self.n_channels = input_shape[0]
        self.Nx = input_shape[1]
        self.Ny = input_shape[2]
        total_basis = n_basis * self.n_channels

        # Branch network: processes latent code
        self.branch_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, total_basis),
        )

        # Trunk network: processes coordinates
        self.trunk_net = nn.Sequential(
            nn.Linear(2, trunk_dim),
            nn.Tanh(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.Tanh(),
            nn.Linear(trunk_dim, total_basis),
        )

        # Bias
        self.bias = nn.Parameter(torch.zeros(self.n_channels))

        # Precompute 64x64 grid coordinates normalized to [-1, 1]
        x_coords = torch.linspace(-1, 1, self.Nx)
        y_coords = torch.linspace(-1, 1, self.Ny)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        grid_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (Nx*Ny, 2)
        self.register_buffer('grid_coords', grid_coords)

    def forward(self, z):
        """z: (B, latent_dim) -> (B, n_channels, Nx, Ny)"""
        B = z.shape[0]

        # Branch: (B, total_basis)
        branch_out = self.branch_net(z)

        # Trunk: (Nx*Ny, total_basis) - recomputed each call (no cache, compatible with torch.compile)
        trunk_out = self.trunk_net(self.grid_coords)

        # Dot product: (B, Nx*Ny, n_channels)
        # Reshape: branch (B, n_channels, n_basis), trunk (Nx*Ny, n_channels, n_basis)
        branch_r = branch_out.view(B, self.n_channels, self.n_basis)
        trunk_r = trunk_out.view(-1, self.n_channels, self.n_basis)

        # Einsum: (B, ch, basis) x (pts, ch, basis) -> (B, pts, ch)
        out = torch.einsum('bcn,pcn->bpc', branch_r, trunk_r)

        # Add bias
        out = out + self.bias

        # Reshape to spatial
        out = out.permute(0, 2, 1).view(B, self.n_channels, self.Nx, self.Ny)
        return out

    def forward_with_grad_coords(self, z):
        """Forward pass with coordinates that allow autograd.
        Returns (output, coords_with_grad) for physics loss computation."""
        B = z.shape[0]
        coords = self.grid_coords.clone().requires_grad_(True)

        branch_out = self.branch_net(z)
        trunk_out = self.trunk_net(coords)  # (Nx*Ny, total_basis)

        branch_r = branch_out.view(B, self.n_channels, self.n_basis)
        trunk_r = trunk_out.view(-1, self.n_channels, self.n_basis)

        out = torch.einsum('bcn,pcn->bpc', branch_r, trunk_r)
        out = out + self.bias
        out = out.permute(0, 2, 1).view(B, self.n_channels, self.Nx, self.Ny)

        return out, coords



class LinearTransitionModel(nn.Module):
    def __init__(self, latent_dim, u_dim, num_prod, num_inj):
        super().__init__()
        self.latent_dim = latent_dim
        self.u_dim = u_dim
        self.num_prod = num_prod
        self.num_inj = num_inj
        self.obs_dim = num_prod * 2 + num_inj

        self.trans_encoder = create_trans_encoder(latent_dim + 1)
        self.trans_encoder.apply(weights_init)

        self.At_layer = nn.Linear(latent_dim, latent_dim * latent_dim)
        self.At_layer.apply(weights_init)
        self.Bt_layer = nn.Linear(latent_dim, latent_dim * u_dim)
        self.Bt_layer.apply(weights_init)
        self.Ct_layer = nn.Linear(latent_dim, self.obs_dim * latent_dim)
        self.Ct_layer.apply(weights_init)
        self.Dt_layer = nn.Linear(latent_dim, self.obs_dim * u_dim)
        self.Dt_layer.apply(weights_init)

    def forward_nsteps(self, zt, dt, U):
        zt_expand = torch.cat([zt, dt], dim=-1)
        hz = self.trans_encoder(zt_expand)

        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.obs_dim, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.obs_dim, self.u_dim)

        Zt_k = []
        Yt_k = []
        for ut in U:
            ut_dt = ut * dt
            zt = (torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1)
                  + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1))
            yt = (torch.bmm(Ct, zt.unsqueeze(-1)).squeeze(-1)
                  + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1))
            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k

    def forward(self, zt, dt, ut):
        zt_expand = torch.cat([zt, dt], dim=-1)
        hz = self.trans_encoder(zt_expand)

        At = self.At_layer(hz).view(-1, self.latent_dim, self.latent_dim)
        Bt = self.Bt_layer(hz).view(-1, self.latent_dim, self.u_dim)
        Ct = self.Ct_layer(hz).view(-1, self.obs_dim, self.latent_dim)
        Dt = self.Dt_layer(hz).view(-1, self.obs_dim, self.u_dim)

        ut_dt = ut * dt
        zt1 = (torch.bmm(At, zt.unsqueeze(-1)).squeeze(-1)
               + torch.bmm(Bt, ut_dt.unsqueeze(-1)).squeeze(-1))
        yt1 = (torch.bmm(Ct, zt1.unsqueeze(-1)).squeeze(-1)
               + torch.bmm(Dt, ut_dt.unsqueeze(-1)).squeeze(-1))
        return zt1, yt1


class DeepONetE2CO(nn.Module):
    """V7 DeepONet E2CO — same encoder and transition as baseline,
    DeepONet-based decoder instead of CNN decoder."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, input_shape,
                 nsteps=2, sigma=0.0, n_basis=64, trunk_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.nsteps = nsteps

        self.encoder = Encoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = DeepONetDecoder(latent_dim, input_shape,
                                       n_basis=n_basis, trunk_dim=trunk_dim)
        self.transition = LinearTransitionModel(latent_dim, u_dim,
                                                num_prod, num_inj)

    def forward(self, inputs):
        """Multi-step forward for training.

        Args:
            inputs: (X, U, Y, dt) where
                X = list of K+1 state tensors [x0, x1, ..., xK]
                U = list of K control tensors
                Y = list of K observation tensors
                dt = time step tensor (batch, 1)

        Returns:
            dict with keys: x0, x0_rec, z0, X_next_pred, X_next,
                           Z_next_pred, Z_next, Y_next_pred, Y
        """
        X, U, Y, dt = inputs

        x0 = X[0]
        z0 = self.encoder(x0)
        x0_rec = self.decoder(z0)

        Z_next_pred, Y_next_pred = self.transition.forward_nsteps(z0, dt, U)

        X_next_pred = []
        Z_next = []
        X_next = X[1:]

        for i_step in range(len(Z_next_pred)):
            z_next_pred = Z_next_pred[i_step]
            x_next_pred = self.decoder(z_next_pred)
            z_next_true = self.encoder(X[i_step + 1])

            X_next_pred.append(x_next_pred)
            Z_next.append(z_next_true)

        return {
            'x0': x0,
            'x0_rec': x0_rec,
            'z0': z0,
            'X_next_pred': X_next_pred,
            'X_next': X_next,
            'Z_next_pred': Z_next_pred,
            'Z_next': Z_next,
            'Y_next_pred': Y_next_pred,
            'Y': Y,
        }

    def predict(self, xt, ut, yt, dt):
        """Single-step prediction for sequential eval."""
        self.eval()
        with torch.no_grad():
            zt = self.encoder(xt)
            zt_next, yt_next = self.transition(zt, dt, ut)
            xt_next_pred = self.decoder(zt_next)
        return xt_next_pred, yt_next

    def predict_latent(self, zt, dt, ut):
        """Latent-space single-step (used by RL)."""
        self.eval()
        with torch.no_grad():
            zt_next, yt_next = self.transition(zt, dt, ut)
        return zt_next, yt_next

    def save_checkpoint(self, path, optimizer=None, epoch=0, best_loss=1e9,
                        adaptive_weights=None):
        state = {
            'model_state': self.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        }
        if optimizer is not None:
            state['optimizer_state'] = optimizer.state_dict()
        if adaptive_weights is not None:
            state['adaptive_weights'] = adaptive_weights
        torch.save(state, path)

    @staticmethod
    def load_checkpoint(path, model, optimizer=None):
        """Load checkpoint, return (epoch, best_loss, adaptive_weights)."""
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        if optimizer is not None and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        adaptive_weights = ckpt.get('adaptive_weights', None)
        return ckpt.get('epoch', 0), ckpt.get('best_loss', 1e9), adaptive_weights

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir):
        """Find the latest ckpt_epoch_NNNN.pt file, return path or None."""
        pattern = os.path.join(checkpoint_dir, "ckpt_epoch_*.pt")
        files = sorted(glob.glob(pattern))
        if not files:
            return None
        return files[-1]
