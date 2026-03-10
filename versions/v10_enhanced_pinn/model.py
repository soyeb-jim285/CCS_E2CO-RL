"""V10 Enhanced PINN E2CO — SIREN decoder with Fourier positional encoding."""

import math
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

from versions.shared.layers import (
    conv_bn_relu, ResidualConv, dconv_bn_nolinear, fc_bn_relu
)


def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)


def fc_bn_relu_2arg(input_dim, output_dim):
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
        return self.fc_mean(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, int(input_shape[1] * input_shape[2] / 16 * 128)),
            nn.ReLU()
        )
        self.fc_layers.apply(weights_init)
        self.upsample_layers = nn.Sequential(
            ResidualConv(128, 128, 3, 3),
            ResidualConv(128, 128, 3, 3),
            ResidualConv(128, 128, 3, 3)
        )
        self.upsample_layers.apply(weights_init)
        self.deconv_layers = nn.Sequential(
            dconv_bn_nolinear(128, 64, 3, 3, stride=(1, 1), padding=1),
            dconv_bn_nolinear(64, 32, 2, 2, stride=(2, 2)),
            dconv_bn_nolinear(32, 16, 3, 3, stride=(1, 1), padding=1),
            dconv_bn_nolinear(16, 16, 2, 2, stride=(2, 2)),
            nn.Conv2d(16, input_shape[0], kernel_size=(3, 3), padding='same')
        )
        self.deconv_layers.apply(weights_init)

    def forward(self, z):
        x = self.fc_layers(z)
        x = x.view(-1, 128,
                    int(self.input_shape[1] / 4),
                    int(self.input_shape[2] / 4))
        x = self.upsample_layers(x)
        return self.deconv_layers(x)


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
        Zt_k, Yt_k = [], []
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


class SIRENLayer(nn.Module):
    """SIREN layer: sin(omega_0 * (Wx + b))."""

    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)

        # SIREN-specific initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                bound = math.sqrt(6 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class EnhancedPINNDecoder(nn.Module):
    """SIREN-based coordinate decoder with Fourier positional encoding.

    Features:
    - Fourier positional encoding of (x,y) coordinates
    - SIREN layers (sin activation with omega_0)
    - Residual connections every 2 layers
    """

    def __init__(self, latent_dim, hidden_dim=256, out_channels=2,
                 omega_0=30.0, fourier_L=4, fourier_sigma=2.0):
        super().__init__()
        self.fourier_L = fourier_L
        self.fourier_sigma = fourier_sigma
        # Fourier encoding adds 2*2*L dims to the 2 raw coordinate dims
        coord_dim = 2 + 2 * 2 * fourier_L  # 2 + 16 = 18 for L=4
        input_dim = latent_dim + coord_dim

        # SIREN layers
        self.layer0 = SIRENLayer(input_dim, hidden_dim, omega_0=omega_0, is_first=True)
        self.layer1 = SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0)
        self.layer2 = SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0)
        self.layer3 = SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0)

        # Residual projections (for skip connections every 2 layers)
        self.skip_proj_0 = nn.Linear(input_dim, hidden_dim)
        self.skip_proj_1 = nn.Linear(hidden_dim, hidden_dim)

        # Output layer (linear, no sin)
        self.output_layer = nn.Linear(hidden_dim, out_channels)
        with torch.no_grad():
            bound = math.sqrt(6 / hidden_dim) / omega_0
            self.output_layer.weight.uniform_(-bound, bound)

    def _fourier_encode(self, coords):
        """Fourier positional encoding: [v, sin(2pi*sigma^0*v), cos(...), ...]."""
        encoded = [coords]
        for l in range(self.fourier_L):
            freq = 2 * math.pi * (self.fourier_sigma ** l)
            encoded.append(torch.sin(freq * coords))
            encoded.append(torch.cos(freq * coords))
        return torch.cat(encoded, dim=-1)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, z, coords):
        """
        z: (B, latent_dim)
        coords: (B, n_points, 2) with requires_grad=True
        Returns: (B, n_points, out_channels)
        """
        z = z.float()
        B, N, _ = coords.shape

        # Fourier encode coordinates
        coords_encoded = self._fourier_encode(coords)  # (B, N, coord_dim)

        z_expanded = z.unsqueeze(1).expand(-1, N, -1)
        inp = torch.cat([z_expanded, coords_encoded], dim=-1)  # (B, N, input_dim)
        inp = inp.reshape(B * N, -1)

        # SIREN layers with residual connections
        skip0 = self.skip_proj_0(inp)
        x = self.layer0(inp)
        x = self.layer1(x)
        x = x + skip0  # residual after 2 layers

        skip1 = self.skip_proj_1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x + skip1  # residual after 2 layers

        out = self.output_layer(x)
        return out.reshape(B, N, -1)


class EnhancedPINNE2CO(nn.Module):
    """V10 Enhanced PINN E2CO — SIREN decoder with Fourier features."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, input_shape,
                 nsteps=2, sigma=0.0, n_collocation_points=1024,
                 siren_omega0=30.0, fourier_L=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.nsteps = nsteps
        self.n_collocation_points = n_collocation_points

        self.encoder = Encoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = Decoder(latent_dim, input_shape)
        self.pinn_decoder = EnhancedPINNDecoder(
            latent_dim, hidden_dim=256, out_channels=input_shape[0],
            omega_0=siren_omega0, fourier_L=fourier_L,
        )
        self.transition = LinearTransitionModel(latent_dim, u_dim,
                                                 num_prod, num_inj)

        Nx, Ny = input_shape[1], input_shape[2]
        xs = torch.linspace(-1, 1, Nx)
        ys = torch.linspace(-1, 1, Ny)
        grid_y, grid_x = torch.meshgrid(xs, ys, indexing='ij')
        grid_coords = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        self.register_buffer('grid_coords', grid_coords)

    def _sample_collocation_coords(self, batch_size, device):
        n_total = self.grid_coords.shape[0]
        indices = torch.randint(0, n_total, (self.n_collocation_points,), device=device)
        coords = self.grid_coords[indices]
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1).clone()
        coords.requires_grad_(True)
        return coords

    def forward(self, inputs):
        X, U, Y, dt = inputs
        x0 = X[0]
        z0 = self.encoder(x0)
        x0_rec = self.decoder(z0)

        Z_next_pred, Y_next_pred = self.transition.forward_nsteps(z0, dt, U)

        X_next_pred = []
        Z_next = []
        X_next = X[1:]
        pinn_preds = []
        collocation_coords_list = []

        for i_step in range(len(Z_next_pred)):
            z_next_pred = Z_next_pred[i_step]
            x_next_pred = self.decoder(z_next_pred)
            z_next_true = self.encoder(X[i_step + 1])

            coords = self._sample_collocation_coords(z_next_pred.shape[0],
                                                      z_next_pred.device)
            pinn_out = self.pinn_decoder(z_next_pred, coords)

            X_next_pred.append(x_next_pred)
            Z_next.append(z_next_true)
            pinn_preds.append(pinn_out)
            collocation_coords_list.append(coords)

        coords_t0 = self._sample_collocation_coords(z0.shape[0], z0.device)
        pinn_pred_t0 = self.pinn_decoder(z0, coords_t0)

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
            'pinn_pred': pinn_preds,
            'pinn_pred_t0': pinn_pred_t0,
            'collocation_coords': collocation_coords_list,
            'collocation_coords_t0': coords_t0,
        }

    def predict(self, xt, ut, yt, dt):
        self.eval()
        with torch.no_grad():
            zt = self.encoder(xt)
            zt_next, yt_next = self.transition(zt, dt, ut)
            xt_next_pred = self.decoder(zt_next)
        return xt_next_pred, yt_next

    def predict_latent(self, zt, dt, ut):
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
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        if optimizer is not None and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        return ckpt.get('epoch', 0), ckpt.get('best_loss', 1e9), ckpt.get('adaptive_weights', None)

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir):
        pattern = os.path.join(checkpoint_dir, "ckpt_epoch_*.pt")
        files = sorted(glob.glob(pattern))
        return files[-1] if files else None
