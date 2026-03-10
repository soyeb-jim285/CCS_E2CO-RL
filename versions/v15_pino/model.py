"""V15 PINO E2CO — FNO decoder + coordinate PINN decoder for dual physics."""

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


class SpectralConv2d(nn.Module):
    """Fourier layer from V6."""

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, x):
        x = x.float()
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNODecoder(nn.Module):
    """FNO-based decoder from V6."""

    def __init__(self, latent_dim, input_shape, modes=12, width=32):
        super().__init__()
        self.input_shape = input_shape
        self.width = width
        self.fc_lift = nn.Linear(latent_dim, width * 16 * 16)
        self.conv0 = SpectralConv2d(width, width, modes, modes)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.conv2 = SpectralConv2d(width, width, modes, modes)
        self.conv3 = SpectralConv2d(width, width, modes, modes)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.fc_proj = nn.Sequential(
            nn.Conv2d(width, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, input_shape[0], 1),
        )

    def forward(self, z):
        x = self.fc_lift(z)
        x = x.view(-1, self.width, 16, 16)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        x1 = self.conv0(x); x2 = self.w0(x); x = x1 + x2; x = F.gelu(x)
        x1 = self.conv1(x); x2 = self.w1(x); x = x1 + x2; x = F.gelu(x)
        x1 = self.conv2(x); x2 = self.w2(x); x = x1 + x2; x = F.gelu(x)
        x1 = self.conv3(x); x2 = self.w3(x); x = x1 + x2
        x = self.fc_proj(x)
        return x


class CoordinatePINNDecoder(nn.Module):
    """MLP decoder from V5 for collocation autograd."""

    def __init__(self, latent_dim, hidden_dim=128, out_channels=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_channels),
        )

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, z, coords):
        z = z.float()
        B, N, _ = coords.shape
        z_expanded = z.unsqueeze(1).expand(-1, N, -1)
        inp = torch.cat([z_expanded, coords], dim=-1)
        inp = inp.reshape(B * N, -1)
        out = self.net(inp)
        return out.reshape(B, N, -1)


class LinearTransitionModel(nn.Module):
    def __init__(self, latent_dim, u_dim, num_prod, num_inj):
        super().__init__()
        self.latent_dim = latent_dim
        self.u_dim = u_dim
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


class PINOE2CO(nn.Module):
    """V15 PINO E2CO — FNO decoder + coordinate PINN decoder."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, input_shape,
                 nsteps=2, sigma=0.0, fno_modes=12, fno_width=32,
                 n_collocation_points=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.nsteps = nsteps
        self.n_collocation_points = n_collocation_points

        self.encoder = Encoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = FNODecoder(latent_dim, input_shape,
                                   modes=fno_modes, width=fno_width)
        self.pinn_decoder = CoordinatePINNDecoder(
            latent_dim, hidden_dim=128, out_channels=input_shape[0])
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
        """FNO decoder only at inference."""
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
        state = {'model_state': self.state_dict(), 'epoch': epoch, 'best_loss': best_loss}
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
