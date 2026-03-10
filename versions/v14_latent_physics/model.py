"""V14 Structured ODE E2CO — physics encoded in dynamics structure (A=-LLᵀ)."""

import os
import glob
import torch
import torch.nn as nn
try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    raise ImportError("V14 requires torchdiffeq: pip install torchdiffeq")

from versions.shared.layers import (
    conv_bn_relu, ResidualConv, dconv_bn_nolinear, fc_bn_relu
)


def weights_init(m):
    if type(m) in [nn.Conv2d, nn.Linear, nn.ConvTranspose2d]:
        torch.nn.init.orthogonal_(m.weight)


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


class StructuredODEFunc(nn.Module):
    """Structured dynamics: dz/dt = A_diffusion(z) + B_advection(z, u) + residual(z, u).

    A_diffusion = -L @ L^T (guaranteed negative semidefinite — dissipative)
    B_advection = MLP(z, u)
    residual = small MLP for remaining dynamics
    """

    def __init__(self, latent_dim, u_dim, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.u_dim = u_dim

        # L matrix for diffusion: A = -L @ L^T ensures negative semidefinite
        self.L = nn.Parameter(torch.randn(latent_dim, latent_dim) * 0.01)

        # Advection: MLP(z, u) -> dz
        self.advection_net = nn.Sequential(
            nn.Linear(latent_dim + u_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Residual MLP for expressiveness
        self.residual_net = nn.Sequential(
            nn.Linear(latent_dim + u_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Initialize residual to near-zero
        with torch.no_grad():
            self.residual_net[-1].weight.mul_(0.01)
            self.residual_net[-1].bias.mul_(0.01)

        self.u = None  # set before integration

    def forward(self, t, z):
        # Diffusion: A @ z where A = -L @ L^T
        A = -self.L @ self.L.T
        dz_diffusion = z @ A.T  # (B, d) @ (d, d)^T = (B, d)

        # Advection
        u_expanded = self.u.expand(z.shape[0], -1)
        zu = torch.cat([z, u_expanded], dim=-1)
        dz_advection = self.advection_net(zu)

        # Residual
        dz_residual = self.residual_net(zu)

        return dz_diffusion + dz_advection + dz_residual


class StructuredODETransition(nn.Module):
    def __init__(self, latent_dim, u_dim, num_prod, num_inj, method='dopri5'):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = num_prod * 2 + num_inj
        self.method = method

        self.ode_func = StructuredODEFunc(latent_dim, u_dim)

        self.obs_net = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.obs_dim),
        )

    def forward_nsteps(self, z0, dt, U):
        Zt_k = []
        Yt_k = []
        zt = z0
        for ut in U:
            self.ode_func.u = ut
            t_span = torch.tensor([0.0, 1.0], device=z0.device, dtype=z0.dtype)
            zt = odeint(self.ode_func, zt, t_span, method=self.method,
                        rtol=1e-3, atol=1e-3)[-1]
            yt = self.obs_net(torch.cat([zt, ut], dim=-1))
            Zt_k.append(zt)
            Yt_k.append(yt)
        return Zt_k, Yt_k

    def forward(self, zt, dt, ut):
        self.ode_func.u = ut
        t_span = torch.tensor([0.0, 1.0], device=zt.device, dtype=zt.dtype)
        zt1 = odeint(self.ode_func, zt, t_span, method=self.method,
                     rtol=1e-3, atol=1e-3)[-1]
        yt1 = self.obs_net(torch.cat([zt1, ut], dim=-1))
        return zt1, yt1


class StructuredODEE2CO(nn.Module):
    """V14 Structured ODE E2CO — physics in dynamics structure, data-only loss."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, input_shape,
                 nsteps=2, sigma=0.0, ode_method='dopri5'):
        super().__init__()
        self.latent_dim = latent_dim
        self.nsteps = nsteps

        self.encoder = Encoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = Decoder(latent_dim, input_shape)
        self.transition = StructuredODETransition(
            latent_dim, u_dim, num_prod, num_inj, method=ode_method)

    def forward(self, inputs):
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
