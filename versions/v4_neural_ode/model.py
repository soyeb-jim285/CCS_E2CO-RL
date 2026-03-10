"""V4 Neural ODE E2CO — replaces linear transition with Neural ODE integration."""

import os
import glob
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

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
        xi_mean = self.fc_mean(x)
        return xi_mean


class Decoder(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super().__init__()
        self.input_shape = input_shape
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim,
                      int(input_shape[1] * input_shape[2] / 16 * 128)),
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
        y = self.deconv_layers(x)
        return y


class ODEFunc(nn.Module):
    """MLP defining dz/dt = f(z, u).
    The control u is set externally before integration."""

    def __init__(self, latent_dim, u_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + u_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.u = None  # set before integration

    def forward(self, t, z):
        # z: (B, latent_dim)
        u_expanded = self.u.expand(z.shape[0], -1)
        zu = torch.cat([z, u_expanded], dim=-1)
        return self.net(zu)


class NeuralODETransition(nn.Module):
    def __init__(self, latent_dim, u_dim, num_prod, num_inj, method='dopri5'):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = num_prod * 2 + num_inj
        self.method = method

        self.ode_func = ODEFunc(latent_dim, u_dim)

        # Observation model: y = g(z, u)
        self.obs_net = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.obs_dim),
        )

    def forward_nsteps(self, z0, dt, U):
        """Multi-step forward with ODE integration."""
        Zt_k = []
        Yt_k = []
        zt = z0
        for ut in U:
            self.ode_func.u = ut
            t_span = torch.tensor([0.0, 1.0], device=z0.device, dtype=z0.dtype)
            # odeint returns (num_t, batch, latent_dim), take last
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


class NeuralODEE2CO(nn.Module):
    """V4 Neural ODE E2CO — uses ODE-based transition instead of linear."""

    def __init__(self, latent_dim, u_dim, num_prod, num_inj, input_shape,
                 nsteps=2, sigma=0.0, ode_method='dopri5'):
        super().__init__()
        self.latent_dim = latent_dim
        self.nsteps = nsteps

        self.encoder = Encoder(latent_dim, input_shape, sigma=sigma)
        self.decoder = Decoder(latent_dim, input_shape)
        self.transition = NeuralODETransition(latent_dim, u_dim,
                                               num_prod, num_inj,
                                               method=ode_method)

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
