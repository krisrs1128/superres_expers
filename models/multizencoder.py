#!/usr/bin/env python
import torch
from torch import nn, optim
from torch.nn import functional as F


class MultiZEncoder(nn.Module):
    """
    Simultaneously Encode / Decode Low and High-Resolution Views

    The architecture separately encodes the low and high resolution views,
    though shares a linear layer right before transforming into z. It also
    separately decodes the two resolutions at the very end, though the encoded
    low and high res z's are passed through the same few transformations before
    then.
    """
    def __init__(self, D_low=50, D_high=200, K=10):
        super(MultiZEncoder, self).__init__()

        # Different initial encodings, but shared afterwards
        theta_shared = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
        )

        self.encoders = {
            "high_res": nn.Sequential(
                nn.Linear(D_high, 20),
                theta_shared,

            ),
            "low_res": nn.Sequential(
                nn.Linear(D_low, 20),
                theta_shared
            )
        }

        self.theta_final = nn.Linear(20, K), nn.Linear(20, K) # mean and logvar

        # Shared initial decodings, differs at final linear layer
        self.decoder_initial = nn.Sequential(
            nn.Linear(K, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU()
        )

        # means and variances, after nonlinearities
        self.decoders = {
            "high_res": (nn.Linear(10, D_high), nn.Linear(10, D_high)),
            "low_res": (nn.Linear(10, D_low), nn.Linear(10, D_low))
        }


    def encode(self, x, res_type="high_res"):
        h = self.encoders[res_type](x)
        mu, logvar = self.theta_final # share final layer
        return mu(h), logvar(h)


    def decode(self, z, res_type="high_res"):
        h = self.decoder_initial(z)
        mu, logvar = self.decoders[res_type]
        return mu(h), logvar(h)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def forward(self, x_hr, x_lr):
        # encode x into z-space: phi(x)
        phi_low_x = self.encode(x_lr, "low_res")
        phi_high_x = self.encode(x_hr, "high_res")
        z_low = self.reparameterize(phi_low_x[0], phi_low_x[1])
        z_high = self.reparameterize(phi_high_x[0], phi_high_x[1])

        # decode x from z-space: theta(z)
        theta_low_z = self.decode(z_low, "low_res")
        theta_high_z = self.decode(z_high, "high_res")
        return theta_low_z, phi_low_x, theta_high_z, phi_high_x


def expected_gsn(x, theta_z):
    sq_resid = (x - theta_z[0]) ** 2
    return -0.5 * torch.sum(theta_z[1] + sq_resid / theta_z[1].exp())


def gsn_kl(phi_x):
    return - 0.5 * torch.sum(1 + phi_x[1] - phi_x[0].pow(2) - phi_x[1].exp())


def loss_elem(x, theta_z, phi_x):
    """
    Reconstruction Loss for VAE
    """
    return - expected_gsn(x, theta_z) + gsn_kl(phi_x)


def train_epoch(model, loader, optimizer):
    """
    Train Single Epoch of Model

    :param model: A torch nn Module object
    :param loader: A unwrapped DataLoader instance, each of whose elements
      provides high and a single corresponding low resolution view.
    :param optimizer: A torch.optim object.
    :return A tuple containing
      - model: The model with updated weights.
      - optimizer: The state of the optimzier at the end of the training epoch.
      - train_loss: The average training loss over the epoch
    """
    model.train()
    train_loss = 0
    for i, (x_hr, x_lr) in enumerate(loader):
        optimizer.zero_grad()
        x_hr = x_hr.flatten()
        x_lr = x_lr.flatten()
        theta_low_z, phi_low_x, theta_high_z, phi_high_x = model(x_hr, x_lr)
        loss = loss_elem(x_lr, theta_low_z, phi_low_x) + \
               loss_elem(x_hr, theta_high_z, phi_high_x)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return model, optimizer, train_loss / len(loader)
