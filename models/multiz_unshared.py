#!/usr/bin/env python
import torch
from torch import nn, optim
from torch.nn import functional as F

class Block(nn.Module):
    def __init__(self, D_in, D_out, D=20):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(D_in, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Linear(D, D),
            nn.ReLU()
        )
        self.meanvar = (nn.Linear(D, D_out), nn.Linear(D, D_out))

    def forward(self, x):
        return self.block(x)


class MultiZUnshared(nn.Module):
    """
    Simultaneously Encode / Decode Low and High-Resolution Views

    The architecture separately encodes the low and high resolution views,
    though shares a linear layer right before transforming into z. It also
    separately decodes the two resolutions at the very end, though the encoded
    low and high res z's are passed through the same few transformations before
    then.
    """
    def __init__(self, D_low=50, D_high=200, K=20, D_block=30):
        super(MultiZUnshared, self).__init__()
        self.hr_encode = Block(D_high, K, D_block)
        self.lr_encode = Block(D_low, K, D_block)

        # completely disjoint encoders
        self.encoders = {
            "high_res": self.hr_encode.block,
            "low_res": self.lr_encode.block
        }

        self.theta_final = {
            "high_res": self.hr_encode.meanvar,
            "low_res": self.lr_encode.meanvar
        }


        # completely disjoint decoders
        self.hr_decode = Block(K, D_high)
        self.lr_decode = Block(K, D_low)
        self.decoders = {
            "high_res": self.hr_decode.block,
            "low_res": self.lr_decode.block
        }

        self.phi_final = {
            "high_res": self.hr_decode.meanvar,
            "low_res": self.lr_decode.meanvar
        }


    def encode(self, x, res_type="high_res"):
        h = self.encoders[res_type](x)
        mu, logvar = self.theta_final[res_type] # share final layer
        return mu(h), logvar(h)


    def decode(self, z, res_type="high_res"):
        h = self.decoders[res_type](z)
        mu, logvar = self.phi_final[res_type]
        return mu(h), logvar(h)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


    def forward(self, x_hr, x_lr):
        # encode x into z-space: phi(x)
        phi_high_x = self.encode(x_hr, "high_res")
        phi_low_x = self.encode(x_lr, "low_res")
        z_high = self.reparameterize(phi_high_x[0], phi_high_x[1])
        z_low = self.reparameterize(phi_low_x[0], phi_low_x[1])

        # decode x from z-space: theta(z)
        theta_high_z = self.decode(z_high, "high_res")
        theta_low_z = self.decode(z_low, "low_res")
        return theta_high_z, phi_high_x, theta_low_z, phi_low_x


def expected_gsn(x, theta_z):
    sq_resid = (x - theta_z[0]) ** 2
    return -0.5 * torch.sum(theta_z[1] + sq_resid / theta_z[1].exp())


def gsn_kl(phi_x):
    return - 0.5 * torch.sum(1 + phi_x[1] - phi_x[0].pow(2) - phi_x[1].exp())


def loss_elem(x, theta_z, phi_x, beta=0.1):
    """
    Reconstruction Loss for VAE
    """
    return - expected_gsn(x, theta_z) + beta * gsn_kl(phi_x)


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
        theta_high_z, phi_high_x, theta_low_z, phi_low_x = model(x_hr, x_lr)
        # loss = loss_elem(x_lr, theta_low_z, phi_low_x) + \
        #        loss_elem(x_hr, theta_high_z, phi_high_x)
        loss = loss_elem(x_hr, theta_high_z, phi_high_x)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return model, optimizer, train_loss / len(loader)
