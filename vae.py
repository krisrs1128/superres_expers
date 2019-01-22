import torch
from torch import nn, optim
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, D=10, K=4):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(D, 20)
        self.fc21 = nn.Linear(20, K)
        self.fc22 = nn.Linear(20, K)
        self.fc3 = nn.Linear(K, 10)
        self.fc41 = nn.Linear(10, D)
        self.fc42 = nn.Linear(10, D)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc41(h3), self.fc42(h3)

    def forward(self, x):
        phi_x = self.encode(x)
        z = self.reparameterize(phi_x[0], phi_x[1])
        theta_z = self.decode(z)
        return theta_z, phi_x


def expected_gsn(x, theta_z):
    sq_resid = (x - theta_z[0]) ** 2
    return -0.5 * torch.sum(theta_z[1] + sq_resid / theta_z[1].exp())


def gsn_kl(phi_x):
    return - 0.5 * torch.sum(1 + phi_x[1] - phi_x[0].pow(2) - phi_x[1].exp())


def loss_fun(x, theta_z, phi_x):
    return - expected_gsn(x, theta_z) + gsn_kl(phi_x)


def train_epoch(model, loader, optimizer, loss_fun):
    model.train()
    train_loss = 0
    for i, x in enumerate(loader):
        optimizer.zero_grad()
        theta_z, phi_x = model(x)
        loss = loss_fun(x, theta_z, phi_x)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    return model, optimizer, train_loss
