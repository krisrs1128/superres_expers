import torch
from torch import nn, optim
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, D=10, k=4):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(D, 20)
        self.fc21 = nn.Linear(20, k)
        self.fc22 = nn.Linear(20, k)
        self.fc3 = nn.Linear(k, 20)
        self.fc4 = nn.Linear(20, D)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_fn(x_hat, x, mu, logvar):
    BCE = F.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE - KLD

def train_epoch():
    model.train()
    for i, (x, _) in enumerate(loader):
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss = loss_fn(x_hat, x, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        print(train_loss)

model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for i in range(5):
    train_epoch()
