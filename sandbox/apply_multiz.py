#!/usr/bin/env python
import sys
sys.path.append("../")
from models.multiz_unshared import MultiZUnshared, train_epoch
from data.simulate import CurvesUnwrapped
import os
import torch
import torch.optim as optim

# some hyperparameters
hr_size = 20
lr_size = 2
n_steps = 1000
D_block = 20
K = 1000
n_sites = 10000
batch_size = 100
out_dir = "./exper_output/"
eta = 1e-4
save_interval = 100

os.makedirs(out_dir)

# generate data and intentionally try to overfit model
curves = CurvesUnwrapped(n_sites=n_sites, n_views=1, hr_size=hr_size, lr_size=lr_size)
loader = torch.utils.data.DataLoader(curves, batch_size=batch_size)
model = MultiZUnshared(D_high=hr_size * 2, D_low=lr_size * 2, K=K, D_block=D_block)
optimizer = optim.Adam(model.parameters(), lr=eta)

for i in range(n_steps):
    model, optimizer, loss = train_epoch(model, loader, optimizer)
    print("{} || {}".format(i, loss))

    if i % save_interval == 0:
        torch.save(model, "{}/model_{}".format(out_dir, i))
