#!/usr/bin/env python

# some hyperparameters
hr_size = 20
lr_size = 2
n_steps = 1000
D_block = 20
K = 1000

# generate data and intentionally try to overfit model
curves = CurvesUnwrapped(n_sites=10000, n_views=1, hr_size=hr_size, lr_size=lr_size)
loader = torch.utils.data.DataLoader(curves, batch_size=100)
model = MultiZUnshared(D_high=hr_size * 2, D_low=lr_size * 2, K=K, D_block=D_block)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

x_hr, x_lr = next(iter(loader))
plt.plot(stack(x_hr).detach().numpy())
for i in range(n_steps):
    model, optimizer, loss = train_epoch(model, loader, optimizer)

    if i % 5 == 0:
        print("iteration " + str(i) + "\t |\t" + str(loss))
        theta_z, _, _, _ = model(stack(x_hr), stack(x_lr))
        plt.close()
        plt.plot(stack(x_hr[0]).detach().numpy())
        plt.plot(theta_z[0].detach().numpy()[0])
        plt.pause(0.001)

plt.show()

# compare truth with reconstruction means
liter = iter(loader)
for i in range(10):
    x_hr, x_lr = next(liter)
    theta_z, _, _, _ = model(stack(x_hr), stack(x_lr))
    for j in range(len(x_hr)):
        plt.scatter(stack(x_hr[j]), theta_z[0][j].detach().numpy(), s=0.5, alpha=0.1)

# compare truth with sampled reconstructions
liter = iter(loader)
for i in range(1):
    x_hr, x_lr = next(liter)
    theta_z, _, _, _ = model(stack(x_hr), stack(x_lr))
    for j in range(len(x_hr)):
        x_rec = model.reparameterize(theta_z[0][j], theta_z[1][j]).detach()
        plt.scatter(stack(x_hr[j]), x_rec.detach().numpy(), s=0.5, alpha=0.1)

# decrease learning rate
for g in optimizer.param_groups:
    g['lr'] = 1e-4
