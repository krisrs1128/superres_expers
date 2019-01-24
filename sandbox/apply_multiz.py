#!/usr/bin/env python

# some hyperparameters
hr_size = 20
lr_size = 2
n_steps = 1000
D_block = 50
K = 200

# generate data and intentionally try to overfit model
curves = CurvesUnwrapped(n_sites=4000, n_views=1, hr_size=hr_size, lr_size=lr_size)
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
for i in range(1):
    x_hr, x_lr = curves[i]
    theta_z, _, _, _ = model(stack(x_hr), stack(x_lr))
    plt.scatter(stack(x_hr), theta_z[0].detach(), s=2)

# compare truth with sampled reconstructions
for j in range(100):
    x_rec = model.reparameterize(theta_z[0], theta_z[1]).detach()
    plt.scatter(x_hr.flatten(), x_rec, s=0.1)

# decrease learning rate
for g in optimizer.param_groups:
    g['lr'] = 1e-4
