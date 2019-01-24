#!/usr/bin/env python

# some parameters
hr_size = 40
lr_size = 2
n_steps = 10000
D_block = 20
K = 500

# generate data and intentionally try to overfit model
curves = CurvesUnwrapped(n_sites=50, n_views=1, hr_size=hr_size, lr_size=lr_size)
loader = torch.utils.data.DataLoader(curves)
model = MultiZUnshared(D_high=hr_size * 2, D_low=lr_size * 2, K=K, D_block=D_block)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
for i in range(n_steps):
    model, optimizer, loss = train_epoch(model, loader, optimizer)
    print("i: " + str(i) + "\t" + str(loss))


# compare truth with reconstruction means
for i in range(10):
    x_hr, x_lr = curves[i]
    theta_z, _, _, _ = model(x_hr.flatten(), x_lr.flatten())
    plt.scatter(x_hr.flatten(), theta_z[0].detach(), s=2)

# compare truth with sampled reconstructions
for j in range(100):
    x_rec = model.reparameterize(theta_z[0], theta_z[1]).detach()
    plt.scatter(x_hr.flatten(), x_rec, s=0.1)

# decrease learning rate
# for g in optimizer.param_groups:
#     g['lr'] = 1e-5
