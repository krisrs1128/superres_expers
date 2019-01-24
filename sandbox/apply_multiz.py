
hr_size = 10
lr_size = 5
n_steps = 1000
D_block = 200
K = 200

curves = CurvesUnwrapped(n_sites=50, n_views=1, hr_size=hr_size, lr_size=lr_size)
loader = torch.utils.data.DataLoader(curves)
model = MultiZUnshared(D_high=hr_size * 2, D_low=lr_size * 2, K=K, D_block=D_block)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
for i in range(n_steps):
    model, optimizer, loss = train_epoch(model, loader, optimizer)
    print("i: " + str(i) + "\t" + str(loss))


# compare truth with reconstruction means
for i in range(10):
    x_hr, x_lr = curves[i]
    _, _, theta_z, _ = model(x_hr.flatten(), x_lr.flatten())
    plt.scatter(x_hr.flatten(), theta_z[0].detach(), s=2)

# compare truth with sampled reconstructions
for j in range(100):
    x_rec = model.reparameterize(theta_z[0], theta_z[1]).detach()
    plt.scatter(x_hr.flatten(), x_rec, s=0.1)

    # x_rec = x_rec.reshape((20, 2))
    # plt.plot(x_rec[:, 0].detach().numpy(), x_rec[:, 1].detach().numpy())
# plt.plot(x_hr[:, 0].numpy(), x_hr[:, 1].numpy())
# plt.show()


for g in optimizer.param_groups:
    g['lr'] = 1e-5
