
curves = CurvesUnwrapped(n_sites=1000, n_views=1, hr_size=20, lr_size=10)
loader = torch.utils.data.DataLoader(curves)
model = MultiZEncoder(D_high=40, D_low=20)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for i in range(50):
    model, optimizer, loss = train_epoch(model, loader, optimizer)
    print("i: " + str(i) + "\t" + str(loss))

x_hr, x_lr = curves[0]
_, _, theta_z, _ = model(x_hr.flatten(), x_lr.flatten())

for j in range(10):
    x_rec = model.reparameterize(theta_z[0], theta_z[1])
    x_rec = x_rec.reshape((20, 2))
    plt.scatter(x_rec[:, 0].detach().numpy(), x_rec[:, 1].detach().numpy(), s=0.5)
plt.scatter(x_hr[:, 0].numpy(), x_hr[:, 1].numpy(), s=0.9)
plt.show()
