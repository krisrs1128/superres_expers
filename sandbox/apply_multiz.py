
curves = CurvesUnwrapped(n_sites=40, hr_size=20, lr_size=10)
loader = torch.utils.data.DataLoader(curves)
model = MultiZEncoder(D_high=40, D_low=20)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
for i in range(50):
    model, optimizer, loss = train_epoch(model, loader, optimizer)
    print("i: " + str(i) + "\t" + str(loss))

x_hr, x_lr = curves[10]
_, _, theta_z, _ = model(x_hr.flatten(), x_lr.flatten())
x_rec = model.reparameterize(theta_z[0], theta_z[1])
#plt.scatter(x_hr.flatten().detach().numpy(), x_rec.detach().numpy())

x_rec = x_rec.reshape((20, 2))
plt.scatter(x_rec[:, 0].detach().numpy(), x_rec[:, 1].detach().numpy())
plt.scatter(x_hr[:, 0].numpy(), x_hr[:, 1].numpy())
