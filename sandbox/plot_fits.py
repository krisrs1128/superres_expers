
plt.plot(stack(x_hr).detach().numpy())


#         plt.close()
#         plt.plot(stack(x_hr[0]).detach().numpy())
#         plt.plot(theta_z[0].detach().numpy()[0])
#         plt.pause(0.001)

# plt.show()

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
