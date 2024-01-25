# %%
_model = model #UNet(n_class=len(labels['AFG']))
weights=torch.load(path2weights)
_model.load_state_dict(weights)
_model.eval()
_model.to(device)
# %%
lognorm_dist = scipy.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu))
lognorm_dist.ppf
# %%
batch = next(iter(train_dl))

xb = batch['obsvariable'].type(torch.float).to(device)
yb = batch['groundtruth'].type(torch.float).to(device)
yb_h = (torch.reshape(torch.sigmoid(_model(xb)), (10,1,8,8)).to(device)-0.5)/0.5

###
fig, axs = plt.subplots(nrows=1,ncols=2,layout='compressed')
f = axs[0].imshow(yb[0,0,:,:].cpu().detach().numpy(),
                  cmap='viridis', vmin=0, vmax=1)
axs[0].set_title('Groundtruth, cdf')
f = axs[1].imshow(yb_h[0,0,:,:].cpu().detach().numpy(),
                  cmap='viridis', vmin=0, vmax=1)
axs[1].set_title('Estimated, cdf')
cbar = fig.colorbar(f, shrink=0.95)

###
max_value = max(lognorm_dist.ppf(yb[0,0,:,:].cpu().detach().numpy()).max(),
                lognorm_dist.ppf(yb_h[0,0,:,:].cpu().detach().numpy()).max())
fig1, axs1 = plt.subplots(nrows=1,ncols=2,layout='compressed')
f1 = axs1[0].imshow(lognorm_dist.ppf(yb[0,0,:,:].cpu().detach().numpy()),
                  cmap='viridis', vmin=0, vmax=max_value)
axs1[0].set_title('Groundtruth, nbldg')
f1 = axs1[1].imshow(lognorm_dist.ppf(yb_h[0,0,:,:].cpu().detach().numpy()),
                  cmap='viridis', vmin=0, vmax=max_value)
axs1[1].set_title('Estimated, nbldg')
axs1 = fig.colorbar(f1, shrink=0.95)