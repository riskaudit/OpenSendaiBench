# %% import packages
from constants import labels, signals, ntiles
from util import OpenSendaiBenchDataset, fitlognorm

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchsummary import summary

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
device = torch.device("mps")
# %% lognorm fit
lognorm_dist_list = fitlognorm(groundtruth_path=
                               '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/groundtruth/METEOR_PROJECT_2002/')

# %% national or country-level
for icountry in range(len(list(labels.keys()))):
    country = list(labels.keys())[icountry]
    train_ds = OpenSendaiBenchDataset(  obsvariables_path=
                                        '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/obsvariables/METEOR_PROJECT_2002/',
                                        groundtruth_path=
                                        '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/groundtruth/METEOR_PROJECT_2002/',
                                        country=country, 
                                        signal = signals[country],
                                        lognorm_dist = lognorm_dist_list[country])
    print(train_ds[1]['groundtruth'].shape)
    print(train_ds[1]['obsvariable'].shape)
# %%
train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
len(train_dl)
train_dl.dataset[1]['groundtruth'].shape
# %% prototyping
iterator = iter(train_dl)
model = models.resnet50(weights='ResNet50_Weights.DEFAULT').to(device)
model.conv1 = nn.Conv2d(14,64,
                        kernel_size = (7,7),
                        stride = (2,2), 
                        padding = (3,3), bias = False).to(device)
model.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 8, 8)).to(device)
model.fc = nn.Identity().to(device)
summary(model.to(device), 
        input_data=(14, 368, 368), 
        batch_dim = 0, 
        col_names = ('input_size', 'output_size', 'num_params'),
        verbose = 0)
# %%
model.to(device)
model.eval()
model.training = False
for batch_idx in range(0,1):
    data_batch = next(iterator)
    xb = data_batch['obsvariable'].type(torch.float).to(device)
    print(xb.shape)
    yb = data_batch['groundtruth'].type(torch.float).to(device)
    out = (torch.sigmoid(model(xb.to(device)))-0.5)/0.5
    print(out.shape)
    print(out)
    break
# %%

print(model)
# %%
summary(model, input_size=(14, 368, 368))
# %%
loss_func = nn.L1Loss()
iterator = iter(train_dl)

for batch_idx in range(len(train_dl)):
    data_batch = next(iterator)
    xb = data_batch['obsvariable'].type(torch.float).to(device)
    print(xb.shape)
    yb = data_batch['groundtruth'].type(torch.float).to(device)
    out = torch.reshape(torch.sigmoid(model(xb)), (10,1,8,8))
    print(out.shape)
    loss = loss_func(out, yb)
    print(loss)
    print(loss.item())
    break
# %%
loss.backward()
# %%
opt = optim.Adam(model.parameters(), lr=1e-4)
opt.step()
opt.zero_grad()
# %%
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss
rmse = RMSELoss()
def metrics_batch(target, output):
    return rmse(output,target)
def loss_batch(loss_func, xb, yb,yb_h, opt=None):
    loss = loss_func(yb_h, yb)
    metric_b = metrics_batch(yb,yb_h)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), metric_b
def loss_epoch(model,loss_func,dataset_dl,opt=None):
    loss=0.0
    metric=0.0
    iterator = iter(train_dl)
    len_data = len(train_dl.dataset)
    for batch_idx in range(len(train_dl)):
        data_batch = next(iterator)
        xb = data_batch['obsvariable'].type(torch.float).to(device)
        yb = data_batch['groundtruth'].type(torch.float).to(device)
        yb_h = (torch.reshape(torch.sigmoid(model(xb)),
                              (10,1,8,8)).to(device)-0.5)/0.5
        loss_b,metric_b=loss_batch(loss_func, xb, yb,yb_h, opt)
        loss+=loss_b
        if metric_b is not None:
            metric+=metric_b
    loss/=len_data
    metric/=len_data
    return loss, metric
def train_val(epochs, model, loss_func, opt, train_dl, val_dl):
    for epoch in range(epochs):
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,opt)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl)
        accuracy=val_metric #100*val_metric
        print("epoch: %d, train loss: %.6f, val loss: %.6f, rmse: %.6f" %(epoch, train_loss,val_loss,accuracy))
# %%
model.train()
num_epochs=25
train_val(num_epochs, model, loss_func, opt, train_dl, val_dl=train_dl)
# %%
path2weights=str("./models/weights_"
                 +datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                 +"_epoch_"+str(num_epochs)+".pt")
torch.save(model.state_dict(), path2weights)
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