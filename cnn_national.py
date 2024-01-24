# %% import packages
from constants import labels

import os
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
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
# %% set device to mps (if working with Mac with MPS)
device = torch.device("mps")
# %% select a country
for icountry in range(len(list(labels.keys()))):
    # icountry = 0
    country = list(labels.keys())[icountry]
    data_path = '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/'
    groundtruth_path = data_path+'groundtruth/METEOR_PROJECT_2002/'+str(country)+'_oed_exposure_20200811/'
    f, ax = plt.subplots(ncols=1, nrows=len(labels[country]), figsize=(7, 5*len(labels[country])))
    lognorm_dist_list = []
    sigma_list = []
    mu_list = []
    for ibldgtype in range(len(labels[country])):
        nbldg_file = groundtruth_path+'attr_rasterized/'+str(country)+'_nbldg_'+str(labels[country][ibldgtype])+'.tif'
        mask_file = groundtruth_path+str(country)+"_country.tif"
        nb = cv2.imread(nbldg_file, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        nb_masked = nb.flatten()[mask.flatten()>0]
        nb_masked = nb_masked[nb_masked != 0]

        x_exp = nb_masked
        mu = np.mean(np.log(nb_masked))
        sigma = np.std(np.log(nb_masked)) 
        mu_exp = np.exp(mu) 
        sigma_exp = np.exp(sigma)
        fitting_params_lognormal = scipy.stats.lognorm.fit(x_exp, floc=0, scale=mu_exp)
        lognorm_dist_fitted = scipy.stats.lognorm(*fitting_params_lognormal)
        t = np.linspace(np.min(x_exp), np.max(x_exp), 100)
        lognorm_dist = scipy.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu))

        lognormal_test = scipy.stats.kstest(x_exp, lognorm_dist.cdf)
        print(lognormal_test)

        lognorm_dist_list.append(lognorm_dist)
        sigma_list.append(sigma)
        mu_list.append(mu)

        sns.distplot(x_exp, ax=ax[ibldgtype], norm_hist=True, kde=False,
                    label='Data')
        ax[ibldgtype].plot(t, lognorm_dist_fitted.pdf(t), lw=2, color='r',
                label='Fitted Model X~LogNorm(mu={0:.1f}, sigma={1:.1f})'.format(lognorm_dist_fitted.mean(), lognorm_dist_fitted.std()))
        ax[ibldgtype].plot(t, lognorm_dist.pdf(t), lw=2, color='g', ls=':',
                label='Original Model X~LogNorm(mu={0:.1f}, sigma={1:.1f})'.format(lognorm_dist.mean(), lognorm_dist.std()))
        ax[ibldgtype].title.set_text(str(labels[country][ibldgtype]))
        ax[ibldgtype].legend(loc='upper right')
    f.savefig('./lognorm/'+str(country)+'.png')
 
# %%
class OpenSendaiBenchDataset(Dataset):
    """
    An implementation of a PyTorch dataset for loading pairs of observable variables and ground truth labels.
    Inspired by https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.
    """
    def __init__(self, sigma: float, mu: float, obsvariables_path: str, groundtruth_path: str, country: str, signals: list, transform: transforms = None):
        """
        Constructs an OpenSendaiBenchDataset.
        :param obsvariables_path: Path to the source folder of observable variables
        :param groundtruth_path: Path to the source folder of corresponding ground truth labels
        :param transform: Callable transformation to apply to images upon loading
        """
        self.obsvariables_path = obsvariables_path
        self.groundtruth_path = groundtruth_path
        self.country = country
        self.signals = signals
        self.transform = transform
        self.lognorm_dist = scipy.stats.lognorm(s=sigma, 
                                                loc=0, scale=np.exp(mu))

    def __len__(self):
        """
        Implements the len(SeaIceDataset) magic method. Required to implement by Dataset superclass.
        When training/testing, this method tells our training loop how much longer we have to go in our Dataset.
        :return: Length of OpenSendaiBenchDataset
        """
        return 100 #len(self.groundtruth_files)/labels[self.country]

    def __getitem__(self, i: int):
        """
        Implements the OpenSendaiBenchDataset[i] magic method. Required to implement by Dataset superclass.
        When training/testing, this method is used to actually fetch data.
        :param i: Index of which image pair to fetch
        :return: Dictionary with pairs of observable variables and ground truth labels.
        """

        obsvariable = np.zeros([len(self.signals),368,368])
        for s in range(len(self.signals)):
            for file in glob.glob(str(os.getcwd()+self.obsvariables_path+
                                    '**/'+self.country+'_*/'+self.country+'_'+
                                    str(i)+'_'+'of_*/2019*_'+self.signals[s]+'.tif')):
                a = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                a = cv2.resize(a, (368,368), interpolation = cv2.INTER_NEAREST)
                obsvariable[s,:,:] = a.reshape(1,a.shape[0],a.shape[1])
                
        groundtruth = np.zeros([len(labels[self.country]),8,8])
        for w in range(len(labels[self.country])): 
            for file in glob.glob(str(os.getcwd()+self.groundtruth_path+
                                      self.country+'*/tiles/images/'+
                                      self.country+'_nbldg_'+labels[self.country][w]+'_'+str(i)+'_'+'of_'+'*.tif')):
                a = cv2.imread(file, cv2.IMREAD_UNCHANGED)
                groundtruth[w,:,:] = self.lognorm_dist.cdf(a.reshape(1,a.shape[0],a.shape[1]))

        obsvariable = torch.from_numpy(obsvariable).float() #.unsqueeze(0)
        # obsvariable_8x8 = torch.from_numpy(obsvariable_8x8).float()
        groundtruth = torch.from_numpy(groundtruth).float() #.unsqueeze(0)
    
        sample = {"obsvariable": obsvariable, "groundtruth": groundtruth}
        if self.transform:
            sample = {"obsvariable": self.transform(obsvariable),
                      "groundtruth": self.transform(groundtruth).squeeze(0).long()}
        return sample

    def visualise(self, i):
        """
        Allows us to visualise a particular SAR/chart pair.
        :param i: Index of which image pair to visualise
        :return: None
        """
        sample = self[i]
        fig1, axs1 = plt.subplots(1,len(self.signals))
        for s in range(len(self.signals)):
            axs1[s].imshow(sample['obsvariable'][s,:,:])
            axs1[s].set_title(str(self.signals[s]))
            axs1[s].set_xticks([])
            axs1[s].set_yticks([])
        plt.tight_layout()
 
        fig2, axs2 = plt.subplots(1,len(labels[self.country]))
        for w in range(len(labels[self.country])): 
            axs2[w].imshow(sample['groundtruth'][w,:,:])
            axs2[w].set_title(labels[self.country][w])
            axs2[w].set_xticks([])
            axs2[w].set_yticks([])
        plt.tight_layout()
# %%
train_ds = OpenSendaiBenchDataset(      obsvariables_path="/obsvariables/", 
                                        groundtruth_path="/groundtruth/", 
                                        country='AFG', 
                                        # signals = ['blue','green','red'])
                                        signals = ['VH','VV','aerosol','blue','green','red','red1','red2','red3','nir','red4','vapor','swir1','swir2'],
                                        sigma=sigma, mu=mu)
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