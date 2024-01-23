# %%
from constants import labels

import os
import cv2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
import rioxarray as rxr
import copy
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
import scipy
from scipy import stats
from PIL import Image 

import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

import torch.nn.functional as nnf
from torch.nn.functional import relu
import torch.nn.functional as F

from torchsummary import summary

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# %%
device = torch.device("mps")
# %%
country = 'AFG'
file_path = '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/groundtruth/METEOR_PROJECT_2002/'+str(country)+'_oed_exposure_20200811/'
nbldg_file = file_path+'attr_rasterized/'+str(country)+'_nbldg_'+str(labels[country][0])+'.tif'
mask_file = file_path+str(country)+"_country.tif"
nb = cv2.imread(nbldg_file, cv2.IMREAD_UNCHANGED)
mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
nb_masked = nb.flatten()[mask.flatten()>0]
nb_masked = nb_masked[nb_masked != 0]
nb_masked.shape
sns.histplot(data=nb_masked, stat='percent')
# %%
x_exp = nb_masked
mu = np.mean(np.log(nb_masked)) #mle estimate - mean of the log of distribution
sigma = np.std(np.log(nb_masked)) # mle estimate - stddev of the log of distrib
mu_exp = np.exp(mu) 
sigma_exp = np.exp(sigma)

fitting_params_lognormal = scipy.stats.lognorm.fit(x_exp, 
                                                   floc=0, #floc=0 points the optimizer to a pretty good place to start, but then tells it to stay there.
                                                   scale=mu_exp)
lognorm_dist_fitted = scipy.stats.lognorm(*fitting_params_lognormal)
t = np.linspace(np.min(x_exp), np.max(x_exp), 100)

lognorm_dist = scipy.stats.lognorm(s=sigma, loc=0, scale=np.exp(mu))

# Plot lognormals
f, ax = plt.subplots(1, sharex='col', figsize=(10, 5))
sns.distplot(x_exp, ax=ax, norm_hist=True, kde=False,
             label='Data exp(X)~N(mu={0:.1f}, sigma={1:.1f})\n X~LogNorm(mu={0:.1f}, sigma={1:.1f})'.format(mu, sigma))
ax.plot(t, lognorm_dist_fitted.pdf(t), lw=2, color='r',
        label='Fitted Model X~LogNorm(mu={0:.1f}, sigma={1:.1f})'.format(lognorm_dist_fitted.mean(), lognorm_dist_fitted.std()))
ax.plot(t, lognorm_dist.pdf(t), lw=2, color='g', ls=':',
        label='Original Model X~LogNorm(mu={0:.1f}, sigma={1:.1f})'.format(lognorm_dist.mean(), lognorm_dist.std()))
# ax.plot(t, lognorm_dist.cdf(t), lw=2, color='b',label='CDF')
ax.title.set_text(str(str(country)+' - lognormCDF or non-exceedance'))
ax.legend(loc='lower right')
plt.show()
f.savefig('./lognorm/'+str(country)+'.png')
plt.close()
 
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
#         return out
# %% prototyping

## encoder
# k = 13
# e11 = nn.Conv2d(14, 12, kernel_size=k, padding=1).to(device) 
# xe11 = relu(e11(xb))
# print(xe11.shape)

# e12 = nn.Conv2d(12, 10, kernel_size=k, padding=1).to(device)
# xe12 = relu(e12(xe11))
# print(xe12.shape)

# pool1 = nn.MaxPool2d(kernel_size=3, stride=2).to(device) 
# xp1 = pool1(xe12)
# print(xp1.shape)

# e21 = nn.Conv2d(10, 8, kernel_size=k, padding=1).to(device) 
# xe21 = relu(e21(xp1))
# print(xe21.shape)

# e22 = nn.Conv2d(8, 6, kernel_size=k, padding=1).to(device) 
# xe22 = relu(e22(xe21))
# print(xe22.shape)

# pool2 = nn.MaxPool2d(kernel_size=3, stride=2).to(device) 
# xp2 = pool2(xe22)
# print(xp2.shape)

# e31 = nn.Conv2d(6, 4, kernel_size=k, padding=1).to(device) 
# xe31 = relu(e31(xp2))
# print(xe31.shape)

# e32 = nn.Conv2d(4, 2, kernel_size=k, padding=1).to(device) 
# xe32 = relu(e32(xe31))
# print(xe32.shape)

# pool3 = nn.MaxPool2d(kernel_size=2, stride=2).to(device) 
# xp3 = pool3(xe32)
# print(xp3.shape)

# e41 = nn.Conv2d(2, 1, kernel_size=k, padding=1).to(device) 
# xe41 = relu(e41(xp3)) 
# print(xe41.shape)

# e42 = nn.Conv2d(1, 1, kernel_size=k, padding=1).to(device) 
# xe42 = relu(e42(xe41))
# print(xe42.shape)

# %%
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # Encoder
        self.e11 = nn.Conv2d(14, 12, kernel_size=13, padding=1) 
        self.e12 = nn.Conv2d(12, 10, kernel_size=13, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2) 

        self.e21 = nn.Conv2d(10, 8, kernel_size=13, padding=1) 
        self.e22 = nn.Conv2d(8, 6, kernel_size=13, padding=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2) 

        self.e31 = nn.Conv2d(6, 4, kernel_size=13, padding=1) 
        self.e32 = nn.Conv2d(4, 2, kernel_size=13, padding=1) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e41 = nn.Conv2d(2, 1, kernel_size=13, padding=1) 
        self.e42 = nn.Conv2d(1, 1, kernel_size=13, padding=1) 


    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        out = relu(self.e42(xe41))

        return out

# %%
model = UNet(n_class=len(labels['AFG'])).to(device)
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
    out = model(xb)
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
        yb_h = model(xb)
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
        print("epoch: %d, train loss: %.6f, val loss: %.6f, rmse: %.2f" %(epoch, train_loss,val_loss,accuracy))
# %%
num_epochs=25
train_val(num_epochs, model, loss_func, opt, train_dl, val_dl=train_dl)
# %%
path2weights=str("./models/weights_"
                 +datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                 +"_epoch_"+str(num_epochs)+".pt")
torch.save(model.state_dict(), path2weights)
# %%
_model = UNet(n_class=len(labels['AFG']))
weights=torch.load(path2weights)
_model.load_state_dict(weights)
_model.eval()
_model.to(device)
# %%
n = 20
x = train_ds[n]['obsvariable'].unsqueeze(0).type(torch.float).to(device)
print(x.shape)
y = train_ds[n]['groundtruth'].to(device)
output=_model(x)
print(output.shape)
# fig, ((ax1,ax2,ax3,ax4,ax5),
#       (ax6,ax7,ax8,ax9,ax10)) = plt.subplots(nrows=2, ncols=5)
# ax1.imshow(output[0,0,:,:].cpu().detach().numpy())
# ax2.imshow(output[0,1,:,:].cpu().detach().numpy())
# ax3.imshow(output[0,2,:,:].cpu().detach().numpy())
# ax4.imshow(output[0,3,:,:].cpu().detach().numpy())
# ax5.imshow(output[0,4,:,:].cpu().detach().numpy())
# ax6.imshow(y[0,:,:].cpu().detach().numpy())
# ax7.imshow(y[1,:,:].cpu().detach().numpy())
# ax8.imshow(y[2,:,:].cpu().detach().numpy())
# ax9.imshow(y[3,:,:].cpu().detach().numpy())
# ax10.imshow(y[4,:,:].cpu().detach().numpy())
fig, (ax1, ax6) = plt.subplots(nrows=1, ncols=2)
ax1.imshow(output[0,0,:,:].cpu().detach().numpy())
ax6.imshow(lognorm_dist.cdf(y[0,:,:].cpu().detach().numpy()))
# %%
