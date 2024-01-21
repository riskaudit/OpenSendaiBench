# %%
from constants import labels

import os
import cv2
import glob
import matplotlib.pyplot as plt
from skimage import io
import rioxarray as rxr
import copy
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
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
class OpenSendaiBenchDataset(Dataset):
    """
    An implementation of a PyTorch dataset for loading pairs of observable variables and ground truth labels.
    Inspired by https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.
    """
    def __init__(self, obsvariables_path: str, groundtruth_path: str, country: str, signals: list, transform: transforms = None):
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
                groundtruth[w,:,:] = a.reshape(1,a.shape[0],a.shape[1])

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
train_ds = OpenSendaiBenchDataset( obsvariables_path="/obsvariables/", 
                                        groundtruth_path="/groundtruth/", 
                                        country='AFG', 
                                        signals = ['blue','green','red'])
                                        # signals = ['VH','VV','aerosol','blue','green','red','red1','red2','red3','nir','red4','vapor','swir1','swir2'])
print(train_ds[1]['groundtruth'].shape)
print(train_ds[1]['obsvariable'].shape)
# %%
train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
len(train_dl)
train_dl.dataset[1]['groundtruth'].shape
# %%
# class UNet(nn.Module):
#     def __init__(self, n_class):
#         super().__init__()
#         self.e11 = nn.Conv2d(3, 64, kernel_size=5, padding=1)
#         self.e12 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
#         self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=6)
#         self.outconv = nn.Conv2d(64, n_class, kernel_size=2)

#     def forward(self, x):
#         xe11 = relu(self.e11(x))
#         xe12 = relu(self.e12(xe11))
#         xu1 = self.upconv1(xe12)
#         out = F.avg_pool2d(self.outconv(xu1),(46,46),46) #, divisor_override=1)

#         return out
# %% prototyping

## encoder

# e11 = nn.Conv2d(3, 64, kernel_size=5, padding=1).to(device) # torch.Size([10, 64, 366, 366])
# xe11 = relu(e11(xb))

# e12 = nn.Conv2d(64, 64, kernel_size=5, padding=1).to(device) # torch.Size([10, 64, 364, 364])
# xe12 = relu(e12(xe11))

# pool1 = nn.MaxPool2d(kernel_size=2, stride=2).to(device) # torch.Size([10, 64, 182, 182])
# xp1 = pool1(xe12)

# e21 = nn.Conv2d(64, 128, kernel_size=5, padding=1).to(device) # torch.Size([10, 128, 180, 180])
# xe21 = relu(e21(xp1))

# e22 = nn.Conv2d(128, 128, kernel_size=5, padding=1).to(device) # torch.Size([10, 128, 178, 178])
# xe22 = relu(e22(xe21))

# pool2 = nn.MaxPool2d(kernel_size=2, stride=2).to(device) # torch.Size([10, 128, 89, 89])
# xp2 = pool2(xe22)

# e31 = nn.Conv2d(128, 256, kernel_size=5, padding=1).to(device) # torch.Size([10, 256, 87, 87])
# xe31 = relu(e31(xp2))

# e32 = nn.Conv2d(256, 256, kernel_size=5, padding=1).to(device) # torch.Size([10, 256, 85, 85])
# xe32 = relu(e32(xe31))

# pool3 = nn.MaxPool2d(kernel_size=2, stride=2).to(device) # torch.Size([10, 256, 42, 42])
# xp3 = pool3(xe32)

# e41 = nn.Conv2d(256, 512, kernel_size=5, padding=1).to(device) # torch.Size([10, 512, 40, 40])
# xe41 = relu(e41(xp3)) 

# e42 = nn.Conv2d(512, 512, kernel_size=5, padding=1).to(device) # torch.Size([10, 512, 38, 38])
# xe42 = relu(e42(xe41))

# pool4 = nn.MaxPool2d(kernel_size=2, stride=2).to(device) # torch.Size([10, 512, 19, 19])
# xp4 = pool4(xe42)

# e51 = nn.Conv2d(512, 1024, kernel_size=5, padding=1).to(device) # torch.Size([10, 1024, 17, 17])
# xe51 = relu(e51(xp4))

# e52 = nn.Conv2d(1024, 1024, kernel_size=5, padding=1).to(device) # torch.Size([10, 1024, 15, 15])
# xe52 = relu(e52(xe51))

# ## decoder
# upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=10, stride=2).to(device) # torch.Size([10, 512, 38, 38])
# xu1 = upconv1(xe52)

# xu11 = torch.cat([xu1, xe42], dim=1) # torch.Size([10, 1024, 38, 38])

# d11 = nn.Conv2d(1024, 512, kernel_size=5, padding=1).to(device) # torch.Size([10, 512, 36, 36])
# xd11 = relu(d11(xu11))

# d12 = nn.Conv2d(512, 512, kernel_size=5, padding=1).to(device) # torch.Size([10, 512, 34, 34])
# xd12 = relu(d12(xd11))

# upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=19, stride=2).to(device) # torch.Size([10, 256, 85, 85])
# xu2 = upconv2(xd12)

# xu22 = torch.cat([xu2, xe32], dim=1) # torch.Size([10, 512, 85, 85])

# d21 = nn.Conv2d(512, 256, kernel_size=5, padding=1).to(device) # torch.Size([10, 256, 83, 83])
# xd21 = relu(d21(xu22))

# d22 = nn.Conv2d(256, 256, kernel_size=5, padding=1).to(device) # torch.Size([10, 256, 81, 81])
# xd22 = relu(d22(xd21))

# upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=18, stride=2).to(device) # torch.Size([10, 128, 178, 178])
# xu3 = upconv3(xd22)

# xu33 = torch.cat([xu3, xe22], dim=1) # torch.Size([10, 256, 178, 178])

# d31 = nn.Conv2d(256, 128, kernel_size=5, padding=1).to(device) # torch.Size([10, 128, 176, 176])
# xd31 = relu(d31(xu33))

# d32 = nn.Conv2d(128, 128, kernel_size=5, padding=1).to(device) # torch.Size([10, 128, 174, 174])
# xd32 = relu(d32(xd31))

# upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=18, stride=2).to(device)  # torch.Size([10, 64, 364, 364])
# xu4 = upconv4(xd32)

# xu44 = torch.cat([xu4, xe12], dim=1) # torch.Size([10, 128, 364, 364])


# d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1).to(device) 
# xd41 = relu(d41(xu44))

# d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1).to(device) 
# xd42 = relu(d42(xd41))

# outconv = nn.Conv2d(64, len(labels['AFG']), kernel_size=7, padding=5).to(device) 
# out = outconv(xd42)
# print(out.shape)
# %%
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        # Encoder
        self.e11 = nn.Conv2d(3, 64, kernel_size=5, padding=1) 
        self.e12 = nn.Conv2d(64, 64, kernel_size=5, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e21 = nn.Conv2d(64, 128, kernel_size=5, padding=1) 
        self.e22 = nn.Conv2d(128, 128, kernel_size=5, padding=1) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e31 = nn.Conv2d(128, 256, kernel_size=5, padding=1) 
        self.e32 = nn.Conv2d(256, 256, kernel_size=5, padding=1) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e41 = nn.Conv2d(256, 512, kernel_size=5, padding=1) 
        self.e42 = nn.Conv2d(512, 512, kernel_size=5, padding=1) 
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e51 = nn.Conv2d(512, 1024, kernel_size=5, padding=1) 
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=5, padding=1) 

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=10, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=5, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=5, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=19, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=5, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=5, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=18, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=5, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=5, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=18, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=7, padding=5)

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
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = F.avg_pool2d(self.outconv(xd42),(46,46),46) #, divisor_override=1)

        return out

# %%
model = UNet(n_class=len(labels['AFG'])).to(device)
print(model)
# %%
summary(model, input_size=(3, 368, 368))
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
n = 10
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
ax6.imshow(y[0,:,:].cpu().detach().numpy())
# %%
