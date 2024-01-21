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
                                        signals = ['VH','VV','aerosol','blue','green','red','red1','red2','red3','nir','red4','vapor','swir1','swir2'])
print(train_ds[1]['groundtruth'].shape)
print(train_ds[1]['obsvariable'].shape)
# %%
train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
len(train_dl)
train_dl.dataset[1]['groundtruth'].shape
# %%
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.e11 = nn.Conv2d(14, 64, kernel_size=5, padding=1)
        self.e12 = nn.Conv2d(64, 128, kernel_size=5, padding=1)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=6)
        self.outconv = nn.Conv2d(64, n_class, kernel_size=2)

    def forward(self, x):
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xu1 = self.upconv1(xe12)
        out = F.avg_pool2d(self.outconv(xu1),(46,46),46)

        return out
# %%
model = UNet(n_class=len(labels['AFG']))
print(model)
# %%
summary(model, input_size=(14, 368, 368))
# %%
loss_func = nn.L1Loss()
iterator = iter(train_dl)

for batch_idx in range(len(train_dl)):
    data_batch = next(iterator)
    xb = data_batch['obsvariable'].type(torch.float)
    yb = data_batch['groundtruth'].type(torch.float)
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
