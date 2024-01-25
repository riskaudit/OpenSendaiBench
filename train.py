# %% import packages
from constants import labels, signals, ntiles
from util import OpenSendaiBenchDataset, fitlognorm
from model import ModifiedResNet50, Segmentation

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
# %% lognorm fit - can be integrated inside the for-loop (for next step)
lognorm_dist_list = fitlognorm(groundtruth_path=
                               '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/groundtruth/METEOR_PROJECT_2002/')
# %% national or country-level
idx = np.random.RandomState(seed=821).permutation(100)+1
iTrain, iTest, iValid = idx[:80], idx[80:90], idx[90:]
# %%
for icountry in range(len(list(labels.keys()))):
    country = list(labels.keys())[icountry]
    if not ntiles[country] != 100:
        train_ds = OpenSendaiBenchDataset(  obsvariables_path = 
                                            '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/obsvariables/METEOR_PROJECT_2002/',
                                            groundtruth_path = 
                                            '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/groundtruth/METEOR_PROJECT_2002/',
                                            ifile = iTrain,
                                            country = country, 
                                            signal = signals[country],
                                            lognorm_dist = lognorm_dist_list[country])
        test_ds  = OpenSendaiBenchDataset(  obsvariables_path = 
                                            '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/obsvariables/METEOR_PROJECT_2002/',
                                            groundtruth_path = 
                                            '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/groundtruth/METEOR_PROJECT_2002/',
                                            ifile = iTest,
                                            country = country, 
                                            signal = signals[country],
                                            lognorm_dist = lognorm_dist_list[country])
        valid_ds = OpenSendaiBenchDataset(  obsvariables_path = 
                                            '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/obsvariables/METEOR_PROJECT_2002/',
                                            groundtruth_path = 
                                            '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/groundtruth/METEOR_PROJECT_2002/',
                                            ifile = iValid,
                                            country = country, 
                                            signal = signals[country],
                                            lognorm_dist = lognorm_dist_list[country])
        train_dl = DataLoader(train_ds, batch_size=10, shuffle=True)
        test_dl  = DataLoader(test_ds)
        valid_dl = DataLoader(valid_ds)
        # %%
        loss_func = nn.L1Loss()
        iterator = iter(train_dl)
        model = ModifiedResNet50(country).to(device)

        # for batch_idx in range(len(train_dl)):
        #     data_batch = next(iterator)
        #     xb = data_batch['obsvariable'].type(torch.float).to(device)
        #     print(xb.shape)
        #     yb = data_batch['groundtruth'].type(torch.float).to(device)
        #     # out = (torch.reshape(torch.sigmoid(model(xb)),
        #     #                         (train_dl.batch_size,len(labels[country]),8,8)).to(device)-0.5)/0.5
        #     out = model(xb)
        #     print(out.shape)
        #     loss = loss_func(out, yb)
        #     print(loss)
        #     print(loss.item())
        #     break

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
                print("epoch: %d, train loss: %.10f, val loss: %.10f, rmse: %.6f" %(epoch, train_loss,val_loss,accuracy))

        # %%
        model.train()
        num_epochs = 50
        train_val(num_epochs,  model.to(device), loss_func, 
                  opt, train_dl, val_dl=valid_dl)
        # %%
        path2weights=str("./models/weights_"
                        +datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                        +"_epoch_"+str(num_epochs)+".pt")
        torch.save(model.state_dict(), path2weights)