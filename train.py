# %% import packages
from constants import labels, signals, ntiles
from util import OpenSendaiBenchDataset, fitlognorm
from model import ModifiedResNet50, Segmentation

from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

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
        # %%
        opt = optim.Adam(model.parameters(), lr=1e-4)
        opt.step()
        opt.zero_grad()
        scheduler = lr_scheduler.LinearLR(opt, 
                                          start_factor=1.0, 
                                          end_factor=0.3, 
                                          total_iters=10)
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
            scheduler.step()
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
        num_epochs = 100
        train_val(num_epochs,  model.to(device), loss_func, 
                  opt, train_dl, val_dl=valid_dl)
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
        import matplotlib.pyplot as plt
        lognorm_dist = lognorm_dist_list[country]
        # %%
        iterator = iter(test_dl)
        batch = next(iterator)
        xb = batch['obsvariable'].type(torch.float).to(device)
        yb = batch['groundtruth'].type(torch.float).to(device)
        yb_h = _model(xb)

        ### cdf charts
        fig, axs = plt.subplots(nrows=2,
                                ncols=len(lognorm_dist_list[country]),layout='compressed')
        for i in range(len(lognorm_dist_list[country])):

            # ground truth
            f = axs[0,i].imshow(yb[0,i,:,:].cpu().detach().numpy(),
                        cmap='viridis', vmin=0, vmax=1)
            axs[0,i].set_title(str('GT - ' + 
                                   str(list(lognorm_dist_list[country].keys())[i])))

            # model prediction
            f1 = axs[1,i].imshow(yb_h[0,i,:,:].cpu().detach().numpy(),
                        cmap='viridis', vmin=0, vmax=1)
            axs[1,i].set_title(str('ES - ' + 
                                   str(list(lognorm_dist_list[country].keys())[i])))

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        cbar = fig.colorbar(f, shrink=0.95)
        cbar = fig.colorbar(f1, shrink=0.95)


        ###
        # max_value = max(lognorm_dist.ppf(yb[0,0,:,:].cpu().detach().numpy()).max(),
        #                 lognorm_dist.ppf(yb_h[0,0,:,:].cpu().detach().numpy()).max())
        # fig1, axs1 = plt.subplots(nrows=1,ncols=2,layout='compressed')
        # f1 = axs1[0].imshow(lognorm_dist.ppf(yb[0,0,:,:].cpu().detach().numpy()),
        #                 cmap='viridis', vmin=0, vmax=max_value)
        # axs1[0].set_title('Groundtruth, nbldg')
        # f1 = axs1[1].imshow(lognorm_dist.ppf(yb_h[0,0,:,:].cpu().detach().numpy()),
        #                 cmap='viridis', vmin=0, vmax=max_value)
        # axs1[1].set_title('Estimated, nbldg')
        # axs1 = fig.colorbar(f1, shrink=0.95)
# %%
