# %% import packages
from constants import labels, signals, ntiles
from util import OpenSendaiBenchDataset, fitlognorm
from model import ModifiedResNet50

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
# %%
    country = list(labels.keys())[icountry]
    if not ntiles[country] != 100:
        # %%
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
        t = 0

        ### cdf charts
        fig, axs = plt.subplots(nrows=2,
                                ncols=len(lognorm_dist_list[country]),layout='compressed',
                                figsize=(10,5))
        for w in range(len(labels[country])): 

            # ground truth
            f = axs[0,w].imshow(yb[t,w,:,:].cpu().detach().numpy(),
                        cmap='viridis', vmin=0, vmax=1)
            axs[0,w].set_title(str('Groundtruth - ' + 
                                   str(list(lognorm_dist_list[country].keys())[w])))

            # model prediction
            f1 = axs[1,w].imshow(yb_h[t,w,:,:].cpu().detach().numpy(),
                        cmap='viridis', vmin=0, vmax=1)
            axs[1,w].set_title(str('Estimated - ' + 
                                   str(list(lognorm_dist_list[country].keys())[w])))

        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        cbar = fig.colorbar(f, shrink=0.95)
        cbar = fig.colorbar(f1, shrink=0.95)
        fig.savefig(str('multibldgtype_cdf_'+country+'.png'),
                    bbox_inches='tight')

        ### nbldg charts
        # get the max first - for cbar use
        max_value = []
        for w in range(len(labels[country])): 
            gt_max = lognorm_dist[labels[country][w]]['modelfit'].ppf(yb[t,w,:,:].cpu().detach().numpy()).round().max()
            es_max = lognorm_dist[labels[country][w]]['modelfit'].ppf(yb_h[t,w,:,:].cpu().detach().numpy()).round().max()
            max_value.append(max(gt_max, es_max))

        # create the plot 
        fig1, axs = plt.subplots(nrows=2,
                                ncols=len(lognorm_dist_list[country]),layout='compressed',
                                figsize=(15, 5))
        for w in range(len(labels[country])): 

            # ground truth
            f = axs[0,w].imshow(lognorm_dist[labels[country][w]]['modelfit'].ppf(yb[t,w,:,:].cpu().detach().numpy()).round(), cmap='viridis', vmin=0, vmax=max_value[w])
            axs[0,w].set_title(str('Groundtruth - ' + 
                                   str(list(lognorm_dist_list[country].keys())[w])))
            axs[0,w].set_yticklabels([])
            axs[0,w].set_xticklabels([])
            axs[0,w].set_xticks([])
            axs[0,w].set_yticks([])
            cbar = fig1.colorbar(f, ax=axs[0,w], 
                                 ticks=[0, max_value[w]],
                                 orientation="horizontal", shrink=0.95)

            # model prediction
            f1 = axs[1,w].imshow(lognorm_dist[labels[country][w]]['modelfit'].ppf(yb_h[t,w,:,:].cpu().detach().numpy()).round(), cmap='viridis', vmin=0, vmax=max_value[w])
            axs[1,w].set_title(str('Estimated - ' + 
                                   str(list(lognorm_dist_list[country].keys())[w])))
            axs[1,w].set_yticklabels([])
            axs[1,w].set_xticklabels([])
            axs[1,w].set_xticks([])
            axs[1,w].set_yticks([])
            cbar = fig1.colorbar(f1, ax=axs[1,w], 
                                 ticks=[0, max_value[w]],
                                 orientation="horizontal", shrink=0.95)
            
        fig1.savefig(str('multibldgtype_nbldg_'+country+'.png'),
                     bbox_inches='tight')

# %%
