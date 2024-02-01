# %% Gen Info
# Six Most Frequent Building Types
# A (Adobe blocks (unbaked sundried mud block) walls)
# C3L (Nonductile reinforced concrete frame with masonry infill walls low-rise) # INF (Informal constructions.)
# UCB (Concrete block unreinforced masonry with lime or cement mortar)
# UFB (Unreinforced fired brick masonry)
# W5 (Wattle and Daub (Walls with bamboo/light timber log/reed mesh and post).)

# Three Most Frequent Building Type Groups
# C3L, C3M, C3H - nonductile concrete
# W, W1, W2, W3, W5 - wooden constructions

# %% import packages
from constants import labels, signals, ntiles
from util import OpenSendaiBenchDatasetGlobal, fitlognorm
from model import ModifiedResNet50

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import glob
import scipy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
device = torch.device("mps")
# %% lognorm fit - can be integrated inside the for-loop (for next step)
lognorm_dist_list = fitlognorm(groundtruth_path=
                               '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/groundtruth/METEOR_PROJECT_2002/')
# %% global level
bldgtype_list = ['A', 'C3L', 'INF', 'UCB', 'UFB', 'W5']
bldgtype_group_list = {'grp1': ['C3L', 'C3M', 'C3H'], 
                         'grp2': ['W', 'W1', 'W2', 'W3', 'W5']}
obsvariables_path = '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/obsvariables/METEOR_PROJECT_2002/'
groundtruth_path = '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/groundtruth/METEOR_PROJECT_2002/'
ratio_train = 0.8
ratio_val = 0.1
ratio_test = 0.1
arg_inputs = ['S2'] # ['S1', 'S1+S2', 'S2']
# %%a
for i in range(len(bldgtype_list)):
    # %%
    bldgtype = bldgtype_list[i]

    TrainFile = []
    TestFile = []
    ValidFile = []
    for j in range(len(signals)): # country index
        country = list(labels.keys())[j] # select country

        # this gets the filename of ground truth, not obsvariables
        if 'S1+S2' in arg_inputs or 'S1' in arg_inputs:
            if 'VV' in signals[country] and bldgtype in labels[country]:
                a = glob.glob(str(groundtruth_path+country+'*/tiles/images/'+country+'_nbldg_'+bldgtype+'_'+'*.tif'))
                a.sort()

                f_remaining, f_test = train_test_split(a, test_size=ratio_test,random_state=j)
                ratio_remaining = 1 - ratio_test
                ratio_val_adjusted = ratio_val / ratio_remaining
                f_train, f_val = train_test_split(f_remaining, test_size=ratio_val_adjusted,random_state=j)

                TrainFile += f_train
                TestFile += f_test
                ValidFile += f_val
        elif 'S2' in arg_inputs:
            if bldgtype in labels[country]: # because we all know the S2 exists for all countries
                a = glob.glob(str(groundtruth_path+country+'*/tiles/images/'+country+'_nbldg_'+bldgtype+'_'+'*.tif'))
                a.sort()

                f_remaining, f_test = train_test_split(a, test_size=ratio_test,random_state=j)
                ratio_remaining = 1 - ratio_test
                ratio_val_adjusted = ratio_val / ratio_remaining
                f_train, f_val = train_test_split(f_remaining, test_size=ratio_val_adjusted,random_state=j)

                TrainFile += f_train
                TestFile += f_test
                ValidFile += f_val


    # %%
    train_ds = OpenSendaiBenchDatasetGlobal(obsvariables_path = 
                                            '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/obsvariables/METEOR_PROJECT_2002/',
                                            FilePathList = TrainFile,
                                            bldgtype = bldgtype,
                                            signal = ['aerosol', 'blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'vapor', 'swir1', 'swir2'],
                                            lognorm_dist_list = lognorm_dist_list)
    test_ds  = OpenSendaiBenchDatasetGlobal(obsvariables_path = 
                                            '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/obsvariables/METEOR_PROJECT_2002/',
                                            FilePathList = TestFile,
                                            bldgtype = bldgtype,
                                            signal = ['aerosol', 'blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'vapor', 'swir1', 'swir2'],
                                            lognorm_dist_list = lognorm_dist_list)
    valid_ds = OpenSendaiBenchDatasetGlobal(obsvariables_path = 
                                            '/Users/joshuadimasaka/Desktop/PhD/GitHub/riskaudit/data/obsvariables/METEOR_PROJECT_2002/',
                                            FilePathList = ValidFile,
                                            bldgtype = bldgtype,
                                            signal = ['aerosol', 'blue', 'green', 'red', 'red1', 'red2', 'red3', 'nir', 'red4', 'vapor', 'swir1', 'swir2'],
                                            lognorm_dist_list = lognorm_dist_list)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    test_dl  = DataLoader(test_ds)
    valid_dl = DataLoader(valid_ds, num_workers=0, pin_memory=True)
    # %%
    loss_func = nn.MSELoss() 
    iterator = iter(train_dl)
    if 'S1+S2' in arg_inputs:
        inC = 14
    elif 'S1' in arg_inputs:
        inC = 2
    elif 'S2' in arg_inputs:
        inC = 12

    # %%
    model = ModifiedResNet50(inC= inC, 
                             outC= 1).to(device)
    # %%
    opt = optim.Adam(model.parameters(), lr=1e-4)
    opt.step()
    opt.zero_grad()
    scheduler = lr_scheduler.LinearLR(opt, 
                                        start_factor=1.0, 
                                        end_factor=0.3, 
                                        total_iters=10)
    # %%
    class MSE(nn.Module):
        def __init__(self):
            super().__init__()
            self.mse = nn.MSELoss()
        def forward(self,yhat,y):
            mse = self.mse(yhat,y)
            return mse
    class MAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.mae = nn.L1Loss()
        def forward(self,yhat,y):
            mae = self.mae(yhat,y)
            return mae
    mae_class = MAE()
    mse_class = MSE()
    def checkpoint(model, filename):
        torch.save(model.state_dict(), filename)
    def metrics_batch(target, output):
        return mae_class(output,target), mse_class(output,target) 
    def loss_batch(loss_func, xb, yb,yb_h, opt=None):
        loss = loss_func(yb_h, yb)
        mae_b, mse_b = metrics_batch(yb,yb_h)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        return loss.item(), mae_b, mse_b
    def loss_epoch(model,loss_func,dataset_dl,opt=None):
        loss=0.0
        mae_out=0.0
        mse_out=0.0
        iterator = iter(dataset_dl)
        len_data = len(dataset_dl.dataset)
        for batch_idx in range(len(train_dl)):
            data_batch = next(iterator)
            xb = data_batch['obsvariable'].type(torch.float).to(device)
            yb = data_batch['groundtruth'].type(torch.float).to(device)
            yb_h = model(xb)
            loss_b,mae_b,mse_b=loss_batch(loss_func, xb, yb,yb_h, opt)
            loss+=loss_b
            if a is not None:
                mae_out+=mae_b
                mse_out+=mse_b
        scheduler.step()
        loss/=len_data
        mae_out/=len_data
        mse_out/=len_data
        return loss, mae_out, mse_out
    def train_val(epochs, model, loss_func, opt, train_dl, test_dl, val_dl):
        for epoch in range(epochs):
            model.train()
            train_loss, train_mae, train_mse=loss_epoch(model,loss_func,train_dl,opt)
            model.eval()
            with torch.no_grad():
                val_loss, val_mae, val_mse =loss_epoch(model,loss_func,val_dl)
                test_loss, test_mae, test_mse=loss_epoch(model,loss_func,test_dl)
            val_mae_out = val_mae #100*val_metric
            val_mse_out = val_mse
            test_mae_out = test_mae
            test_mse_out = test_mse
            print("epoch: %d, train loss MSE: %.6f, val loss MSE: %.6f, test_mae: %.6f, test_mse: %.6f, val_mae: %.6f, val_mse: %.6f" 
                  %(epoch, train_loss,val_loss,test_mae_out,test_mse_out,val_mae_out,val_mse_out))
    # %%
    model.train()
    num_epochs = 100
    train_val(num_epochs,  model.to(device), loss_func, 
                opt, train_dl, test_dl, valid_dl)
    # %%
    path2weights=str("./models/weights_"
                       +datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
                    +"_epoch_"+str(num_epochs)+"_bldgtype_"+bldgtype+"_MSEloss_MAE_MSE_S2only.pt")
    torch.save(model.state_dict(), path2weights)
    # %%
    _model = model #UNet(n_class=len(labels['AFG']))
    weights=torch.load(path2weights)
    _model.load_state_dict(weights)
    _model.eval()
    _model.to(device)
    # %%
    iterator = iter(test_dl)
    # %%
    batch = next(iterator)
    xb = batch['obsvariable'].type(torch.float).to(device)
    yb = batch['groundtruth'].type(torch.float).to(device)
    yb_h = _model(xb)
    t = 0 # batch number, zero if no bacth number for test_dl

    ### cdf charts
    fig, axs = plt.subplots(nrows=1, ncols=2,
                            layout='compressed',
                            figsize=(10,5))
    # ground truth
    f = axs[0].imshow(yb[t,0,:,:].cpu().detach().numpy(),
                cmap='viridis', vmin=0, vmax=1)
    axs[0].set_title(str('Groundtruth - ' + bldgtype))
    # model prediction
    f1 = axs[1].imshow(yb_h[t,0,:,:].cpu().detach().numpy(),
                cmap='viridis', vmin=0, vmax=1)
    axs[1].set_title(str('Estimated - ' + bldgtype))
    # plot
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    cbar = fig.colorbar(f, shrink=0.95)
    cbar = fig.colorbar(f1, shrink=0.95)
    fig.savefig(str('global_cdf_'+bldgtype+'.png'),
                bbox_inches='tight')

    ### nbldg charts
    # get the max first - for cbar use
    lognorm_dist =  scipy.stats.lognorm(s=batch['sigma'].cpu(), 
                                        loc=0, 
                                        scale=np.exp(batch['mu'].cpu()))
    gt_max = lognorm_dist.ppf(yb[t,0,:,:].cpu().detach().numpy()).round().max()
    es_max = lognorm_dist.ppf(yb_h[t,0,:,:].cpu().detach().numpy()).round().max()
    max_value = max(gt_max, es_max)
    # create the plot 
    fig1, axs = plt.subplots(nrows=1, ncols=2,layout='compressed',
                            figsize=(15, 5))
    # ground truth
    f = axs[0].imshow(lognorm_dist.ppf(yb[t,0,:,:].cpu().detach().numpy()).round(), cmap='viridis', vmin=0, vmax=max_value)
    axs[0].set_title(str('Groundtruth - ' + bldgtype))
    axs[0].set_yticklabels([])
    axs[0].set_xticklabels([])
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    cbar = fig1.colorbar(f, ax=axs[0], 
                            ticks=[0, max_value],
                            orientation="vertical", shrink=0.95)
    # model prediction
    f1 = axs[1].imshow(lognorm_dist.ppf(yb_h[t,0,:,:].cpu().detach().numpy()).round(), cmap='viridis', vmin=0, vmax=max_value)
    axs[1].set_title(str('Estimated - ' + bldgtype))
    axs[1].set_yticklabels([])
    axs[1].set_xticklabels([])
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    cbar = fig1.colorbar(f1, ax=axs[1], 
                            ticks=[0, max_value],
                            orientation="vertical", shrink=0.95)
    # plot
    fig1.savefig(str('global_nbldg_'+bldgtype+'.png'),
                    bbox_inches='tight')