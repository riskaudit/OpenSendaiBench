from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from constants import labels, signals

import torch
import torch.nn as nn
from torchvision import models
import pytorch_lightning as pl

device = torch.device("mps")

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class ModifiedResNet50(nn.Module):
    def __init__(self, inC: int, outC: int):
        super(ModifiedResNet50, self).__init__()

        self.inC = inC
        self.outC = outC

        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.model.conv1 = nn.Conv2d(inC, 64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False)
        self.model.avgpool = nn.AdaptiveAvgPool3d(output_size=(outC, 8, 8))
        self.model.fc = nn.Identity()

    def forward(self, x):
        return (torch.reshape(torch.sigmoid(self.model(x)),
                (x.shape[0],self.outC,8,8))-0.5)/0.5
    
# class Segmentation(pl.LightningModule):
#     def __init__(self,
#                  model,
#                  n_classes: int,
#                  criterion: callable,
#                  learning_rate: float):
#         super().__init__()
#         self.model = model
#         self.learning_rate = learning_rate

#     def forward(self, x):
#         """
#         Implement forward function.
#         :param x: Inputs to model.
#         :return: Outputs of model.
#         """
#         return self.model(x)

#     def training_step(self, batch: dict, batch_idx: int):
#         """
#         Perform a pass through a batch of training data.
#         :param batch: Batch of image pairs
#         :param batch_idx: Index of batch
#         :return: Loss from this batch of data for use in backprop
#         """
#         x, y = batch["sar"], batch["chart"].squeeze().long()
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         self.log("train_loss", loss, sync_dist=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch["sar"], batch["chart"].squeeze().long()
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         y_hat_pred = y_hat.argmax(dim=1)
#         self.metrics.update(y_hat_pred, y)
#         self.r2_score.update(y_hat_pred.view(-1), y.view(-1))
#         return loss

#     def validation_epoch_end(self, outputs):
#         loss = torch.stack(outputs).mean().detach().cpu().item()
#         self.log("val_loss", loss, sync_dist=True)
#         self.log_dict(self.metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)
#         self.log_dict(self.r2_score.compute(), on_step=False, on_epoch=True, sync_dist=True)
#         self.metrics.reset()
#         self.r2_score.reset()

#     def test_step(self, batch, batch_idx):
#         x, y = batch["sar"], batch["chart"].squeeze().long()
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         y_hat_pred = y_hat.argmax(dim=1)
#         self.test_metrics.update(y_hat_pred, y)
#         self.test_r2_score.update(y_hat_pred.view(-1), y.view(-1))
#         return loss

#     def test_epoch_end(self, outputs):
#         loss = torch.stack(outputs).mean().detach().cpu().item()
#         self.log("test_loss", loss, sync_dist=True)
#         self.log_dict(self.test_metrics.compute(), on_step=False, on_epoch=True, sync_dist=True)
#         self.log_dict(self.test_r2_score.compute(), on_step=False, on_epoch=True, sync_dist=True)
#         self.test_metrics.reset()
#         self.test_r2_score.reset()

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(),
#                                      lr=self.learning_rate)
#         optimizer.step()
#         optimizer.zero_grad()
#         return {
#             "optimizer": optimizer
#         }
