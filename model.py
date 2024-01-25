from constants import labels, signals

import torch
import torch.nn as nn
from torchvision import models

class ModifiedResNet50(nn.Module):
    def __init__(self, country: str):
        super(ModifiedResNet50, self).__init__()
        self.country = country
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.model.conv1 = nn.Conv2d(len(signals[country]),64, kernel_size = (7,7), stride = (2,2), padding = (3,3), bias = False)
        self.model.avgpool = nn.AdaptiveAvgPool3d(output_size=(len(labels[country]), 8, 8))
        self.model.fc = nn.Identity()

    def forward(self, x):
        return (torch.reshape(torch.sigmoid(self.model(x)),
                (x.batch_size,len(labels[self.country]),8,8))-0.5)/0.5