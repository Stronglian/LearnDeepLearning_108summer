# -*- coding: utf-8 -*-
"""
模組架設
"""
"""
https://pytorch.org/docs/stable/torchvision/models.html
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
"""
#import torchvision.models as m
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

#m.AlexNet # model

alexnet_LOAD = torchvision.models.alexnet(pretrained=True) # call model

print(alexnet_LOAD.named_modules)

#%% RES BLOCK
# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out