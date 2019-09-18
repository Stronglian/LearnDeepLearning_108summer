# -*- coding: utf-8 -*-
"""
模組架設
"""
"""
https://pytorch.org/docs/stable/torchvision/models.html
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""
#import torchvision.models as m
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
#import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchsummary import summary # pip install torchsummary
#%% TEST

#model_features = torchvision.models.alexnet(pretrained=True)#.features # call model
#
#print(model_features)

#%% RES BLOCK
"""
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py
"""
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

#%%
class Modle_TEST(nn.Module):
    
    def __init__(self, num_resBlock=1, num_classes=15, useNet="alexNet"):
        """
        num_resBlock 沒作用? 還是無法被顯示?
        """
        net_list = ["alexNet", "vgg"]
        super(Modle_TEST, self).__init__()
        if useNet not in net_list:
            raise ValueError("only", *net_list)
        self.useNet = useNet
        # 網路
        if useNet == "alexNet":
            self.extraction_features = torchvision.models.alexnet(pretrained=True).features
        elif useNet == "vgg":
            self.extraction_features = torchvision.models.vgg19(pretrained=True).features
        
        if useNet == "alexNet":
            self.resBlock = ResidualBlock(in_channels = 256, out_channels = 256)
        elif useNet == "vgg":
            self.resBlock = ResidualBlock(in_channels = 512, out_channels = 512)
        # 要怎麼設計 複數 res net 
#        self.num_resBlock = num_resBlock
#        self.resBlock = list()
#        for _i in range(self.num_resBlock):
#            self.resBlock.append(ResidualBlock(in_channels = 256, out_channels = 256))
        
#            self.droup1  = nn. Dropout(p=0.5)
#            self.linear1 = nn.Linear(in_features=512 * 7 * 7, out_features=4096, bias=True) # If set to ``False``, the layer will not learn an additive bias.
#            self.relu1   = nn.ReLU()
#            
#            self.droup2  = nn.Dropout(p=0.5)
#            self.linear2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
#            self.relu2   = nn.ReLU()
#            
#            self.droup3  = nn.Dropout(p=0.5)
#            self.linear3 = nn.Linear(in_features=4096, out_features=2048, bias=True)
#            self.relu3   = nn.ReLU()
#            
#            self.linear4 = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
            
        if useNet == "alexNet":
            classifier_in_channel = 256 * 6 * 6
        if useNet == "vgg":
            classifier_in_channel = 512 * 7 * 7
        # 以 parm grad 算，後面有六層?
        self.classifier = nn.Sequential( 
            nn.Dropout(p=0.8),
            nn.Linear(classifier_in_channel, 4096), # 9216
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            
            nn.Linear(2048, num_classes),
        )
        
        return
    
    def forward(self, x):
        data = self.extraction_features(x)  # struct 2 VGG
        print("extraction_features =>", data.size(), flush=True)
        
        data = self.resBlock(data)
        print("resBlock =>", data.size(), flush=True)
#        for _i in range(self.num_resBlock):
#            data = self.resBlock[_i](data)
            
        data = data.view((data.size(0), -1)) # FLATTEN
        print("view =>", data.size(), flush=True)
        
#        if self.useNet == "vgg":
#            data = self.droup1(data)
#            data = self.linear1(data) # dense
#            data = self.relu1(data)
#            
#            data = self.droup2(data)
#            data = self.linear2(data)
#            data = self.relu2(data)
#            
#            data = self.droup3(data)
#            data = self.linear3(data)
#            data = self.relu3(data)
#            
#            data = self.linear4(data)
#        if self.useNet == "alexNet":
        data = self.classifier(data)
        print("classifier =>", data.size(), flush=True)
#        
#        return data # struct 1
        return F.softmax(data, dim = 1) # struct 2
#%%
class Dataset_TEST(Dataset):
    def __init__(self, t_type, strFolderData = "../_DataSet/forP/"):
        """
            t_type: train / test
            d_type: image / label / attribute
        """
        if t_type not in ["train", "test"]:
            raise ValueError("t_type:%s not in [train, test]"%(t_type))
        else:
            self.t_type = t_type
        self.UpdateDataset(strFolderData, boolNew = True)
        
    def UpdateDataset(self, strFolderData, boolNew = False, t_key = "TMP"):
        """
            t_type: train / test
            d_type: image / label / attribute
        """
        if boolNew:
            self.dataSet = {"test":{}, "train":{}}
            for _n in os.listdir(strFolderData):
                t_type_tmp = _n.split("_", 1)[0]
                if t_type_tmp != self.t_type:
                    continue
                d_type_tmp = _n.split("_", 1)[1].split(".", 1)[0]
                self.dataSet[t_type_tmp][d_type_tmp] = np.load(strFolderData+_n) 
        else:
            for _n in os.listdir(strFolderData):
                self.dataSet[t_key] = np.load(strFolderData+_n) 
        # 
#        print("shape:", self.dataSet[self.t_type]["image"].shape, end = "")
        self.dataSet[self.t_type]["image"] = self.dataSet[self.t_type]["image"].reshape(len(self.dataSet[self.t_type]["image"]), 3, 224, 224)
#        print("=>", self.dataSet[self.t_type]["image"].shape)
        return
    
    def __getitem__(self, index):
        return (self.dataSet[self.t_type]["image"][index], 
                self.dataSet[self.t_type]["label"][index], 
                self.dataSet[self.t_type]["attribute"][index])
    def __len__(self):
        return len(self.dataSet[self.t_type]["label"])
#%%
#def LoadImage(strFolderData = "../_DataSet/forP/", strDataNPY = "train_image.npy", intNum = 2):
#    dataImg = np.load(strFolderData+strDataNPY, allow_pickle=True)
#    return dataImg[:intNum]
#%%
if __name__ == "__main__":
    hidden_size = 500
    num_classes = 15
    num_epochs = 0
    batch_size = 8
    learning_rate = 0.001
    
    processUnit = 'cpu'  # 因為 RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
#    processUnit = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_tmp = torch.device(processUnit)
#    if torch.cuda.is_available() and False:
#        torch.set_default_tensor_type('torch.cuda.FloatTensor')
#    else:
#        torch.set_default_tensor_type('torch.FloatTensor')
        
#%%
    model_tmp = Modle_TEST(num_classes=num_classes, useNet="alexNet").to(device_tmp)
    # summary
    summary(model_tmp, input_size=(3, 224, 224), device=processUnit) 
    
#%%
for _i, param in enumerate(model_tmp.parameters()):
    print(_i, "=>", param.requires_grad)
#    param.requires_grad = False
#%% LOAD    
#    d_train = Dataset_TEST("train")
#    l_train = DataLoader(dataset=d_train, 
#                         batch_size=batch_size, 
#                         shuffle=True) 
#%% Train
#    total_step = len(l_train)
#    for epoch in range(num_epochs):
#        for _i, (img, lab, attr) in enumerate(l_train):
#            img_ten  = (img/ 255.0).float().to(device_tmp) 
##            lab_ten  = lab.long().to(device_tmp)
##            attr_ten = attr.float().to(device_tmp)
#            # Forward pass
#            outputs = model_tmp(img_ten)
 
