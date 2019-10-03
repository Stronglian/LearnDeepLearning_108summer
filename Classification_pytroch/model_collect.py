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
import torch.nn as nn # 與 from torch import nn 有何不同
from torch.nn import functional as F
#import torchvision.transforms as transforms

from torch.utils.data import Dataset #, DataLoader
from torchsummary import summary # pip install torchsummary
#%% TEST

#model_features = torchvision.models.alexnet(pretrained=True)#.features # call model

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
    
    def __init__(self, num_resBlock=1, num_classes=15, type_cla=0, useNet="alexNet", boolUseAttr=False):
        """
        num_resBlock 沒作用? 還是無法被顯示?
        """
        net_list = ["alexNet", "vgg"]
        super(Modle_TEST, self).__init__()
        if useNet not in net_list:
            raise ValueError("only", *net_list)
        self.useNet       = useNet
        self.num_resBlock = num_resBlock
        self.type_cla     = type_cla
        self.boolUseAttr  = boolUseAttr
        self.num_classes  = num_classes
        # 網路
        if self.useNet == "alexNet":
            self.extraction_features = torchvision.models.alexnet(pretrained=True).features
        elif self.useNet == "vgg":
            self.extraction_features = torchvision.models.vgg19(pretrained=True).features
        
        if self.useNet == "alexNet":
            feature_extraction_out_channel = 256
        elif self.useNet == "vgg":
            feature_extraction_out_channel = 512
#        self.resBlock = ResidualBlock(in_channels = feature_extraction_out_channel, out_channels = feature_extraction_out_channel)
        # 複數 res net 
        self.resBlocks = nn.Sequential()
        for _i in range(self.num_resBlock):
            self.resBlocks.add_module("ResBlcok%d"%(_i), 
                                      ResidualBlock(in_channels = feature_extraction_out_channel, 
                                                    out_channels = feature_extraction_out_channel))
        if self.type_cla in [0, 1, 2]:
            if self.useNet == "alexNet":
                resblock_out_channel = 256 * 6 * 6 # 9216
            elif useNet == "vgg":
                resblock_out_channel = 512 * 7 * 7 #25088
#        elif type_cla in [3] and useNet == "alexNet":
#            resblock_out_channel = 256
        # 
        if self.type_cla == 0:
            self.classifier = nn.Sequential( 
                nn.Dropout(p=0.8),
                nn.Linear(resblock_out_channel, 4096), 
                nn.ReLU(inplace=True),
                
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                
                nn.Dropout(p=0.5),
                nn.Linear(4096, 2048),
                nn.ReLU(inplace=True),
                
                nn.Linear(2048, num_classes),
            )
        elif self.type_cla == 1:
            self.classifier = nn.Sequential( 
                nn.Dropout(p=0.8),
                nn.Linear(resblock_out_channel, 4096), 
                nn.ReLU(inplace=True),
                
                nn.Linear(4096, num_classes),
            )
        elif self.type_cla == 2: #直接接出來
            self.classifier = nn.Sequential( 
                nn.Linear(resblock_out_channel, num_classes),
            )
        elif self.type_cla == 3:
            self.classifier_pre = nn.Sequential( 
                nn.Conv2d(feature_extraction_out_channel, 128, 
                          kernel_size=3), # , stride=1, padding=0
                nn.Conv2d(128, 64, 
                          kernel_size=4), 
                )
            self.classifier = nn.Sequential( 
                nn.Linear(64, num_classes),
            )
        # USE attr
        if self.boolUseAttr:
            pass
        return
    
    def forward(self, x, **darg):
        if self.boolUseAttr:
            attr = darg["attr"]
        data = self.extraction_features(x)  # struct 2 VGG
#        print("extraction_features =>", data.size(), flush=True)
        
        if self.num_resBlock > 0:
            data = self.resBlocks(data)
#            print("resBlock =>", data.size(), flush=True)
            
        if self.type_cla in [3]:
            data = self.classifier_pre(data)
        data = data.view((data.size(0), -1)) # FLATTEN
#        print("view =>", data.size(), flush=True)
        
        data = self.classifier(data)
#        print("classifier =>", data.size(), flush=True)
#        
        return data # struct 1
#        return F.softmax(data, dim = 1) # struct 2
#%% input img and attr
class Modle_Attr(nn.Module): # struct 3
    def __init__(self, num_resBlock=0, num_classes=15, type_cla=3, useNet="alexNet", boolUseAttr=True):
        super(Modle_Attr, self).__init__()
        # parm
        self.num_resBlock = num_resBlock
        self.num_classes  = num_classes
        self.type_cla     = type_cla
        self.useNet       = useNet
        self.boolUseAttr  = boolUseAttr
        # 
        if self.useNet == "alexNet":
            self.extraction_features = torchvision.models.alexnet(pretrained=True).features
        feature_extraction_out_channel = 256
        # ATTR 轉換成可用
        self.transAttr = nn.Sequential(
                nn.Linear(85, 64)
                )
        # 複數 res net 
        if self.num_resBlock > 0:
            self.resBlocks = nn.Sequential()
            for _i in range(self.num_resBlock):
                self.resBlocks.add_module("ResBlcok%d"%(_i), 
                                          ResidualBlock(in_channels  = feature_extraction_out_channel, 
                                                        out_channels = feature_extraction_out_channel))
        # class
        if self.type_cla == 3:
            self.classifier_pre = nn.Sequential( 
                nn.Conv2d(feature_extraction_out_channel, 128, 
                          kernel_size=3), # , stride=1, padding=0
                nn.Conv2d(128, 64, 
                          kernel_size=4), 
                )
            self.classifier = nn.Sequential( 
                nn.Linear(64 + 64, num_classes),
            )
    def forward(self, img, attr):
        # 1st IN
        data = self.extraction_features(img)  # struct 2 VGG
#        print("extraction_features =>", data.size(), flush=True)
        
            
        if self.num_resBlock > 0:
            data = self.resBlocks(data)
#            print("resBlock =>", data.size(), flush=True)
            
        if self.type_cla in [3]:
            data = self.classifier_pre(data)
        data = data.view((data.size(0), -1)) # FLATTEN
#        print("view =>", data.size(), flush=True)
        
        attr_in = self.transAttr(attr)
        attr_in = data.view((attr_in.size(0), -1)) # FLATTEN
#        print("transAttr =>", attr_in.size(), flush=True)
        
        data = torch.cat((data, attr_in), dim=1)
        
        data = self.classifier(data)
#        print("classifier =>", data.size(), flush=True)
        
        return data
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
    model_tmp = Modle_Attr(num_resBlock=1, num_classes=num_classes, useNet="alexNet", type_cla=3).to(device_tmp)
    # summary
    summary(model_tmp, input_size=[(3, 224, 224), (1, 85)], device=processUnit) 
    
