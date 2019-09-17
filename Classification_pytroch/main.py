# -*- coding: utf-8 -*-
"""
參考: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py
"""

#%%
import torch, os
#from torch.nn import Module
#from torch.nn import functional as F
#from torch import optim
#from torch.optim.lr_scheduler import StepLR
#import torchvision.models as models 
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from model_collect import Modle_TEST, Dataset_TEST
from utils_collect import OWNLogger
import numpy as np
#%%
batch_size = 8
num_epochs = 350 # 0 for TEST
#num_classes = 15
batch_size = 16 # 8:3.6GB,
learning_rate = 0.001

#model_weight_folder = "./result/struct1_e350_b16_b16_e350/"
#model_weight_path = "model_b16_e350.ckpt"
model_weight_folder = None
model_weight_path   = None
model_struct        = "struct2"
model_discription   = "b%d_e%d"%(batch_size, num_epochs) # 兩種輸出 3-8 比例

#processUnit = 'cpu'  # 因為 RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
processUnit = 'cuda' if torch.cuda.is_available() else 'cpu'
device_tmp = torch.device(processUnit)

#%% logger
saveFolder = "./result/{0}_e{2:0>2d}_b{3}_{1}/".format(model_struct, model_discription, num_epochs, batch_size)
#saveFolder = "./" # for TEST
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)
log = OWNLogger(logNPY=saveFolder,
                lossName=["loss_lab"])
#%% DATA LOAD
d_train = Dataset_TEST("train")
l_train = DataLoader(dataset=d_train, 
                     batch_size=batch_size,
                     shuffle=True)

d_test  = Dataset_TEST("test")
l_test  = DataLoader(dataset=d_test, 
                     batch_size=batch_size,
                     shuffle=False)

total_step = len(l_train)
#%% model
model_main = Modle_TEST(num_resBlock=1).to(device_tmp)

summary(model_main, input_size=(3, 224, 224), device=processUnit) 
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
if model_weight_folder:
#    if processUnit == "cpu":
    model_main.load_state_dict(torch.load(model_weight_folder + model_weight_path,
                                          map_location='cpu' if processUnit == "cpu" else None))
#    else:
#        model_main.load_state_dict(torch.load(model_weight_folder + model_weight_path))
#%%
# LOG
log.ShowLocalTime()
log.UpdateProgSetting(itrMax = total_step, 
                      batch_size = batch_size, 
                      epochs = num_epochs, 
                      model_weight_folder = model_weight_folder,
                      model_weight_path = model_weight_path,
                      model_discription = model_discription)
min_loss_avg = 9999
#%% Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_main.parameters(), lr=learning_rate)  
#%% Train the model
log.SetLogTime("train")
for epoch in tqdm.tqdm(range(num_epochs)):
    log.SetLogTime("e%02d"%(epoch), boolPrint=True)
    loss_list = list()
    for _i, (img, lab, attr) in enumerate(l_train):
        img_ten  = (img / 255.0).float().to(device_tmp) 
        lab_ten  = lab.long().to(device_tmp)
#        attr_ten = attr.float().to(device_tmp)
        
        # Forward pass
        outputs = model_main(img_ten)
        loss = criterion(outputs, lab_ten)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (_i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, _i+1, total_step, loss.item()))
        # LOSS cont
        loss_list.append(loss.item())
    loss_avg = np.average(loss_list)
    log.AppendLossIn("loss_lab",  loss_avg)
    
    print('Epoch [{}/{}], Loss avg:{:.4f}'
          .format(epoch+1, num_epochs, loss_avg))
    if loss_avg > min_loss_avg:
        torch.save(model_main.state_dict(), '%s%s_%s_e%03d_lo%.3f.ckpt'%(saveFolder, model_struct, model_discription, epoch, loss_avg))
    log.SetLogTime("e%02d"%(epoch), mode = "end")
log.SetLogTime("train", mode = "end")
log.SaveLog2NPY(boolPrint=True)
#%% Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad(): # model_main.eval() # 似乎同意義?
    correct = 0
    total = 0
    for images, labels, attributes in l_test:
        img_ten  = (images / 255.0).float().to(device_tmp) 
        lab_ten  = labels.long().to(device_tmp)
#        attr_ten = attributes.float().to(device_tmp)
        
        outputs = model_main(img_ten)
        _, predicted = torch.max(outputs.data, 1)
        total += lab_ten.size(0)
        correct += (predicted == lab_ten).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(len(d_test), 100 * correct / total))

# Save the model checkpoint
torch.save(model_main.state_dict(), '%smodel_%s.ckpt'%(saveFolder, model_discription))
#%% 分析
if num_epochs != 0:
    from utils_collect import ShowLossAnalysisFigNPY_1, CalEpochTimeCost
    ShowLossAnalysisFigNPY_1(log.logNPY);
    CalEpochTimeCost(log.logNPY);