# -*- coding: utf-8 -*-
"""
參考: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py # 錯範本
用這個啦: https://github.com/pytorch/examples/blob/master/mnist/main.py
晚點再來修
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
from utils_collect import OWNLogger, ConfusionMatrix
import numpy as np
#%% Train the model
def Train(args, model, device, train_loader, epoch, criterion, optimizer):
    model.train() # 記得這行
    loss_list = list()
    for _i, (img, lab, attr) in enumerate(train_loader):
        img_ten  = (img / 255.0).float().to(device) 
        lab_ten  = lab.long().to(device)
#        attr_ten = attr.float().to(device)
        
        optimizer.zero_grad()
        # Forward pass
        outputs = model(img_ten)
        loss = criterion(outputs, lab_ten)
#        loss = criterion(outputs, attr_ten)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        if (_i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, _i+1, total_step, loss.item()))
        # LOSS cont
        loss_list.append(loss.item())
        
    loss_avg = np.average(loss_list)
    
    print('Epoch [{}/{}], Loss avg:{:.4f}'.format(epoch+1, num_epochs, loss_avg))
        
    return loss_avg
#%% Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# 
def Test(args, model, device, test_loader, epoch, conMat=None, boolDEBUG=False):
    model.eval() # 似乎同意義?
    with torch.no_grad():
        correct = 0
        total = 0
        for _i, (images, labels, attributes) in enumerate(test_loader):
            img_ten  = (images / 255.0).float().to(device) 
            lab_ten  = labels.long().to(device)
    #        attr_ten = attributes.float().to(device_tmp)
            
            outputs = model(img_ten)
            outputs = nn.functional.softmax(outputs, dim = 1)
            
            _, predicted = torch.max(outputs, 1) # 不用 .data
            total += lab_ten.size(0)
            correct += (predicted == lab_ten).sum().item()
            
            if conMat != None:
                conMat.InputData(np.array(lab_ten.cpu()), np.array(predicted.cpu()))
            
            if boolDEBUG:
                print(np.array(lab_ten.cpu()))
                print("=>", np.array(predicted.cpu()))
    Accuracy_tmp = 100 * correct / total
    print('e{}, Accuracy of the network on the {} test images: {} %'.format(epoch, total, Accuracy_tmp))
    return Accuracy_tmp
#%%
if __name__ == "__main__":
    #%%
    args = list()
    num_epochs    = 200 # 0 for TEST # 100 就開始在低點飄
    num_unfreezeTime = 200 # 80
    num_class     = 15
    batch_size    = 16 # 8:3.6GB,
    learning_rate = 0.01
    useNet        = "alexNet" # "vgg"
    type_cla      = 3 # classifier type
    num_freezeNet = (31 if useNet == "vgg" else 9) # alexNet
    num_resBlock  = 0
    
#    model_weight_folder = "./result/struct2_alexNet_e10_b16_b16_e10_ut80/"
#    model_weight_path = "%s.ckpt" %("model_b16_e200_ut80")
#    model_discription   = "b%d_e%d_ut%d%s"%(batch_size, num_epochs, num_unfreezeTime, "TEST") 
    model_weight_folder = None
    model_weight_path   = None
    model_discription   = "b%d_e%d_ut%d%s"%(batch_size, num_epochs, num_unfreezeTime, "noRes") 
    
    model_struct        = "struct1_%s_c%d"%(useNet, type_cla)
    
#    processUnit = 'cpu'  # 因為 RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
    processUnit = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_tmp = torch.device(processUnit)
    #%% logger
    saveFolder = "./result/{0}_e{2:0>2d}_b{3}_{1}/".format(model_struct, model_discription, num_epochs, batch_size)
    #saveFolder = "./" # for TEST
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    log = OWNLogger(logNPY=saveFolder,
                    lossName=["loss_lab", "acc_valid"])
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
    model_main = Modle_TEST(num_resBlock=num_resBlock, useNet=useNet, num_classes=num_class, type_cla=type_cla).to(device_tmp)
    
    summary(model_main, input_size=(3, 224, 224), device=processUnit) 
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    if model_weight_folder:
        print("LOAD", model_weight_folder + model_weight_path)
        model_main.load_state_dict(torch.load(model_weight_folder + model_weight_path,
                                              map_location='cpu' if processUnit == "cpu" else None))
    #%% fine tune
    for _i, parm in enumerate(model_main.parameters()):
        if _i > num_freezeNet: 
    #        parm.requires_grad = True
            break
        else:
            parm.requires_grad = False
    
#     base
#    optimizer = torch.optim.Adam(model_main.classifier.parameters(), lr=learning_rate) # 只訓練自製的分類器
    #%% LOG
    log.ShowLocalTime()
    log.UpdateProgSetting(itrMax = total_step, 
                          batch_size = batch_size, 
                          epochs = num_epochs, 
                          model_weight_folder = model_weight_folder,
                          model_weight_path = model_weight_path,
                          model_discription = model_discription,
                          type_cla          = type_cla,
                          num_resBlock      = num_resBlock)
    #%% Loss and optimizer
    criterion = nn.functional.cross_entropy # CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model_main.parameters(), lr=learning_rate) # Adam 、SGD
    min_loss_avg = 9999
    max_valid    = 0
    #%% TRAIN
    if log != None and num_epochs != 0:
        log.SetLogTime("train")
    for epoch in tqdm.tqdm(range(num_epochs)):
        if log != None:
            log.SetLogTime("e%02d"%(epoch), boolPrint=True)
        # TRAIN
        loss_avg = Train(args, model_main, device_tmp, l_train, epoch, criterion, optimizer)
        
        if loss_avg < min_loss_avg:
            torch.save(model_main.state_dict(), '%s%s_%s_e%03d_lo%.3f.ckpt'%(saveFolder, model_struct, model_discription, epoch, loss_avg))
            min_loss_avg = loss_avg
        # VALID
        acc_tmp  = Test(args, model_main, device_tmp, l_test, epoch)
        if acc_tmp > max_valid:
            torch.save(model_main.state_dict(), '%s%s_%s_e%03d_acc%.3f.ckpt'%(saveFolder, model_struct, model_discription, epoch, acc_tmp))
            max_valid = acc_tmp
        
        # epcoh 到 可以訓練前面了
        if epoch == num_unfreezeTime:
            for _parm in model_main.parameters():
                _parm.requires_grad = True
        if log != None:    
            log.AppendLossIn("loss_lab",  loss_avg)
            log.AppendLossIn("acc_valid",  acc_tmp)
            log.SetLogTime("e%02d"%(epoch), mode = "end")
    #%% TEST
    if num_epochs == 0 or True:
        conMatOut = ConfusionMatrix(num_class=num_class)
        Test(args, model_main, device_tmp, l_test, -1, conMat=conMatOut, boolDEBUG=True);
        conMat = conMatOut.GetConfusionMatrix(boolOrg=True)
        log.dictLog["LOSS"]['conMat'] = conMat
    #%% SAVE
    if log != None and num_epochs != 0:
        log.SetLogTime("train", mode = "end")
        log.SaveLog2NPY(boolPrint=True)
    # Save the model checkpoint
    torch.save(model_main.state_dict(), '%smodel_%s_END.ckpt'%(saveFolder, model_discription))
    #%% 分析
    if num_epochs != 0:
        from utils_collect import ShowLossAnalysisFigNPY_1, CalEpochTimeCost
        ShowLossAnalysisFigNPY_1(log.logNPY, max_show = "max", x_sub=50);
        CalEpochTimeCost(log.logNPY);