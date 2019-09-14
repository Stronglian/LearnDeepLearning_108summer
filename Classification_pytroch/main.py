# -*- coding: utf-8 -*-
"""
https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py
"""

#%%
import torch 
#from torch.nn import Module
#from torch.nn import functional as F
#from torch import optim
#from torch.optim.lr_scheduler import StepLR
#import torchvision.models as models 
import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from model_collect import Modle_TEST, Dataset_TEST
from torchsummary import summary
#%%
batch_size = 8
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 8
learning_rate = 0.001

processUnit = 'cpu'  # 因為 RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
#processUnit = 'cuda' if torch.cuda.is_available() else 'cpu'
device_tmp = torch.device(processUnit)

#%%

#%% model
model_main = Modle_TEST(num_resBlock=1).to(device_tmp)

summary(model_main, input_size=(3, 224, 224), device=processUnit) 
#%% DATA LOAD
d_train = Dataset_TEST("train")
l_train = DataLoader(dataset=d_train, 
                     batch_size=batch_size,
                     shuffle=True)

d_test  = Dataset_TEST("test")
l_test  = DataLoader(dataset=d_test, 
                     batch_size=batch_size,
                     shuffle=False)

#%% Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_main.parameters(), lr=learning_rate)  
#%% Train the model
total_step = len(l_train)
for epoch in tqdm.tqdm(range(num_epochs)):
    for _i, (img, lab, attr) in enumerate(l_train):
        img_ten  = (img/ 255.0).float().to(device_tmp) 
        lab_ten  = lab.long().to(device_tmp)
        attr_ten = attr.float().to(device_tmp)
        
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
#%% Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in l_test:
        images = images.reshape(-1, 28*28).to(device_tmp)
        labels = labels.to(device_tmp)
        outputs = model_main(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model_main.state_dict(), 'model.ckpt')