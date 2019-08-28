# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import json
import os
#from PIL im
#%%
def show_photo(img):
    plt.figure(figsize=(15, 15))
    for i in range(1,8):
        image = img[i,:,:]
        plt.subplot(8, 8, i)
        plt.axis('off')
        plt.imshow(image)
    plt.show()
    return

#%%
    
def LoadJSON(nameJSON):#, nameDict):
    #讀取
    try:
        with open(nameJSON, 'r') as inputfile:
            nameDict = json.load(inputfile)
    except FileNotFoundError:
        nameDict = dict() #{name:{filename:{"id":,"date":},},}
    return nameDict
#%%
def DumpJSON(nameJSON, nameDict):
    with open(nameJSON, 'w') as outfile:
        json.dump(nameDict, outfile)
    return

def LoadNPY(nameNPY, shape = None):
    try:
#        with np.load(nameNPY) as inputfile:
#            nameArr = inputfile.copy()
        nameArr = np.load(nameNPY, allow_pickle=True).item()
    except FileNotFoundError:
        nameArr = np.zeros(shape)
    return nameArr

def SaveNPY(nameNPY, nameArr):
    np.save(nameNPY, nameArr)
    return

#%% LOAD DATA
    
def GetData(dict_input, dict_key,  batch_index, batch_size, index_shuffle, dtype = np.float):
#    batch_data  = dict_input[dict_key][index_shuffle[batch_index : batch_index+batch_size,:,:]].astype(np.float)
#    return batch_data
    return dict_input[dict_key][index_shuffle[batch_index : batch_index+batch_size,:,:]].astype(dtype)

#%%
    
if __name__ == "__main":
    pass