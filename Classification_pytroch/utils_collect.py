# -*- coding: utf-8 -*-
"""
需要在拿過來
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
#from PIL im
#%% show
#def show_photo(img):
#    plt.figure(figsize=(15, 15))
#    for i in range(1,8):
#        image = img[i,:,:]
#        plt.subplot(8, 8, i)
#        plt.axis('off')
#        plt.imshow(image)
#    plt.show()
#    return
def show_result(img_list, boolSave = False, strName = "tmp", strFolder = "./"):
    img_list = img_list.clip(0, 255).astype(np.int)
    for _i in range(len(img_list)):
        plt.imshow(img_list[_i, :, :, :])
#        plt.imshow(img_list[_i])
        plt.axis('off')
        plt.show()
    return

def show_result_row(img_list, boolSave = False, strName = "tmp", strFolder = "./"):
    """
    還要再處理輸出照片大小
    """
    img_list = img_list.clip(0, 255).astype(np.int)
    img_out = img_list[0, :, :, :].copy()
    for _i in range(1, len(img_list)):
        img_tmp = img_list[_i, :, :, :].copy()
        img_out = np.concatenate((img_out, img_tmp), axis=1)
    plt.imshow(img_out)
    plt.axis('off')
    if boolSave:
        plt.savefig("%s%s.jpg"%(strFolder, strName))
    plt.show()
    return

def show_val_info(strOut, listValue):
    print(strOut, "len:", len(listValue), "avg:", np.average(listValue), "max:", np.max(listValue), "min:", np.min(listValue))
    
#%%
def LoadJSON(nameJSON):#, nameDict):
    #讀取
    try:
        with open(nameJSON, 'r') as inputfile:
            nameDict = json.load(inputfile)
    except FileNotFoundError as e:
        print(e)
        nameDict = dict() #{name:{filename:{"id":,"date":},},}
    return nameDict

def DumpJSON(nameJSON, nameDict):
    with open(nameJSON, 'w') as outfile:
        json.dump(nameDict, outfile)
    return

def LoadNPY(nameNPY, shape = None):
    try:
#        with np.load(nameNPY) as inputfile:
#            nameArr = inputfile.copy()
        nameArr = np.load(nameNPY, allow_pickle = True).item()
    except FileNotFoundError as e:
        print(e)
        nameArr = np.zeros(shape)
    return nameArr

def SaveNPY(nameNPY, nameArr):
    np.save(nameNPY, nameArr)
    return