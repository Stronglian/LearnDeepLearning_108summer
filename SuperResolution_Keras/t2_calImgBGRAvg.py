# -*- coding: utf-8 -*-
"""
"""
#import os
import numpy as np
from utils_collect import LoadNPY, DataLoader
LEN_BGR = 3
#%%

##shuffle
#index_shuffle = np.array([i for i in range(len(dataSet["dataset32_x"]))], dtype=np.int)
#np.random.shuffle(index_shuffle)
#%%
def CalPixelSUM(dataSet_in):
    amount_data = len(dataSet_in["dataset128_x"])
    print("len:", amount_data)
    amount_pixel = 0 #== 6648*32*32*(1+4+16)
    log_BGR = np.zeros((len([_i for _i in dataSet_in.keys() if _i.split("_")[-1] != "x"]), LEN_BGR))
    for _i, _k in enumerate(dataSet_in.keys()):
        if _k.split("_")[-1] != "x":
            continue
        print(_i//2, _k, dataSet_in[_k].shape)
        tmp_amount = 1
        for _s in dataSet_in[_k].shape[:-1]:
            tmp_amount *= _s
        amount_pixel += tmp_amount
        print(tmp_amount, "=>", amount_pixel)
        for _j in range(LEN_BGR):
            log_BGR[_i//2, _j] = np.sum(dataSet_in[_k][:, :, :, _j])
    sum_BGR = np.sum(log_BGR, axis = 0)/amount_pixel
    return sum_BGR
#%% 
if __name__ == "__main__":
    loader = DataLoader(dataFolder="./datasetNPY/")
    dataSet = loader.dataSet
    
    #dataFolder = "./datasetNPY/"
    #subfolderList = os.listdir(dataFolder)
    #dataSet = dict()
    #for _n in subfolderList:
    #    tmpDict = LoadNPY(dataFolder+_n)
    #    dataSet.update(tmpDict)
    #del tmpDict
    
    sum_BGR = CalPixelSUM(dataSet)
    sum_BGR = np.round(sum_BGR, 4)
    print("sum_BGR:", end=" ") 
    print(*sum_BGR, sep = ", ")
    