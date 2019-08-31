# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import json
import os
#from PIL im
#%%
#def show_photo(img):
#    plt.figure(figsize=(15, 15))
#    for i in range(1,8):
#        image = img[i,:,:]
#        plt.subplot(8, 8, i)
#        plt.axis('off')
#        plt.imshow(image)
#    plt.show()
#    return
def show_result(img_list):
    img_list = img_list.clip(0, 255).astype(np.int)
    for _i in range(len(img_list)):
        plt.imshow(img_list[_i,:,:,::-1])
#        plt.imshow(img_list[_i])
        plt.axis('off')
        plt.show()
    return

def show_result_row(img_list):
    """
    還要再處理輸出照片大小
    """
    img_list = img_list.clip(0, 255).astype(np.int)
    img_out = img_list[0,:,:,::-1].copy()
    for _i in range(1, len(img_list)):
        img_tmp = img_list[_i, :, :, ::-1].copy()
        img_out = np.concatenate((img_out, img_tmp), axis=1)
    plt.imshow(img_out)
    plt.axis('off')
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

#%% Time
"""
https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/369869/
https://www.programiz.com/python-programming/time
"""
import time
#import pytz
#us = pytz.timezone('US/Pacific')
class OWNLogger:
    def __init__(self, logNPY = None, lossName = list()):
        # Dict
        self.dictLog = dict()
        self.dictLog["time"] = dict()
        self.dictLog["loss"] = dict()
        for _l in lossName:
            self.dictLog["loss"][_l] = list()
        # SAVE
        if logNPY is None:
            self.logNPY = "./log_from%s.npy"%(self.ShowLocalTime())
        elif logNPY[-1] == '/':
            self.logNPY = "%slog_from%s.npy"%(logNPY, self.ShowLocalTime())
        return
    
    def __del__(self):
        self.SaveLog2NPY()
        return
    # SAVE
    def SaveLog2NPY(self):
        np.save(self.logNPY, self.dictLog)
        return
    def LoadLog(self, nameNPY):
        if len(list(self.dictLog.keys())) == 0:
            self.dictLog = np.load(nameNPY, allow_pickle=True).item()
        return
    def ShowAllLog(self):
        print("Dict:", self.logNPY)
        for _key in list(self.dictLog.keys()):
            for _key2 in list(self.dictLog[_key].keys()):
                print(_key, _key2, time.ctime(self.dictLog[_key][_key2]))
        return
    # TIME
    def SetLogTime(self, tag, mode = "start"):
        """
        mode:
            "start"
            "end"
        tag
        """
        self.dictLog["time"][tag+"_"+mode] = time.time()
        if mode == "end":
            if tag+"_start" not in list(self.dictLog["time"].keys()):
                raise ValueError("%s_start not in LOG"%(tag))
            print("%s, It cost %.5f sec."%(tag, self.dictLog["time"][tag+"_end"] - self.dictLog["time"][tag+"_start"]))
        return
    def ShowDateTime(self, intput_time_struct):
        print(time.strftime("%Y-%m-%d %H:%M:%S", intput_time_struct))
        return time.strftime("%Y-%m-%d %H:%M:%S", intput_time_struct)
    def ShowLocalTime(self):
        return self.ShowDateTime(time.localtime())
    # LoSS
    def AppendLossIn(self, lossName, lossValue):
        if lossName not in list(self.dictLog["loss"].keys()):
            raise ValueError("%s not in loss list"%(lossName))
        self.dictLog["loss"][lossName].append(lossValue)
        return
    def ShowLineChart(self, lossName):
        """折線圖顯示
        """
        pass
#%%
    
if __name__ == "__main__":
#    t = OWNLogger()
#    t.ShowLocalTime()
    plt.axis('off')
    plt.show()
    
    pass