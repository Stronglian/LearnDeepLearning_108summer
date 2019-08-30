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

#%% Time
"""
https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/369869/
https://www.programiz.com/python-programming/time
"""
import time
class OWNLogger:
    def __init__(self, logNPY = None):
        self.dictLog = dict()
        if logNPY is None:
            self.logNPY = "./log_from%s.npy"%(self.ShowLocalTime())
        elif logNPY[-1] == '/':
            self.logNPY = "%slog_from%s.npy"%(logNPY, self.ShowLocalTime())
        return
    
    def __del__(self):
        self.SaveLog2NPY()
        return
    
    def SetLogTime(self, tag, mode = "start"):
        """
        mode:
            "start"
            "end"
        tag
        """
        self.dictLog[tag+"_"+mode] = time.time()
        if mode == "end":
            if tag+"_start" not in list(self.dictLog.keys()):
                raise ValueError("%s_start not in LOG"%(tag))
            print("%s, It cost %.5f sec."%(tag, self.dictLog[tag+"_end"] - self.dictLog[tag+"_start"]))
        return
    def ShowDateTime(self, intput_time_struct):
        print(time.strftime("%Y-%m-%d %H:%M:%S", intput_time_struct))
        return time.strftime("%Y-%m-%d %H:%M:%S", intput_time_struct)
    def ShowLocalTime(self):
        return self.ShowDateTime(time.localtime())
    def SaveLog2NPY(self):
        np.save(self.logNPY, self.dictLog)
        return
    def LoadLog(self, nameNPY):
        if len(list(self.dictLog.keys())) == 0:
            self.dictLog = np.load(nameNPY, allow_pickle=True).item()
        return
    def ShowAllLog(self):
        print("Dict", self.logNPY)
        for _key in list(self.dictLog.keys()):
            print(_key, time.ctime(self.dictLog[_key]))
        return
#%%
    
if __name__ == "__main__":
    t = OWNLogger()
    t.ShowLocalTime()
    pass