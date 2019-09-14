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
#%%
#%% Time
"""
https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/369869/
https://www.programiz.com/python-programming/time
"""
import time
#import pytz
#us = pytz.timezone('US/Pacific')
class OWNLogger:
    def __init__(self, logNPY = None, lossName = list(), dictSetting = dict()):
        # Dict
        self.dictLog = dict()
        self.dictLog["TIME"] = dict()
        self.dictLog["LOSS"] = dict()
        self.dictLog["SET"] = dictSetting
        if len(lossName) == 0:
            lossName = ["loss"]
        for _l in lossName:
            self.dictLog["LOSS"][_l] = list()
        # SAVE
        if logNPY is None:
            self.logNPY = "./log_from%s.npy"%(self.ShowLocalTime())
        elif logNPY[-1] == '/':
            self.logNPY = "%slog_from%s.npy"%(logNPY, self.ShowLocalTime())
        return
    
#    def __del__(self):
#        self.SaveLog2NPY()
#        return
    def UpdateProgSetting(self, **dictSetting):
        self.dictLog["SET"].update(dictSetting)
        return
    # SAVE
    def SaveLog2NPY(self, boolPrint = False):
        if boolPrint:
            print("SaveLog2NPY", self.logNPY)
        np.save(self.logNPY, self.dictLog)
        return
    def LoadLog(self, nameNPY, boolForce = False):
        self.logNPY = nameNPY
        if len(list(self.dictLog.keys())) == 0 or boolForce:
            self.dictLog = np.load(self.logNPY, allow_pickle=True).item()
        print("LOAD", self.logNPY)
        return
    def ShowAllLog(self):
        print("Dict:", self.logNPY)
        for _key in list(self.dictLog.keys()):
            for _key2 in list(self.dictLog[_key].keys()):
                print(_key, _key2, time.ctime(self.dictLog[_key][_key2]))
        return
    # TIME
    def SetLogTime(self, tag, mode = "start", boolPrint = False):
        """
        mode:
            "start"
            "end"
        tag:
            epoch%d
            train
        """
        if boolPrint:
            print("=>", tag, mode)
        self.dictLog["TIME"][tag+"_"+mode] = time.time()
        if mode == "end":
            if tag+"_start" not in list(self.dictLog["TIME"].keys()):
                raise ValueError("%s_start not in LOG"%(tag))
            print("%s, It cost %.5f sec."%(tag, self.dictLog["TIME"][tag+"_end"] - self.dictLog["TIME"][tag+"_start"]))
        return
    def ShowDateTime(self, intput_time_struct, boolPrint = True):
        if boolPrint:
            print(time.strftime("%Y-%m-%d %H:%M:%S", intput_time_struct))
        return time.strftime("%Y-%m-%d %H:%M:%S", intput_time_struct)
    def ShowLocalTime(self):
        return self.ShowDateTime(time.localtime())
    # LOSS
    def AppendLossIn(self, lossName = None, lossValue = None):
        if lossName is None:
            lossName = "loss"
        if lossName not in list(self.dictLog["LOSS"].keys()):
            raise ValueError("%s not in loss list"%(lossName))
        self.dictLog["LOSS"][lossName].append(lossValue)
        return
#    def ShowLineChart(self, lossName):
#        """折線圖顯示
#        """
#        pass
#%%