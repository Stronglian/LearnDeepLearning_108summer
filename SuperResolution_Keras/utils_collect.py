# -*- coding: utf-8 -*-

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

def show_val_info(strOut, listValue):
    print(strOut, "avg:", np.average(listValue), "max:", np.max(listValue), "min:", np.min(listValue))
#%%    
def LoadJSON(nameJSON):#, nameDict):
    #讀取
    try:
        with open(nameJSON, 'r') as inputfile:
            nameDict = json.load(inputfile)
    except FileNotFoundError:
        nameDict = dict() #{name:{filename:{"id":,"date":},},}
    return nameDict
#%% data - basic
def DumpJSON(nameJSON, nameDict):
    with open(nameJSON, 'w') as outfile:
        json.dump(nameDict, outfile)
    return

def LoadNPY(nameNPY, shape = None):
    try:
#        with np.load(nameNPY) as inputfile:
#            nameArr = inputfile.copy()
        nameArr = np.load(nameNPY, allow_pickle = True).item()
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
    return dict_input[dict_key][index_shuffle[batch_index : batch_index+batch_size], :, :].astype(dtype)

class DataLoader:
    def __init__(self, dataFolder, batch_size):
        """
        當作資料夾裡面是乾淨(只有需要)的東西的來做
        """
#        dataFolder = "./datasetNPY/"
        self.dataFolder = dataFolder
        self.UpdateDataset(dataFolder);
        # shuffle
        self.ShuffleIndex(index_shuffle = None, boolReset = True);
#        self.ShuffleIndex();
        # 
        self.batch_size = batch_size
        # 分配 valid，看是要指定，還是直接取隨機
        return
    def UpdateDataset(self, dataFolder, boolNew = True):
        subfolderList = os.listdir(dataFolder)
        if boolNew:
            self.dataSet = dict()
        for _n in subfolderList:
            tmpDict = LoadNPY(dataFolder+_n)
            self.dataSet.update(tmpDict)
        del tmpDict
        return
    def ShuffleIndex(self, index_shuffle = None, boolReset = False):
        """
        沒參數直接 call 就是 重新 shuffle 要數列而已。
        """
        if boolReset:
            self.index_shuffle = np.array([i for i in range(len(self.dataSet[list(self.dataSet.keys())[0]]))], dtype=np.int)
        if index_shuffle is None:
            index_shuffle = self.index_shuffle
        np.random.shuffle(index_shuffle);
        return
#    def ShuffleIndex(self, boolReset = False):
#        if boolReset:
#            self.index_shuffle = np.array([i for i in range(len(self.dataSet[list(self.dataSet.keys())[0]]))], dtype=np.int)
#        np.random.shuffle(self.index_shuffle);
#        return
    def GetData(self, dict_key,  batch_index, batch_size = None, dtype = np.float, ctype = None):
        """
        要回傳特定的量與類型
        """
        batch_size = batch_size if batch_size else self.batch_size
        if ctype == "remaining": # 資料分類
            class_split = self.index_shuffle[batch_index : ]
        else:
            """
            都沒給的話
            """
            class_split = self.index_shuffle[batch_index : batch_index + batch_size]
        return self.dataSet[dict_key][class_split, :, :].astype(dtype)
    def GetLen(self, d_type = "train"):
        """
        獲取指定資料數量
        """
        pass
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
    def SetLogTime(self, tag, mode = "start"):
        """
        mode:
            "start"
            "end"
        tag
        """
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
    def AppendLossIn(self, lossName, lossValue):
        if lossName not in list(self.dictLog["LOSS"].keys()):
            raise ValueError("%s not in loss list"%(lossName))
        self.dictLog["LOSS"][lossName].append(lossValue)
        return
    def ShowLineChart(self, lossName):
        """折線圖顯示
        """
        pass
#%%
    
if __name__ == "__main__" and False :
#    t = OWNLogger()
#    t.ShowLocalTime()
#    plt.axis('off')
#    plt.show()
    strNPYname = "./w3_NPY/log_from2019-08-31 16_10_45.npy"
    
    def plotData(plt, data):
      x = [p[0] for p in data]
      y = [p[1] for p in data]
      plt.plot(x, y, '-o')
  
    tmpLogger = OWNLogger() #LOAD 不進來!???
    tmpLogger.LoadLog(strNPYname, boolForce=True)
    tmp_dictLog = tmpLogger.dictLog
#    tmp_dictLog = np.load(strNPYname, allow_pickle=True).item()
    
    MAX_SHOW_ALL = 2000
    MAX_SHOW_MIN = 200
    def S_Clip(in_list, MAX_SHOW):
        if max(in_list) > MAX_SHOW:
            in_list = np.clip(in_list, 0, MAX_SHOW)
        return in_list
    for _i, _n_loss in enumerate(tmp_dictLog["LOSS"].keys()):
        if _i != 2:
            continue
        loss_amount = len(tmp_dictLog["LOSS"][_n_loss])
        e_list = [_i for _i in range(loss_amount)]
        loss_list = tmp_dictLog["LOSS"][_n_loss].copy()
        # show value
        max_loss = np.max(loss_list)
        min_loss = np.min(loss_list)
        print("%s, len:%d, max:%d, min:%d"%(_n_loss, loss_amount, max_loss, min_loss))
        # mack max/min list
        max_list = []
        max_num = loss_list[0]
        min_list = []
        min_num = loss_list[0]
        for _l in loss_list:
            if _l >= max_num:
                max_num = _l
            if _l <= min_num:
                min_num = _l
            max_list.append(max_num)
            min_list.append(min_num)
            
        # clip
        if max_loss > MAX_SHOW_ALL:
            loss_list = np.clip(loss_list, 0, MAX_SHOW_ALL)
        max_list = S_Clip(max_list, 1000000)
        min_list = S_Clip(min_list, MAX_SHOW_MIN)
        # show plt
        plt.plot(e_list, min_list)#, linewidth=2.5)#, "-o")
        plt.show()
#        break
    
    
    pass