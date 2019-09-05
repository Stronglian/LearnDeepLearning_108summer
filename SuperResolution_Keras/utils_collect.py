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
    print(strOut, "len:", len(listValue), "avg:", np.average(listValue), "max:", np.max(listValue), "min:", np.min(listValue))
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
    def __init__(self, dataFolder, batch_size, train_ratio = 0.8):
        """
        當作資料夾裡面是乾淨(只有需要)的東西的來做
        """
        self.dataFolder = dataFolder
        self.UpdateDataset(dataFolder, boolNew = True);
        # shuffle
        self.ShuffleIndex(index_shuffle = None, boolReset = True);
#        self.ShuffleIndex();
        # 
        self.batch_size  = batch_size
        self.train_ratio = train_ratio
        # 分配 valid，看是要指定，還是直接取隨機
        return
    def UpdateDataset(self, dataFolder, boolNew = False):
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
        np.random.shuffle(index_shuffle); # 會連帶
        return
#    def ShuffleIndex(self, boolReset = False):
#        if boolReset:
#            self.index_shuffle = np.array([i for i in range(len(self.dataSet[list(self.dataSet.keys())[0]]))], dtype=np.int)
#        np.random.shuffle(self.index_shuffle);
#        return
    def GetLen(self, d_key = "dataset32_y"):
        """
        獲取指定資料數量
        """
        return len(self.dataSet[d_key])
    def CalMaxIter(self):
        """
        直接算最大 ITER
        """
        self.iter_max = int( (self.GetLen()//self.batch_size) * self.train_ratio);
        return self.iter_max;
    def GetData(self, dict_key,  batch_index, batch_size = None, dtype = np.float, ctype = None):
        """
        要回傳特定的量與類型
        """
        if ctype == "remaining": # 資料分類 # 取剩下的，主要用於 valid
            class_split = self.index_shuffle[batch_index : batch_size]
        else: # None and other
            batch_size = batch_size if batch_size else self.batch_size
            class_split = self.index_shuffle[batch_index : batch_index + batch_size]
        return self.dataSet[dict_key][class_split].astype(dtype)
    # 用 ITER 跑?
    def __iter__(self): # ??
        self.batch_index = 0
        return self
    def __next__(self):
        if self.batch_index >= self.iter_max * self.batch_size: # 結束
            raise StopIteration
        else:
            batch_in  = self.GetData("dataset32_x",  self.batch_index, self.batch_size)
            batch_mid = self.GetData("dataset64_x",  self.batch_index, self.batch_size)
            batch_out = self.GetData("dataset128_x", self.batch_index, self.batch_size)
            self.batch_index += self.batch_size
            return batch_in, batch_mid, batch_out
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
    def AppendLossIn(self, lossName, lossValue):
        if lossName not in list(self.dictLog["LOSS"].keys()):
            raise ValueError("%s not in loss list"%(lossName))
        self.dictLog["LOSS"][lossName].append(lossValue)
        return
#    def ShowLineChart(self, lossName):
#        """折線圖顯示
#        """
#        pass
#%% show fig
MAX_SHOW_ALL = 2000
def MakeMaxMinList(loss_list):
    max_list = []
    min_list = []
    max_num = loss_list[0]
    min_num = loss_list[0]
    for _l in loss_list:
        if _l > max_num:
            max_num = _l
        if _l < min_num:
            min_num = _l
        max_list.append(max_num)
        min_list.append(min_num)
    return max_list, min_list

def S_Clip(in_list, max_show = 255, min_show = 0):
    if max(in_list) > max_show:
        in_list = np.clip(in_list, min_show, max_show)
    return in_list

def ShowFig(x_list, input_list, max_show = None, strShowSaveTitle = "TMP", 
            boolSave = False, strSaveFolder = "./"):
    if max_show == None:
        max_show = np.average(input_list)
    input_list = S_Clip(input_list, max_show)
    plt.title(strShowSaveTitle)
    plt.plot(x_list, input_list)#, linewidth=2.5)#, "-o")
    if boolSave:
        plt.savefig("%s%s.jpg"%(strSaveFolder, strShowSaveTitle))
    plt.show()
    return

#%%
if __name__ == "__main__":
    # load log
    LOSS = "loss"
    strNPYname = "./w3_NPY/log_from2019-08-31 16_10_45.npy"
    if not os.path.exists(strNPYname):
        raise IOError(strNPYname, "not exists")
#    tmpLogger = OWNLogger() 
#    tmpLogger.LoadLog(strNPYname, boolForce=True)
#    tmp_dictLog = tmpLogger.dictLog
    tmp_dictLog = np.load(strNPYname, allow_pickle=True).item()
    # pick data
    for _i, _n_loss in enumerate(tmp_dictLog[LOSS].keys()):
        print(_i, _n_loss)
        # info 
        loss_amount = len(tmp_dictLog[LOSS][_n_loss]) # 資料數量
        x_list = [_j for _j in range(loss_amount)]    # 橫軸
        loss_list = tmp_dictLog[LOSS][_n_loss].copy() # 主要資料
        # show value
        max_loss = np.max(loss_list)
        min_loss = np.min(loss_list)
        avg_loss = np.average(loss_list)
        print("%s, len:%d, max:%.2f, min:%.5f, avg:%.5f"%(_n_loss, loss_amount, max_loss, min_loss, avg_loss))
        show_val_info(_n_loss, loss_list)
        # mack max/min list
        max_list, min_list = MakeMaxMinList(loss_list)
        # 顯示 # 需要浮動限制?
        ShowFig(x_list, loss_list, max_show = 2000,   strShowSaveTitle = "%s_all"%(_n_loss), boolSave = False)
#        ShowFig(x_list, max_list,  max_show = 110000, strShowSaveTitle = "%s_max"%(_n_loss), boolSave = True)
        ShowFig(x_list, min_list,  max_show = 400,    strShowSaveTitle = "%s_min"%(_n_loss), boolSave = False)
        break
