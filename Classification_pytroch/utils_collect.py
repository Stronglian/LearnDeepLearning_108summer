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

def show_val_info(strOut, listValue, boolReturnDict = False, boolPrint = True):
    val_len = len(listValue)
    val_max = np.max(listValue)
    val_min = np.min(listValue)
    val_max_arg = np.argmax(listValue)
    val_min_arg = np.argmin(listValue)
    val_avg = np.average(listValue)
#    print(strOut, "len:", len(listValue), "avg:", np.average(listValue), "max:", np.max(listValue), "min:", np.min(listValue))    
    if boolPrint:
        print("%s: len:%d, avg:%.5f, max:%.2f, min:%.5f, arg_max:%d, arg_min:%d"%(strOut, val_len, val_avg, val_max, val_min, val_max_arg, val_min_arg))
    if boolReturnDict:
        return {"len":val_len, "avg":val_avg, "max":val_max, "min":val_min, "arg_max":val_max_arg, "arg_mi:":val_min_arg}
    
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
        return time.strftime("%Y-%m-%d %H_%M_%S", intput_time_struct)
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
#%% LOSS FIG
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

def ShowFig(x_list, input_list, max_show = None, boolClip = True, x_sub=10,
            strShowSaveTitle = "TMP", boolSave = False, strSaveFolder = "./"):
    if max_show == None:
        max_show = np.average(input_list)
    if boolClip:
        input_list = S_Clip(input_list, max_show)
    plt.title("%s (limit: %.2f)"%(strShowSaveTitle, max_show))
    plt.xticks(np.arange(0, len(x_list)+1, x_sub))
    plt.plot(x_list, input_list)#, linewidth=2.5)#, "-o")
    if boolSave:
        plt.savefig("%s%s.jpg"%(strSaveFolder, strShowSaveTitle))
    plt.show()
    return

def ShowValMaxMinFig(x_list, in_list, strLossName, max_show = None, boolSave = False, 
                     boolDictShow = {"val":True, "max":False, "min":True}, **darg):
    ### mack max/min list
    max_list, min_list = MakeMaxMinList(in_list)
    ### 顯示 # 需要浮動限制?
    if max_show:
        if max_show == "max":
            max_show = np.max(in_list)
        elif max_show == "avg":
            max_show = np.average(in_list)
        else:
            raise ValueError("TESTING")
    else:
        max_show = np.max(in_list)
    try:
        if boolDictShow["val"]:
            ShowFig(x_list, in_list, max_show = max_show, 
                    strShowSaveTitle = "%s_all"%(strLossName), boolSave = boolSave, **darg)
    except KeyError:
        pass
    
    try:
        if boolDictShow["max"]:
            ShowFig(x_list, max_list,  max_show = max_show, 
                    strShowSaveTitle = "%s_max"%(strLossName), boolSave = boolSave, **darg)
    except KeyError:
        pass
    
    try:
        if boolDictShow["min"]:
            ShowFig(x_list, min_list, max_show = max_show, 
                    strShowSaveTitle = "%s_min"%(strLossName), boolSave = boolSave, **darg)
    except KeyError:
        pass
    
    return

def ShowLossAnalysisFigNPY_1(strNPYname, boolSave = False, LOSS = "LOSS", 
                           type_list = ["avg"], **darg):
    """
    AMOUNT_LOSS_NUM: 有幾個結果
    簡化成一、兩張圖
    """
    ## LOAD NPY
    if os.path.exists(strNPYname):
        print("LOAD", strNPYname)
    else:
        raise IOError(strNPYname, "not exists")
    tmp_dictLog = np.load(strNPYname, allow_pickle=True).item()
    tmp_dictLog = tmp_dictLog[LOSS] # 只取用需要的
#    print("key:", tmp_dictLog.keys())
    ## SHOW
#    for _i, _n_loss in enumerate(np.sort(list(tmp_dictLog.keys()))): #
    for _i, _n_loss in enumerate(tmp_dictLog.keys()): #
        print(_i, _n_loss, "==="*20)
        ### info 
        loss_amount = len(tmp_dictLog[_n_loss])    # 資料數量
        x_list = [_j for _j in range(loss_amount)] # 橫軸
#        if _i // AMOUNT_LOSS_NUM == 2 or True: # LOSS
        print("LOSS"); 
        loss_list = tmp_dictLog[_n_loss].copy()    # 主要資料
        ### show info
        show_val_info(_n_loss, loss_list);
        ShowValMaxMinFig(x_list, loss_list, _n_loss,  boolSave = boolSave,
                         boolDictShow = {"val":True, "max":False, "min":True},
                         **darg);
    return

def CalEpochTimeCost(strNPYname, boolCalAll = False):
    ## LOAD NPY
    if os.path.exists(strNPYname):
        print("LOAD", strNPYname)
    else:
        raise IOError(strNPYname, "not exists")
    tmp_dictLog = np.load(strNPYname, allow_pickle=True).item()
    ## PICK
    tmp_dictLog = tmp_dictLog["TIME"]
    tmp_dictLog_key = list(tmp_dictLog.keys())
#    print("key:", tmp_dictLog.keys())
    tag_list = np.unique([_t.rsplit("_", 1)[0] for _t in tmp_dictLog_key])
#    print("tag_list:", tag_list)
    ## CAL DIFF
    dictTimeCost = {"Epoch":[], "Valid":[], "Total":0}
    for _tag in tag_list:
        if (("train" in _tag) or ("Valid" in _tag)) and not boolCalAll:
            continue
        se_list = [_t for _t in tmp_dictLog_key if _tag in _t]
        time_s = [_se for _se in se_list if "start" in _se]
        time_e = [_se for _se in se_list if "end"   in _se]
        time_cost = np.round(tmp_dictLog[time_e[0]] - tmp_dictLog[time_s[0]], 5)
#        print("%s cost: %.3f"%(_tag, time_cost))
        if "Valid" in _tag:
            dictTimeCost["Valid"].append(time_cost)
        elif "train" in _tag:
            dictTimeCost["Total"] = time_cost
        else: # epoch
            dictTimeCost["Epoch"].append(time_cost)
    ## CAL AVG
    time_cost_avg = np.average(dictTimeCost["Epoch"])
    print("%s avg cost: %.3f sec./epoch"%(_tag, time_cost_avg))
    return time_cost_avg, dictTimeCost
#%%
if __name__ == "__main__":
    logNPY = "./result/%s/%s.npy"%("struct2_alexNet_e100_b16_b16_e100requires_gradF", "log_from2019-09-18 06_54_08")
    ShowLossAnalysisFigNPY_1(logNPY, max_show = "avg", x_sub=10);
    CalEpochTimeCost(logNPY);