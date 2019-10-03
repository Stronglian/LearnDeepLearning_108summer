# -*- coding: utf-8 -*-
"""
確認資料集
AwA2
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils_collect import show_result_row, SaveNPY
#%%


#%% view data
folderData = "../_DataSet/forP/"
dictData = {"test":{}, "train":{}} # dict()
for _n in os.listdir(folderData):
    t_type = _n.split("_", 1)[0]
    d_type = _n.split("_", 1)[1].split(".", 1)[0]
    print(_n, t_type, d_type, sep = ", ")
    if t_type == "train":
        if d_type == "image":    # RGB 圖像
            trainImg = np.load(folderData+_n)
        if d_type == "label":     # 15類，所以 0~14
            trainLab = np.load(folderData+_n)
        if d_type == "attribute": # 0.0~1.0
            trainAtt = np.load(folderData+_n) 
    dictData[t_type][d_type] = np.load(folderData+_n)

#%%
tmp_i = 10
show_result_row(trainImg[tmp_i:tmp_i+5])

#%%

tmp_dict_test = dictData["test"]

list_la_pre = []
class_count = -1
for _i in range(len(tmp_dict_test["label"])):
    tmp_la = tmp_dict_test["label"][_i]
    if tmp_la in list_la_pre:
        continue
    list_la_pre.append(tmp_la)
    class_count += 1
    print("class:%2d, i:%3d =>"%(class_count, _i), flush=True)
    show_result_row(tmp_dict_test["image"][_i:_i+2]);