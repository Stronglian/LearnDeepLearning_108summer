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
            
show_result_row(trainImg[1150:1155])
