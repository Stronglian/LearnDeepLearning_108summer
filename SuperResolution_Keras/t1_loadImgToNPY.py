# -*- coding: utf-8 -*-

import cv2
import os 
import numpy as np

from utils_collect import SaveNPY

dataFolder = "../_DataSet/forK/"
subfolderList = os.listdir(dataFolder)

for i in range(len(subfolderList)): #i = 0
    subfolder = subfolderList[i]
    tmpFolderList = os.listdir(dataFolder + "/" + subfolder)
    imgList = list()
    imgCountList = list()
    for j in range(len(tmpFolderList)): #j = 0
        # read
        imgName = tmpFolderList[j]
        imgTmp  = cv2.imread(dataFolder + "/" + subfolder + "/" + imgName)
        imgList.append(imgTmp)
        imgCountList.append(int(imgName.split("(", 1)[1].split(")", 1)[0]))
#        if j == 20:
#            break
    # save
    dictTmp = dict()
    dictTmp[subfolder + "_x"] = np.array(imgList)
    dictTmp[subfolder + "_y"] = np.array(imgCountList)
    SaveNPY(subfolder+".npy", dictTmp)