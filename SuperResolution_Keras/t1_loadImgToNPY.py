# -*- coding: utf-8 -*-

import cv2
import os 
import numpy as np

from utils_collect import SaveNPY

dataFolder = "D:/UPLOAD/hao/super_resolution_keras/dataset/"
subfolderList = os.listdir(dataFolder)

for _i in range(len(subfolderList)):
    if subfolderList[_i][0] != "d":
        subfolderList.pop(_i)

saveFolder = "./datasetNPY/"
try:
    os.mkdir(saveFolder)
    print(saveFolder, "已建立")
except:
    print(saveFolder, "已存在")

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
    SaveNPY(saveFolder + subfolder+".npy", dictTmp)
    
    #show
    print(subfolder, "DONE")