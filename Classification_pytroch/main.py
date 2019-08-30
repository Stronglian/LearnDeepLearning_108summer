# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:46:11 2019

@author: str85
"""

#import torch
import os
import numpy as np

folderData = "../_DataSet/forT/"
dictData = dict()
for _n in os.listdir(folderData):
    t_type = _n.split("_", 1)[0]
    d_type = _n.split("_", 1)[1].split(".", 1)[0]
    print(_n, t_type, d_type, sep = ", ")
    if t_type == "train":
        if d_type == "image":
            trainImg = np.load(folderData+_n)
#    dictData[t_type][d_type] = np.load(folderData+_n)