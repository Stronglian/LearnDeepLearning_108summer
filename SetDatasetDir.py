# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:41:04 2019
import zipfile
https://stackoverflow.com/questions/3451111/unzipping-files-in-python
"""

import os 
import zipfile
#try:
#    import zipfile
#except:
#    print("TRY install zipfile~")
#    try:
#        %conda install zipfile
#    except:
#        %pip install zipfile
#finally:
#    import zipfile
    
dictFileFolder = {"forK":"dataset_for_keras.zip",
                  "forT":"dataset_for_torch.zip"}

folderFile = "./"

for _name in list(dictFileFolder.values()):
    if  not (os.path.exists(folderFile + _name)):
        print(folderFile + _name, "not exists.")
        
#CWD = os.getcwd().replace("\\", "/")

folderDataset = "_DataSet/"

for _key in list(dictFileFolder.keys()):
    try:
        os.makedirs(folderDataset+_key)
    except:
        print(_key, "is exists.")
    with zipfile.ZipFile(folderFile + dictFileFolder[_key], "r") as zip_ref:
        print("START unzip", dictFileFolder[_key])
        zip_ref.extractall(folderDataset + _key)
        print("END")
    
