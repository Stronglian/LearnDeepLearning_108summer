# -*- coding: utf-8 -*-

from utils_collect import LoadNPY
import os 
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%%
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Flatten, Conv2D, Activation, MaxPooling2D, Conv2DTranspose

#%%
from keras.applications.vgg16 import VGG16

#%% FLOW CONTROL
INT_FLOW_CONTROL = 1
DICT_FLOW_NAME = {1:"載入資料庫", 
                  2:"",
                  3:""}

#%%
dataFolder = "./datasetNPY/"
subfolderList = os.listdir(dataFolder)
#os.listdir(dataFolder+subfolderList[0])


#%% LOAD DATASET
dataSet = dict()
for _n in subfolderList:
    tmpDict = LoadNPY(dataFolder+_n)
    dataSet.update(tmpDict)

#%% MODEL
# mainModel
def MakeModel_TEST(shape=(64,64,3)):
    # 忘了怎架 R block 果斷放棄
    input_img = Input(shape)
    
    # encoder layers
#    encoded = Conv2D(3,(3,3),strides=(1, 1), padding='valid')(input_img)
#    encoded = Conv2D(16,(3,3),strides=(1, 1), padding='valid')(encoded)
#    encoded = Conv2D(32,(3,3),strides=(1, 1), padding='valid')(encoded)
#    encoded = Conv2D(64,(3,3),strides=(1, 1), padding='valid')(encoded)
#    encoded = Conv2D(128,(3,3),strides=(1, 1), padding='valid')(encoded)
#    encoded_output = Conv2D(256,(3,3),strides=(1, 1), padding='valid')(encoded)
##    
#    # decoder layers
#    decoded = Conv2DTranspose(256,(3,3),strides=(1, 1), padding='valid')(encoded_output)
#    decoded = Conv2DTranspose(128,(3,3),strides=(1, 1), padding='valid')(decoded)
#    decoded = Conv2DTranspose(64,(3,3),strides=(1, 1), padding='valid')(decoded)
#    decoded = Conv2DTranspose(32,(3,3),strides=(1, 1), padding='valid')(decoded)
#    decoded = Conv2DTranspose(16,(3,3),strides=(1, 1), padding='valid')(decoded)
#    decoded_output = Conv2DTranspose(3,(3,3),strides=(1, 1), padding='valid')(decoded)
    
#    # decoder layers
#    decoded = Conv2DTranspose(256,(5,5),strides=(1, 1), padding='valid')(input_img)
#    decoded = Conv2DTranspose(128,(3,3),strides=(2, 2), padding='valid')(decoded)
##    decoded = Conv2DTranspose(128,(5,5),strides=(1, 1), padding='valid')(decoded)
##    decoded = Conv2DTranspose(64,(3,3),strides=(1, 1), padding='valid')(decoded)
###    decoded = Conv2DTranspose(64,(3,3),strides=(2, 2), padding='valid')(decoded)
##    decoded = Conv2DTranspose(64,(3,3),strides=(1, 1), padding='valid')(decoded)
##    decoded = Conv2DTranspose(16,(3,3),strides=(1, 1), padding='valid')(decoded)
    
    layer_ = Conv2DTranspose(512,(3,3), strides=(2, 2), padding='same')(input_img)
    layer_ = Conv2D(256, (3,3), strides=(1, 1), padding='same')(layer_)
    layer_ = Conv2DTranspose(512,(3,3), strides=(2, 2), padding='same')(layer_)
    layer_ = Conv2D(32, (3,3), strides=(2, 2), padding='same')(layer_)
    layer_output = Conv2DTranspose(3,(3,3),strides=(1, 1), padding='same')(layer_)
    
    # construct the autoencoder model
    outputModel = Model(inputs=input_img, outputs=layer_output)
    return outputModel

#%%
k = 32
model1 = MakeModel_TEST(shape=(k, k, 3))
#k = 64
#model2 = MakeModel_TEST(shape=(k, k, 3))
model1.summary()
#print("==="*10)
#model2.summary()

#%% LOSS SET - base
# 載
lossModel = VGG16(weights='imagenet', include_top=False)
lossModel.trainable=False
for layer in lossModel.layers:
    layer.trainable=False
# 設置
selectedLayers = [1,2,9,10,17,18] 
selectedOutputs = [lossModel.layers[i].output for i in selectedLayers]
# 建立
lossModel = Model(lossModel.inputs,selectedOutputs)
lossModel.name = "lossModel_VGG"
#%% LOSS SET - LINK

lossModelOutputs1 = lossModel(model1.output)

partModel_1 = Model(model1.input, lossModelOutputs1)
partModel_1.name = "m1_32to64"

# with model
partModel_1.compile('adam',loss='mse')

#lossModelOutputs2 = lossModel(model2.output)
#partModel_2 = Model(model2.input, lossModelOutputs2)
#partModel_2.name = "m2_64to128"
#partModel_2.compile('adam',loss='mse')

#fullModel   = 
#%% train parm set
epochs = 1
itr = 936 #936
batch_size = 32

#%% TRAIN
for epoch in range(epochs):
    batch_index = 0
    for step in range(itr): #936
        batch_in  = dataSet["dataset32_x"][batch_index : batch_index+batch_size,:,:].astype(np.float)
        batch_out = dataSet["dataset64_x"][batch_index : batch_index+batch_size,:,:].astype(np.float)
        batch_index += batch_size
        
#        triple_batch = np.concatenate((batch,batch,batch),axis=-1)
        batch_lossModel = lossModel.predict(batch_out)
        
        loss = partModel_1.train_on_batch(batch_in, batch_lossModel)
        
        if step%100 == 0 :
            print('itr:',step,' total_loss:', loss[0], ' loss:',loss[1:])
    print('==========epcohs:',epoch,' loss:', loss)
    
#%% SAVE MODEL

