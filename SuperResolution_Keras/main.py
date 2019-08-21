# -*- coding: utf-8 -*-

from utils_collect import LoadNPY
import os 
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%% keras 設定
# 自動增長 GPU 記憶體用量
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# 設定 Keras 使用的 Session
tf.keras.backend.set_session(sess)

#%%
#from keras import backend as K
from keras.models import Model#, Sequential
from keras.layers import Conv2DTranspose, Input, Conv2D #, Dense,  Flatten, Activation, MaxPooling2D

#%%
from keras.applications.vgg16 import VGG16

#%% FLOW CONTROL
INT_FLOW_CONTROL = [1]
DICT_FLOW_NAME = {1:"載入資料庫", 
                  2:"建構",
                  3:"訓練",
                  4:"儲存模型",
                  5:"載入"}

#%%
dataFolder = "./datasetNPY/"
subfolderList = os.listdir(dataFolder)
#os.listdir(dataFolder+subfolderList[0])
modelSaveFolder = "./save/weight/"

#%% LOAD DATASET
dataSet = dict()
for _n in subfolderList:
    tmpDict = LoadNPY(dataFolder+_n)
    dataSet.update(tmpDict)
#shuffle
index_shuffle = np.array([i for i in range(len(dataSet["dataset32_x"]))], dtype=np.int)
index_shuffle = np.random.shuffle(index_shuffle)
#
def GetData(dict_input, dict_key,  batch_index, batch_size, index_shuffle, dtype = np.float):
#    batch_data  = dict_input[dict_key][index_shuffle[batch_index : batch_index+batch_size,:,:]].astype(np.float)
#    return batch_data
    return dict_input[dict_key][index_shuffle[batch_index : batch_index+batch_size,:,:]].astype(dtype)
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
    layer_output = Conv2DTranspose(3,(3,3),strides=(1, 1), padding='same', activation='sigmoid')(layer_)
    
    # construct the autoencoder model
    outputModel = Model(inputs=input_img, outputs=layer_output)
    return outputModel

#%%
k = 32
model1 = MakeModel_TEST(shape=(k, k, 3))
model1.summary()

#k = 64
#model2 = MakeModel_TEST(shape=(k, k, 3))
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
lossModel = Model(lossModel.inputs, selectedOutputs)
lossModel.name = "lossModel_VGG"
#%% LOSS SET - LINK
## model 1
#lossModelOutputs1 = lossModel(model1.output)
#
#partModel_1 = Model(model1.input, lossModelOutputs1)
#partModel_1.name = "m1_32to64"
#
## with model
#partModel_1.compile('adam',loss='mse')

## model 2
#lossModelOutputs2 = lossModel(model2.output)
#partModel_2 = Model(model2.input, lossModelOutputs2)
#partModel_2.name = "m2_64to128"
#partModel_2.compile('adam',loss='mse')
#
## model 3
#
##fullModel   = 
model1.compile('adam',loss='mse')
#%% train parm set
epochs = 1
batch_size = 8 #if 32 : 4G VRAM 不足，16 頂
itr = int(len(dataSet["dataset32_x"])//batch_size) #207.75

#%% TRAIN
for epoch in range(epochs):
    batch_index = 0
    for step in range(itr): #936
        batch_in  = dataSet["dataset32_x"][batch_index : batch_index+batch_size,:,:].astype(np.float)/255.0
        batch_mid = dataSet["dataset64_x"][batch_index : batch_index+batch_size,:,:].astype(np.float)/255.0
#        batch_out = dataSet["dataset128_x"][batch_index : batch_index+batch_size,:,:].astype(np.float)/255.0
#        batch_in = GetData(dataSet, "dataset32_x",  batch_index, batch_size, index_shuffle)
#        batch_mid = GetData(dataSet, "dataset64_x",  batch_index, batch_size, index_shuffle)
#        batch_out = GetData(dataSet, "dataset128_x",  batch_index, batch_size, index_shuffle)
        batch_index += batch_size
        
#        batch_lossModel1 = lossModel.predict(batch_mid)
#        batch_lossModel2 = lossModel.predict(batch_out)
        
#        loss1 = partModel_1.train_on_batch(batch_in, batch_lossModel1)
#        loss2 = partModel_2.train_on_batch(batch_mid, batch_lossModel2)
        loss1 = model1.train_on_batch(batch_in, batch_mid)
#        
#        loss3 = partModel_2.train_on_batch(partModel_1.predict(batch_in), batch_lossModel2)
        
#        if step%100 == 0 :
##            print('itr:',step,' total_loss:', loss[0], ' loss:',loss[1:])
#            print('itr:%4d, \n1- total_loss:%7.4f loss:'%(step, loss1[0]), loss1[1:])
##            print('2- total_loss:%7.4f loss:'%(loss2[0]), loss2[1:])
##            print('3- total_loss:%7.4f loss:'%(loss3[0]), loss3[1:])
    print('==========epcohs:',epoch,' loss:', loss1)
    
#%% SAVE MODEL
#partModel_1.save_weights(modelSaveFolder + 'only1T_%s_e%d_b%d.h5'%(partModel_1.name, epochs, batch_size))
#partModel_2.save_weights(partModel_2.name+'.h5')
#%% USE
#predict1 = partModel_1.predict(dataSet["dataset32_x"][:5,:])
#predict1_img = predict1[0]#.reshape(5, 64, 64, 3)
##predict2 = partModel_1.predict(predict1_img)
##predict2_img = predict2[0]#.reshape(5, 64, 64, 3)
predict1 = model1.predict(dataSet["dataset32_x"][:5,:])

