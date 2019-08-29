# -*- coding: utf-8 -*-

from utils_collect import LoadNPY, GetData
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
from keras.layers import Conv2DTranspose, Input, Conv2D, Add, Lambda #, Dense,  Flatten, Activation, MaxPooling2D
from model_collect import res_block, normalize, denormalize, upsample
##%%
#from keras.applications.vgg16 import VGG16

#%% FLOW CONTROL
INT_FLOW_CONTROL = [1]
DICT_FLOW_NAME = {1:"載入資料庫", 
                  2:"",
                  3:""}

#%%
dataFolder = "./datasetNPY/"
subfolderList = os.listdir(dataFolder)
#os.listdir(dataFolder+subfolderList[0])
saveFolder = "./w/"
try:
    os.makedirs(saveFolder)
except:
    print( saveFolder, "is exsist.")
#%% LOAD DATASET
dataSet = dict()
for _n in subfolderList:
    tmpDict = LoadNPY(dataFolder+_n)
    dataSet.update(tmpDict)
#shuffle
index_shuffle = np.array([i for i in range(len(dataSet["dataset32_x"]))], dtype=np.int)
index_shuffle = np.random.shuffle(index_shuffle)
#
#%% MODEL
# mainModel
def Model_TEST(scale=2, num_filters=64, num_res_blocks=8, res_block_scaling=None, model_name = None, x_in = None): #origin (4, 64, 16, None)
    if not None:
        x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

    x = b = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)
#    x = upsample(x, scale, num_filters)
#    x = b = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return b, Model(x_in, x, name=model_name)
#    # 忘了怎架 R block 果斷放棄
#    input_img = Input(shape)
#    # https://github.com/krasserm/super-resolution/blob/master/example-edsr.ipynb
#    # https://github.com/krasserm/super-resolution/blob/master/model/edsr.py
#    # https://github.com/krasserm/super-resolution/blob/master/model/common.py
#    # construct the autoencoder model
#    outputModel = Model(inputs=input_img, outputs=layer_output)
#    return outputModel

#%%
m_branch, model1 = Model_TEST(model_name="x32-x64_model")
model1.summary()
model1.compile('adam',loss='mse')

#k = 64
_, model2 = Model_TEST(model_name="x64-x128_model", x_in=m_branch)
model2.summary()
model2.compile('adam',loss='mse')


#%% train parm set
epochs = 10
batch_size = 16 #if 32 : 4G VRAM 不足，16 頂
itr = int(len(dataSet["dataset32_x"])//batch_size) #207.75

minLoss1 = minLoss2 = 10000000
#%% TRAIN #要照她的嗎? https://github.com/krasserm/super-resolution/blob/master/train.py
for epoch in range(epochs):
    batch_index = 0
    print("epoch", epoch)
    for step in range(itr): #936
        batch_in  = dataSet["dataset32_x"][batch_index : batch_index+batch_size,:,:].astype(np.float)
        batch_mid = dataSet["dataset64_x"][batch_index : batch_index+batch_size,:,:].astype(np.float)
        batch_out = dataSet["dataset128_x"][batch_index : batch_index+batch_size,:,:].astype(np.float)
#        batch_in = GetData(dataSet, "dataset32_x",  batch_index, batch_size, index_shuffle) # 未 /255
#        batch_mid = GetData(dataSet, "dataset64_x",  batch_index, batch_size, index_shuffle)
#        batch_out = GetData(dataSet, "dataset128_x",  batch_index, batch_size, index_shuffle)
        batch_index += batch_size
        
#        
        loss1 = model1.train_on_batch(batch_in, batch_mid)
        loss2 = model2.train_on_batch(batch_in, batch_out)
        
        if step%100 == 0 :
            print("itr: %d loss1: %d, loss2: %d"%(step, loss1, loss2))
#            print('itr:',step,' total_loss:', loss[0], ' loss:',loss[1:])
#            print('itr:%4d, \n1- total_loss:%7.4f loss:'%(step, loss1[0]), loss1[1:])
#            print('2- total_loss:%7.4f loss:'%(loss2[0]), loss2[1:])
#            print('3- total_loss:%7.4f loss:'%(loss3[0]), loss3[1:])
        if loss1 < minLoss1:
            minLoss1 = loss1
            model1.save(saveFolder + '%s_e%d_b%d_lo%.5f.h5'%(model1.name, epochs, batch_size, loss1))
        if loss2 < minLoss2:
            minLoss2 = loss2
            model1.save(saveFolder + '%s_e%d_b%d_lo%.5f.h5'%(model1.name, epochs, batch_size, loss1))
            model2.save(saveFolder + '%s_e%d_b%d_lo%.5f.h5'%(model2.name, epochs, batch_size, loss2))

    print('==========epcohs: %d, loss1: %.5f, loss2:, %.5f'%(epoch, loss1, loss2))
    
#%% SAVE MODEL
model1.save(saveFolder + '%s_e%d_b%d_lo%.5f.h5'%(model1.name, epochs, batch_size, loss1))
model2.save(saveFolder + '%s_e%d_b%d_lo%.5f.h5'%(model2.name, epochs, batch_size, loss2))
#partModel_2.save(partModel_2.name+'.h5')
#%% USE
predict1 = model1.predict(dataSet["dataset32_x"][:5,:])
#predict1_img = predict1[0]#.reshape(5, 64, 64, 3)
predict2 = model2.predict(dataSet["dataset32_x"][:5,:])
#predict2_img = predict2[0]#.reshape(5, 64, 64, 3)

