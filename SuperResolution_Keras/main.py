# -*- coding: utf-8 -*-

from utils_collect import LoadNPY, GetData, OWNLogger, show_result
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
from keras.layers import Input, Conv2D, Add, Lambda #, Dense,  Flatten, Activation, MaxPooling2D
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
saveFolder = "./w3/"
try:
    os.makedirs(saveFolder)
except:
    print("saveFolder", saveFolder, "is exsist.")

log = OWNLogger(logNPY = saveFolder, lossName=["loss_32-64", "loss_64-128", "loss_32-128"])
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
    if x_in is None:
        x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = b = Conv2D(num_filters, 3, padding='same')(x)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])

#    x = b = upsample(x, scale, num_filters)
#    x = Conv2D(3, 3, padding='same')(x)
    x = upsample(x, scale, num_filters)
    x = b = Conv2D(3, 3, padding='same')(x)

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
#_, model2 = Model_TEST(model_name="x32-x128_model", x_in=m_branch)
_, model2 = Model_TEST(model_name="x64-x128_model")
model2.summary()
model2.compile('adam',loss='mse')

#model3 = Model(input = model1.input, output = [m_branch, model2.output])
#%% train parm set
epochs = 40
batch_size = 16 #if 32 : 4G VRAM 不足，16 頂
itr = int(len(dataSet["dataset32_x"])//batch_size) #207.75
print("epoch: %d, batch_szie: %d, itr max: %d"%(epochs, batch_size, itr))
minLoss1 = minLoss2 = minLoss3 = 100000000000
log.ShowLocalTime()
log.SetLogTime("train")
#%% TRAIN #要照她的嗎? https://github.com/krasserm/super-resolution/blob/master/train.py
for epoch in range(epochs):
    batch_index = 0
    log.SetLogTime("e%2d"%(epoch))
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
#        loss2 = model2.train_on_batch(batch_in, batch_out)
        loss2 = model2.train_on_batch(batch_mid, batch_out)
        loss3 = model2.train_on_batch(model1.predict(batch_in), batch_out)
        
        if step%100 == 0 :
            print("itr: %d loss1: %d, loss2: %d"%(step, loss1, loss2))
#            print('itr:',step,' total_loss:', loss[0], ' loss:',loss[1:])
#            print('itr:%4d, \n1- total_loss:%7.4f loss:'%(step, loss1[0]), loss1[1:])
#            print('2- total_loss:%7.4f loss:'%(loss2[0]), loss2[1:])
#            print('3- total_loss:%7.4f loss:'%(loss3[0]), loss3[1:])
        if loss1 < minLoss1:
            minLoss1 = loss1
            if epoch > 0:
                print("save model1")
                model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model1.name, batch_size, loss1))
        if loss2 < minLoss2 and epoch > 0:
            minLoss2 = loss2
            if epoch > 0:
                print("save model1, model2")
                model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model1.name, batch_size, loss1))
                model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model2.name, batch_size, loss2))
        if loss3 < minLoss3 and epoch > 0:
            minLoss3 = loss3
            if epoch > 0:
                print("save model1, model2")
                model1.save_weights(saveFolder + 'e%d_%s_b%d_3_lo%.5f_w.h5'%(epochs, model1.name, batch_size, loss1))
                model2.save_weights(saveFolder + 'e%d_%s_b%d_3_lo%.5f_w.h5'%(epochs, model2.name, batch_size, loss2))
        # save weight
        model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model1.name, batch_size, loss1))
        model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model2.name, batch_size, loss2))
        # save loss
        log.AppendLossIn("loss_32-64", loss1)
        log.AppendLossIn("loss_64-128", loss2)
        log.AppendLossIn("loss_32-128", loss3)
        
    print('==========epcohs: %d, loss1: %.5f, loss2:, %.5f'%(epoch, loss1, loss2))
    log.SetLogTime("e%2d"%(epoch), mode = "end")
#%% SAVE MODEL
model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model1.name, batch_size, loss1))
model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model2.name, batch_size, loss2))
log.SetLogTime("train", mode = "end")
#partModel_2.save(partModel_2.name+'.h5')
log.SaveLog2NPY()
#%% USE
predict1 = model1.predict(dataSet["dataset32_x"][:5,:])
#predict1_img = predict1[0]#.reshape(5, 64, 64, 3)
predict2 = model2.predict(dataSet["dataset64_x"][:5,:])
#predict2_img = predict2[0]#.reshape(5, 64, 64, 3)
predictFinal = model2.predict(predict1)
