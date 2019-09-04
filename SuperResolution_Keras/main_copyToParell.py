# -*- coding: utf-8 -*-

from utils_collect import LoadNPY, GetData, OWNLogger
from utils_collect import show_result, show_result_row
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
from model_collect import psnr, ssim
##%%
#from keras.applications.vgg16 import VGG16

#%% FLOW CONTROL
INT_FLOW_CONTROL = [1]
DICT_FLOW_NAME = {1:"載入資料庫", 
                  2:"網路建構",
                  3:"訓練",
                  7:"載入權重", #可以拿上次不錯、架構相似的繼續訓練
                  4:"驗證",
                  5:"測試",
                  6:"評估"}
#%% 參數設定 - 
# train
epochs = 40
batch_size = 16 #if 32 : 4G VRAM 不足，16 頂
model_weight_path = None # list
#%% logger 
saveFolder = "./result/_e{1}_b{2}_{0}".format("1", epochs, batch_size)
try:
    os.makedirs(saveFolder)
except:
    print("saveFolder", saveFolder, "is exsist.")

log = OWNLogger(logNPY = saveFolder, lossName=["loss_32-64", "loss_32-128", "PSNR", "SSIM"])
#%% LOAD DATASET
dataFolder = "./datasetNPY/"
subfolderList = os.listdir(dataFolder)
dataSet = dict()
for _n in subfolderList:
    tmpDict = LoadNPY(dataFolder+_n)
    dataSet.update(tmpDict)
#shuffle
index_shuffle = np.array([i for i in range(len(dataSet["dataset32_x"]))], dtype=np.int)
np.random.shuffle(index_shuffle)
#%% MODEL
# mainModel
def Model_TEST(scale = 2, num_filters = 64, num_res_blocks = 8, res_block_scaling = None, model_name = None, x_in = None): #origin (4, 64, 16, None)
    if x_in is None:
        x_in = Input(shape=(None, None, 3))
        x = Lambda(normalize)(x_in)
        x = b1 = Conv2D(num_filters, 3, padding='same')(x)
    else:
        x = b1 = Conv2D(num_filters, 3, padding='same')(x_in)
    
    for i in range(num_res_blocks):
        b1 = res_block(b1, num_filters, res_block_scaling)
    b1 = Conv2D(num_filters, 3, padding='same')(b1)
    x = Add()([x, b1])

    x = b2 = upsample(x, scale, num_filters)
    x = Conv2D(3, 3, padding='same')(x)
#    x = upsample(x, scale, num_filters)
#    x = b2 = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize)(x)
    return b2, Model(x_in, x, name=model_name)
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
#%% LOAD MODEL
if model_weight_path:
    model1.load_weights(model_weight_path[0], by_name=True)
    model2.load_weights(model_weight_path[1], by_name=True)
#%% train parm set
itr = int(len(dataSet["dataset32_x"])//batch_size) #207.75
print("epoch: %d, batch_szie: %d, itr max: %d"%(epochs, batch_size, itr))
minLoss1 = minLoss2 = minLoss3 = 100000000000
# LOG
log.ShowLocalTime()
log.SetLogTime("train")
log.UpdateProgSetting(itrMax = itr, batch_size = batch_size, epochs = epochs)
#%% TRAIN #要照她的嗎? https://github.com/krasserm/super-resolution/blob/master/train.py
for epoch in range(epochs):
    batch_index = 0
    log.SetLogTime("e%2d"%(epoch))
    print("epoch", epoch)
    for step in range(itr): # 壓制這個，把剩下的當 valid 也是方案
        batch_in  = GetData(dataSet, "dataset32_x",  batch_index, batch_size, index_shuffle) 
        batch_mid = GetData(dataSet, "dataset64_x",  batch_index, batch_size, index_shuffle)
        batch_out = GetData(dataSet, "dataset128_x", batch_index, batch_size, index_shuffle)
        batch_index += batch_size
        
#        
        loss1 = model1.train_on_batch(batch_in, batch_mid)
#        loss2 = model2.train_on_batch(batch_mid, batch_out)
        loss3 = model2.train_on_batch(model1.predict(batch_in), batch_out)
        
        if step%100 == 0 :
            print("itr: %d loss1: %d, loss3: %d"%(step, loss1, loss3))
#            print("itr: %d loss1: %d, loss2: %d, loss3: %d"%(step, loss1, loss2, loss3))
        if loss1 < minLoss1:
            print("e%d min %s: %.3f -> %.3f"%(epoch, "loss1", minLoss1, loss1))
            if epoch > 0:
                print("save model1")
                model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epoch, model1.name, batch_size, loss1))
            minLoss1 = loss1
#        if loss2 < minLoss2 and epoch > 0:
#            print("e%d min %s: %.3f -> %.3f"%(epoch, "loss2", minLoss2, loss2))
#            if epoch > 0:
#                print("save model1, model2")
#                model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epoch, model1.name, batch_size, loss1))
#                model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epoch, model2.name, batch_size, loss2))
#            minLoss2 = loss2
        if loss3 < minLoss3 and epoch > 0:
            print("e%d min %s: %.3f -> %.3f"%(epoch, "loss3", minLoss3, loss3))
            if epoch > 0:
                print("save model1, model2")
#                model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_lo3-%.5f_w.h5'%(epoch, model1.name, batch_size, loss1, loss3))
#                model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_lo3-%.5f_w.h5'%(epoch, model2.name, batch_size, loss2, loss3))
                model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_lo_w.h5'%(epoch, model1.name, batch_size, loss1))
                model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_lo_w.h5'%(epoch, model2.name, batch_size, loss3))
            minLoss3 = loss3
    # 可能要用 PSENR SSIM 來評估 除存與否
    if epoch % 1 == 0:
        # save loss
        log.AppendLossIn("loss_32-64", loss1)
#        log.AppendLossIn("loss_64-128", loss2)
        log.AppendLossIn("loss_32-128", loss3)
#        log.AppendLossIn("PSNR", psnr_epoch)
#        log.AppendLossIn("SSIM", ssim_epoch)
    # save weight
    model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_END_w.h5'%(epoch, model1.name, batch_size, loss1))
#    model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_END_w.h5'%(epoch, model2.name, batch_size, loss2))
    model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_END_w.h5'%(epoch, model2.name, batch_size, loss3))
        
    log.SetLogTime("e%2d"%(epoch), mode = "end")
#    print('==========epcohs: %d, loss1: %.5f, loss2:, %.5f, loss3:, %.5f ======='%(epoch, loss1, loss2, loss3))
    print('==========epcohs: %d, loss1: %.5f, loss3:, %.5f ======='%(epoch, loss1, loss3))
    # epoch 結束後，shuffle
    if epoch % 1 == 0:
        np.random.shuffle(index_shuffle)
#%% SAVE MODEL 上面存過了
#model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model1.name, batch_size, loss1))
#model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model2.name, batch_size, loss2))
log.SetLogTime("train", mode = "end")
#partModel_2.save(partModel_2.name+'.h5')
log.SaveLog2NPY()
#%% USE
predict1 = model1.predict(dataSet["dataset32_x"][:5,:])
predict2 = model2.predict(dataSet["dataset64_x"][:5,:])
predictFinal = model2.predict(predict1)
