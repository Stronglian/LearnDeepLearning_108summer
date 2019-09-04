# -*- coding: utf-8 -*-

from utils_collect import OWNLogger, DataLoader
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
INT_FLOW_CONTROL = [1, 2, 3]
DICT_FLOW_NAME = {1:"載入資料庫", 
                  2:"網路建構",
                  3:"訓練",
                  7:"載入權重", #可以拿上次不錯、架構相似的繼續訓練
                  4:"驗證",
                  5:"測試",
                  6:"評估"}
#%% 參數設定 - 
# train
epochs = 3
batch_size = 16 #if 32 : 4G VRAM 不足，16 頂
model_weight_path = None # list
#model_weight_path = ["load_weight_0904/", None] # list
#%% logger 
saveFolder = "./result/_e{1}_b{2}_{0}/".format("TEST", epochs, batch_size)
try:
    os.makedirs(saveFolder)
except:
    print("saveFolder", saveFolder, "is exsist.")

log = OWNLogger(logNPY = saveFolder, 
                lossName=["loss_32-64", "loss_32-128", 
                          "PSNR_32-64", "PSNR_32-128",
                          "SSIM_32-64", "SSIM_32-128"])
#%% LOAD DATASET
dataloader = DataLoader(dataFolder = "./datasetNPY/", batch_size = batch_size)
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

#%%
m_branch, model1 = Model_TEST(model_name="x32-x64_model")
model1.summary()
model1.compile('adam',loss='mse')

#_, model2 = Model_TEST(model_name="x32-x128_model", x_in=m_branch)
_, model2 = Model_TEST(model_name="x64-x128_model")
model2.summary()
model2.compile('adam',loss='mse')

#model3 = Model(input = model1.input, output = [model1.output, model2.output])
#model3.compile('adam',loss='mse')
#%% LOAD MODEL
if model_weight_path:
    model1.load_weights(model_weight_path[0], by_name=True)
    if model_weight_path[1]:
        model2.load_weights(model_weight_path[1], by_name=True)
#%% train parm set
itr_max = dataloader.CalMaxIter() # int(len(dataSet["dataset32_x"])//batch_size) #207.75
print("epoch: %d, batch_szie: %d, itr max: %d"%(epochs, batch_size, itr_max))
minLoss1 = minLoss2 = minLoss3 = 100000000000

maxPSNR1 = maxSSIM1 = maxPSNR3 = maxSSIM3 = None
def Cal_PSNR_SSIM(in1, in2, max_val=255):
    t1 = tf.convert_to_tensor(np.clip(in1, 0, 255))
    t2 = tf.convert_to_tensor(in2.astype(np.float32))
    
    out_psnr = tf.image.psnr(t1,  t2, max_val=255)
    out_ssim = tf.image.ssim(t1,  t2, max_val=255)
    return out_psnr, out_ssim
def CalMax(inVal_list, maxVal):
    inValAvg = np.average(inVal_list)
    if inValAvg > maxVal or not maxVal:
        maxVal = inValAvg
        boolDO = True
    else:
        boolDO = False
    return maxVal, boolDO
_DO_arr = {1:[0, 0], 3:[0, 0]}
# LOG
log.ShowLocalTime()
log.SetLogTime("train")
log.UpdateProgSetting(itrMax = itr_max, batch_size = batch_size, epochs = epochs, model_weight_path = model_weight_path)
# SET
strShowLoss = "\re%02d it%03d %s: 'min' %.3f <- %.3f =="
strModelName_Loss = 'e%d_%s_b%d_lo%.5f_w.h5'
strModelName_P_S  = 'e%d_%s_b%d_P%.2f_S%.5f_w.h5'
boolFirst = [True, True, True]
#%% TRAIN #要照它的嗎? https://github.com/krasserm/super-resolution/blob/master/train.py
for epoch in range(epochs):
    print("epoch", epoch)
    log.SetLogTime("e%2d"%(epoch), boolPrint=True)
    batch_index = 0
    for step, (batch_in, batch_mid, batch_out) in enumerate(dataloader):
#    for step in range(itr_max): # 壓制這個，把剩下的當 valid 也是方案
#        batch_in  = dataloader.GetData("dataset32_x",  batch_index, batch_size)
#        batch_mid = dataloader.GetData("dataset64_x",  batch_index, batch_size)
#        batch_out = dataloader.GetData("dataset128_x",  batch_index, batch_size)
#        batch_index += batch_size
        
        loss1 = model1.train_on_batch(batch_in, batch_mid)
        loss3 = model2.train_on_batch(model1.predict(batch_in), batch_out)
        
        if step%100 == 0 :
            print("itr: %d loss1: %d, loss3: %d"%(step, loss1, loss3))
        if loss1 < minLoss1:
            print(strShowLoss%(epoch, step, "loss1", minLoss1, loss1))
            if epoch > 0 or (epoch == 1 and boolFirst[0]):
                boolFirst[0] = False
                print("save model1")
                model1.save_weights(saveFolder + strModelName_Loss%(epoch, model1.name, batch_size, loss1))
            minLoss1 = loss1
        if loss3 < minLoss3:
            print(strShowLoss%(epoch, step, "loss3", minLoss3, loss3))
            if epoch > 0 or (epoch == 1 and boolFirst[2]):
                boolFirst[2] = False
                print("save model1, model2")
                model1.save_weights(saveFolder + strModelName_Loss%(epoch, model1.name, batch_size, loss1))
                model2.save_weights(saveFolder + strModelName_Loss%(epoch, model2.name, batch_size, loss3))
            minLoss3 = loss3
    # 可能要用 PSENR SSIM 來評估 除存與否
    log.SetLogTime("e%2d_Valid"%(epoch), boolPrint=True)
    remainingIndex = step*batch_size
    batch_in   = dataloader.GetData("dataset32_x",  remainingIndex, ctype = "remaining")
    batch_mid  = dataloader.GetData("dataset64_x",  remainingIndex, ctype = "remaining")
    batch_out  = dataloader.GetData("dataset128_x", remainingIndex, ctype = "remaining")
    ## 預測
    predit1 = model1.predict(batch_in)
    predit2 = model2.predict(predit1)
    ## 包成 tensor
    tmp_psnr1, tmp_ssim1 = Cal_PSNR_SSIM(predit1, batch_mid)
    tmp_psnr3, tmp_ssim3 = Cal_PSNR_SSIM(predit2, batch_out)
    ## (最可能出問題)計算
    init_op = tf.initialize_all_variables() # 不知道會不會影響 KERAS????!?!?!??!?!
    with tf.Session() as sess:
        sess.run(init_op) #execute init_op
        #print the random values that we sample
        out_psnr1 = sess.run(tmp_psnr1)
        out_ssim1 = sess.run(tmp_ssim1)
        out_psnr3 = sess.run(tmp_psnr3)
        out_ssim3 = sess.run(tmp_ssim3)
    ## 計算最大與決定是否存權重
    maxPSNR1, _DO_arr[1][0] = CalMax(out_psnr1, maxPSNR1)
    maxSSIM1, _DO_arr[1][1] = CalMax(out_ssim1, maxSSIM1)
    maxPSNR3, _DO_arr[3][0] = CalMax(out_psnr3, maxPSNR3)
    maxSSIM3, _DO_arr[3][1] = CalMax(out_ssim3, maxSSIM3)
    
    if   all(_DO_arr[3]):
        model1.save_weights(saveFolder + strModelName_P_S%(epoch, model1.name, batch_size, maxPSNR1, maxSSIM1))
        model2.save_weights(saveFolder + strModelName_P_S%(epoch, model2.name, batch_size, maxPSNR3, maxSSIM3))
    elif all(_DO_arr[1]):
        model1.save_weights(saveFolder + strModelName_P_S%(epoch, model1.name, batch_size, maxPSNR1, maxSSIM1))
    
    log.SetLogTime("e%2d_Valid"%(epoch), mode = "end")
    # LOSS 紀錄
    if epoch % 1 == 0:
        # save loss
        log.AppendLossIn("loss_32-64",  loss1)
        log.AppendLossIn("loss_32-128", loss3)
        log.AppendLossIn("PSNR_32-64",  out_psnr1) # 全存
        log.AppendLossIn("SSIM_32-64",  out_ssim1)
        log.AppendLossIn("PSNR_32-128", out_psnr3)
        log.AppendLossIn("SSIM_32-128", out_ssim3)
    # save weight
    model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_END_w.h5'%(epoch, model1.name, batch_size, loss1))
    model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_END_w.h5'%(epoch, model2.name, batch_size, loss3))
        
    log.SetLogTime("e%2d"%(epoch), mode = "end")
    print('==========epcohs: %d, loss1: %.5f, loss3:, %.5f ======='%(epoch, loss1, loss3))
    # epoch 結束後，shuffle
    if epoch % 1 == 0:
        dataloader.ShuffleIndex()
#%% SAVE MODEL 上面存過了
#model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model1.name, batch_size, loss1))
#model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model2.name, batch_size, loss2))
log.SetLogTime("train", mode = "end")
#partModel_2.save(partModel_2.name+'.h5')
log.SaveLog2NPY()
#%% USE
predict1 = model1.predict(dataloader.dataSet["dataset32_x"][:5,:])
predict2 = model2.predict(dataloader.dataSet["dataset64_x"][:5,:])
predictFinal = model2.predict(predict1)
