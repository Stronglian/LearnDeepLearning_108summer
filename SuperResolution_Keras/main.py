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

#%% keras import
#from keras import backend as K
from keras.models import Model#, Sequential
from keras.layers import Input, Conv2D, Add, Lambda #, Dense,  Flatten, Activation, MaxPooling2D
from model_collect import res_block, normalize, denormalize, upsample
#from model_collect import psnr, ssim
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
epochs = 1 #25
batch_size = 16 #if 32 : 4G VRAM 不足，16 頂
model_weight_folder = "./"
model_weight_path = None # list
#model_weight_path = ["e40_x32-x64_model_b16_lo365.12378_w.h5", None] # "e40_x64-x128_model_b16_lo337.87949_w.h5"
model_discription = "x_Y-struct_e+3"
#%% logger 
saveFolder = "./result/_e{1:0>2d}_b{2}_{0}/".format(model_discription, epochs, batch_size)
try:
    os.makedirs(saveFolder)
except:
    print("saveFolder", saveFolder, "is exsist.")
#dictLossName = 
log = OWNLogger(logNPY = saveFolder, 
                lossName=["loss_32-64", "loss_32-128", 
                          "PSNR_32-64", "PSNR_32-128",
                          "SSIM_32-64", "SSIM_32-128"])
#%% LOAD DATASET
dataloader = DataLoader(dataFolder = "./datasetNPY/", batch_size = batch_size)
#%% MODEL
# mainModel
def Model_Block(scale = 2, num_filters = 64, num_res_blocks = 8, 
                res_block_scaling = None, model_name = None, x_in = None,
                name_id = "", name_output = None): #origin (4, 64, 16, None)
    if x_in is None:
        x_in = Input(shape = (None, None, 3))
        x = Lambda(normalize)(x_in)
        x = b1 = Conv2D(num_filters, 3, padding = 'same')(x)
    else:
        x = b1 = x_in
    
    for i in range(num_res_blocks):
        b1 = res_block(b1, num_filters, res_block_scaling)
    b1 = Conv2D(num_filters, 3, padding='same')(b1) # 老師畫的圖 沒有這步驟
    x = Add()([x, b1])

    x = b2 = upsample(x, scale, num_filters, name_id = name_id)
    x = Conv2D(3, 3, padding = 'same')(x)
#    x = upsample(x, scale, num_filters)
#    x = b2 = Conv2D(3, 3, padding='same')(x)

    x = Lambda(denormalize, name = name_output)(x)
#    return b2, Model(input = x_in, output = x, name=model_name)
    return x_in, x, b2 # in, out, branch
#%% MODEL compile
x_in, x_64,  m_branch = Model_Block(name_id="_32-64", name_output = "to64") # 32-64
_,    x_128, _        = Model_Block(x_in = m_branch, name_id = "_64-128", name_output = "to128") # 64-128

#model_all = Model(input = x_in, output = [x_64, x_128], name = "x32to64to128_model")
#model_all.compile(optimizer='adam', loss = ['mse', 'mse'])
model_all = Model(input = x_in, output = {"to64":x_64, "to128":x_128}, name = "x32to64to128")
model_all.compile(optimizer='adam', loss = {"to64":'mse', "to128":'mse'})

model_all.summary()
#%% LOAD MODEL
if model_weight_path and (7 in INT_FLOW_CONTROL):
    model_all.load_weights(model_weight_folder +model_weight_path[0], by_name=True)
#%% train parm set & loss log set
itr_max = dataloader.CalMaxIter() # int(len(dataSet["dataset32_x"])//batch_size) #207.75
print("epoch: %d, batch_szie: %d, itr max: %d"%(epochs, batch_size, itr_max))
def Cal_PSNR_SSIM(in1, in2, max_val=255):
    t1 = tf.convert_to_tensor(np.clip(in1, 0, 255))
    t2 = tf.convert_to_tensor(in2.astype(np.float32))
    
    out_psnr = tf.image.psnr(t1,  t2, max_val=255)
    out_ssim = tf.image.ssim(t1,  t2, max_val=255)
    return out_psnr, out_ssim
def CalMax(inVal_list, maxVal):
    inValAvg = np.average(inVal_list)
    if not maxVal: # is None:
        maxVal = inValAvg
        boolDO = True
    if inValAvg > maxVal:
        maxVal = inValAvg
        boolDO = True
    else:
        boolDO = False
    return maxVal, boolDO
### init
AMOUNT_OUT = 2

minLoss = list()
maxPSNR = list()
maxSSIM = list()
# DO
_DO_arr = dict()
boolFirst = list()
# tmp
ten_psnr = list()
ten_ssim = list()
out_psnr = list()
out_ssim = list()

for _i in range(AMOUNT_OUT):
    minLoss.append(100000000000)
    maxPSNR.append(None)
    maxSSIM.append(None)
    # DO
    _DO_arr[_i] = [0, 0]
    boolFirst.append(True)
    # tmp
    ten_psnr.append(None)
    ten_ssim.append(None)
    out_psnr.append(None)
    out_ssim.append(None)
    
# LOG
log.ShowLocalTime()
log.SetLogTime("train")
log.UpdateProgSetting(itrMax = itr_max, 
                      batch_size = batch_size, 
                      epochs = epochs, 
                      model_weight_path = model_weight_path,
                      model_discription = model_discription)
# SET
strShowLoss = "e%02d it%03d %s: %s %.3f <- %.3f"
strModelName_Loss = 'e%d_%s_b%d_lo%d_%.5f_w.h5'
strModelName_P_S  = 'e%d_%s_b%d_%d_P%.2f_S%.5f_w.h5'
#%% TRAIN #要照它的嗎? https://github.com/krasserm/super-resolution/blob/master/train.py
for epoch in range(epochs):
    print("epoch", epoch)
    log.SetLogTime("e%2d"%(epoch), boolPrint=True)
    batch_index = 0
    for step, (batch_in, batch_mid, batch_out) in enumerate(dataloader):
        loss_out = model_all.train_on_batch(x = batch_in, y = {"to64":batch_mid, "to128":batch_out})
#        loss_out = model_all.train_on_batch(x = batch_in, y = [batch_mid, batch_out])
        
        if step%50 == 0 :
            print("itr: %d loss:"%(step), *loss_out)
        for _l_i in range(AMOUNT_OUT-1, -1, -1): # 先測 32-128 的 loss 
            if loss_out[_l_i] < minLoss[_l_i]:
                print(strShowLoss%(epoch, step, "loss%d"%(_l_i), '"min"', minLoss[_l_i], loss_out[_l_i]))
                minLoss[_l_i] = loss_out[_l_i]
                if epoch > 0 or (epoch == 1 and boolFirst[_l_i]):
                    boolFirst[_l_i] = False
                    print("save model")
                    model_all.save_weights(saveFolder + strModelName_Loss%(epoch, model_all.name, batch_size, _l_i, loss_out[_l_i]))
        break # for TEST
    # 可能要用 PSENR SSIM 來評估 除存與否
    log.SetLogTime("e%02d_Valid"%(epoch), boolPrint=True)
    remainingIndex = (step+1)*batch_size
    batch_in   = dataloader.GetData("dataset32_x",  remainingIndex, batch_size = 5, ctype = "remaining")
    batch_mid  = dataloader.GetData("dataset64_x",  remainingIndex, batch_size = 5, ctype = "remaining")
    batch_out  = dataloader.GetData("dataset128_x", remainingIndex, batch_size = 5, ctype = "remaining")
#    batch_in   = dataloader.GetData("dataset32_x",  remainingIndex, ctype = "remaining")
#    batch_mid  = dataloader.GetData("dataset64_x",  remainingIndex, ctype = "remaining")
#    batch_out  = dataloader.GetData("dataset128_x", remainingIndex, ctype = "remaining")
    ## 預測
    predit_eval = model_all.predict(batch_in)
    ## 包成 tensor
#    for _l_i in range(AMOUNT_OUT):
    ten_psnr[0], ten_ssim[0] = Cal_PSNR_SSIM(predit_eval[0], batch_mid)
    ten_psnr[1], ten_ssim[1] = Cal_PSNR_SSIM(predit_eval[1], batch_out)
    ## (最可能出問題)計算
#    init_op = tf.global_variables_initializer() # 不知道會不會影響 KERAS????!?!?!??!?! # NEW
    with tf.Session() as sess:
        for _l_i in range(AMOUNT_OUT):
#            sess.run(init_op)
            out_psnr[_l_i] = sess.run(ten_psnr[_l_i])
            out_ssim[_l_i] = sess.run(ten_ssim[_l_i])
    ## 計算最大與決定是否存權重
    for _l_i in range(AMOUNT_OUT):
        maxPSNR[_l_i], _DO_arr[_l_i][0] = CalMax(out_psnr[_l_i], maxPSNR[_l_i])
        maxSSIM[_l_i], _DO_arr[_l_i][1] = CalMax(out_ssim[_l_i], maxSSIM[_l_i])
    for _l_i in range(AMOUNT_OUT):
        if all(_DO_arr[_l_i]):
            model_all.save_weights(saveFolder + strModelName_P_S%(epoch, model_all.name, batch_size, _l_i, maxPSNR[_l_i], maxSSIM[_l_i]))
    
    log.SetLogTime("e%2d_Valid"%(epoch), mode = "end")
    # LOSS 紀錄
    if epoch % 1 == 0:
        # save loss
        log.AppendLossIn("loss_32-64",  loss_out[0])
        log.AppendLossIn("PSNR_32-64",  out_psnr[0]) 
        log.AppendLossIn("SSIM_32-64",  out_ssim[0])
        log.AppendLossIn("loss_32-128", loss_out[1])
        log.AppendLossIn("PSNR_32-128", out_psnr[1])
        log.AppendLossIn("SSIM_32-128", out_ssim[1])
#    # save weight
#    model_all.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_lo%.5f_END_w.h5'%(epoch, model_all.name, batch_size, loss_out[0], loss_out[1]))
        
    log.SetLogTime("e%2d"%(epoch), mode = "end")
    print('==========epcohs: %d, loss0: %.5f, loss1:, %.5f ======='%(epoch, loss_out[0], loss_out[1]))
    # epoch 結束後，shuffle
    if epoch % 1 == 0:
        dataloader.ShuffleIndex()
    break # for TEST
#%% SAVE MODEL 上面存過了
#model1.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model1.name, batch_size, loss1))
#model2.save_weights(saveFolder + 'e%d_%s_b%d_lo%.5f_w.h5'%(epochs, model2.name, batch_size, loss2))
log.SetLogTime("train", mode = "end")
#partModel_2.save(partModel_2.name+'.h5')
log.SaveLog2NPY()
#%% USE
predict_part = model_all.predict(dataloader.dataSet["dataset32_x"][:5,:])
