#!/usr/bin/env python
# coding: utf-8

# # 모듈호출

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import animation, rc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder

#tiff 이미지 호출
import skimage.io

import os
import gc
import json
import cv2
import collections
import math
import glob

#케라스 딥러닝 적용
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping, Callback
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical ,Sequence, get_custom_objects
from swa.keras import SWA # swa optimizer - https://pypi.org/project/keras-swa/
from tensorflow.keras.models import load_model

import albumentations as albu

#warning 무시
import warnings
warnings.filterwarnings('ignore')

print(tf.__version__)


# ## GPU SEtting 

# In[2]:


#GPU 사용량 할당, 필요한 만큼 사용하도록
import tensorflow as tf

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
#         print(e)
        
use_GB = 12 #사용할 용량 , 기가 단위
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*use_GB)])
    except RuntimeError as e:
        # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
        print(e)


# ## Variable setting

# In[3]:


DATA_PATH = 'dataset'
MODEL_NAME = 'MLP'
version = 'V8'
SEED = 2020
IMG_SIZE = 256
SEQ_LEN = 25 # 16^2 = 256
BATCH_SIZE = 1


# # 데이터 호출

# In[5]:


#데이터 호출
train_df = pd.read_csv('train_result_rgb_gray_20200616.csv')

#train x 지정
train_x = train_df.loc[:,'R_0':'GRAY_255']
#train y 지정, one hot encoding
train_y = train_df['gleason_score']
train_y = np.array(train_y)
train_y = to_categorical(train_y, 6)

display(train_df.head(3))
display(train_x.head(3))
display(train_y[:3])
#train/test 지정
#X_train, X_val = train_test_split(merge_df.loc[:,'R_0':'B_255'], test_size=.2, stratify=merge_df['isup_grade'], random_state=SEED)


# ## MODEL

# In[6]:


class Gelu(Activation):
    def __init__(self, activation, **kwargs):
        super(Gelu, self).__init__(activation, **kwargs)
        self.__name__='gelu'
        
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

get_custom_objects().update({'gelu': Gelu(gelu)})


# In[7]:


#qudratic weighted kappa score
def QWK(y_true, y_pred):

    y_true = K.cast(K.argmax(y_true, axis=-1), dtype='int32')
    y_pred = K.cast(K.argmax(y_pred, axis=-1), dtype='int32')

    # Figure out normalized expected values
    min_rating = K.minimum(K.min(y_true), K.min(y_pred))
    max_rating = K.maximum(K.max(y_true), K.max(y_pred))

    # shift the values so that the lowest value is 0
    # (to support scales that include negative values)
    y_true = K.map_fn(lambda y: y - min_rating, y_true, dtype='int32')
    y_pred = K.map_fn(lambda y: y - min_rating, y_pred, dtype='int32')

    # Build the observed/confusion matrix
    num_ratings = max_rating - min_rating + 1
    observed = tf.math.confusion_matrix(y_true, y_pred,
                                num_classes=num_ratings)
    num_scored_items = K.shape(y_true)[0]

    weights = K.expand_dims(K.arange(num_ratings), axis=-1) - K.expand_dims(K.arange(num_ratings), axis=0)
    weights = K.cast(K.pow(weights, 2), dtype='float64')

    hist_true = tf.math.bincount(y_true, minlength=num_ratings)
    hist_true = hist_true[:num_ratings] / num_scored_items
    hist_pred = tf.math.bincount(y_pred, minlength=num_ratings)
    hist_pred = hist_pred[:num_ratings] / num_scored_items
    expected = K.dot(K.expand_dims(hist_true, axis=-1), K.expand_dims(hist_pred, axis=0))

    # Normalize observed array
    observed = observed / num_scored_items

    # If all weights are zero, that means no disagreements matter.
    score = tf.where(K.any(K.not_equal(weights, 0)), 
                     K.sum(weights * observed) / K.sum(weights * expected), 
                     0)
    
    return 1. - score


class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose != 0 :#&gt: 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


# ## Model setting 

# In[8]:


# model_path = 'model_history/mlp/'+MODEL_NAME+'_'+version+'_epoch:{epoch:02d}_vloss:{val_loss:.2f}_vqwk:{val_QWK:.2f}.hdf5'
#model_path = 'model_history/mlp/'+MODEL_NAME+'_'+version+'.hdf5'
# earlystopper = EarlyStopping(
#     monitor='val_loss', 
#     patience=100, 
#     verbose=1,
#     mode='min'
# )
# modelsaver = ModelCheckpoint(
#     model_path, 
#     monitor='val_loss', 
#     verbose=1, 
#     #save_best_only=True, #가장 좋은거만 저장
#     save_weights_only=True, #weights 값만 저장
#     #save_freq = 20, #저장하는 epoch 주기
#     #mode='min' #monitor 값이 제일 최소가 될 때 저장
#     period=50
# )

# lrreducer = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=.1,
#     patience=10,
#     verbose=1,
#     min_lr=5e-10
# )
# cosine_scheduler = CosineAnnealingScheduler(T_max=100, eta_max=6e-4, eta_min=3e-5)
# swa = SWA(start_epoch=20, lr_schedule='manual', swa_lr=3e-4, swa_freq=5, verbose=1,batch_size=4096)


# In[ ]:





# In[9]:


# model = Sequential()
# model.add(Dense(2000, activation=gelu, input_dim=768))
# model.add(BatchNormalization())

# model.add(Dense(5000, activation=gelu))
# model.add(BatchNormalization())

# model.add(Dense(8000, activation=gelu))
# model.add(BatchNormalization())

# model.add(Dense(11000, activation=gelu))
# model.add(BatchNormalization())

# model.add(Dense(14000, activation=gelu))
# model.add(BatchNormalization())

# model.add(Dense(11000, activation=gelu))
# model.add(BatchNormalization())

# model.add(Dense(8000, activation=gelu))
# model.add(BatchNormalization())

# model.add(Dense(5000, activation=gelu))
# model.add(BatchNormalization())

# model.add(Dense(2000, activation=gelu))
# model.add(BatchNormalization())

# model.add(Dense(500, activation=gelu))
# model.add(BatchNormalization())

# model.add(Dense(6, activation='sigmoid')) # OUTPUT
# model.summary()



# model.compile(loss='binary_crossentropy',
#               optimizer=adams,
#               metrics=['accuracy',QWK]
#               )


# ## Model training

# In[8]:


lr = 3e-4
lr_d = 0.0
patience = 200
dr_rate = 0.01

# K-Fold 교차검증
kfold = 10
kf = KFold(n_splits=kfold, shuffle=True, random_state=2020)


# In[9]:


# train model
history_list = []
for enum, (train_index,valid_index) in enumerate(kf.split(train_x,train_y)):
    file_path = 'model_history/mlp/'+MODEL_NAME+'_'+version+'_fold_'+str(enum)+'.hdf5'
    #check_point = ModelCheckpoint(file_path, monitor="val_loss", verbose=1,save_best_only=True, mode="min")
    #early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    earlystopper = EarlyStopping(
        monitor='val_loss', 
        patience=500, 
        verbose=1,
        mode='min'
    )
    modelsaver = ModelCheckpoint(
        file_path, 
        monitor='val_loss', 
        verbose=1, 
        save_best_only=True, #가장 좋은거만 저장
        save_weights_only=True, #weights 값만 저장
        #save_freq = 20, #저장하는 epoch 주기
        mode='min' #monitor 값이 제일 최소가 될 때 저장
        #period=50
    )

    
    cosine_scheduler = CosineAnnealingScheduler(T_max=100, eta_max=6e-4, eta_min=3e-5,verbose=1)
    swa = SWA(start_epoch=20, lr_schedule='manual', swa_lr=3e-4, swa_freq=5, verbose=1,batch_size=4096)
    
    #print(train_index,valid_index)
    
    kf_x_train = train_x.iloc[train_index]
    kf_y_train = train_y[train_index]
    
    kf_x_val = train_x.iloc[valid_index]
    kf_y_val = train_y[valid_index]

    model = Sequential()
    model.add(Dense(1200, activation=gelu, input_dim=1024))
    model.add(BatchNormalization())

    model.add(Dense(1500, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(1800, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(2100, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(2400, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(2100, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(1800, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(1500, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(1200, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(900, activation=gelu))
    model.add(BatchNormalization())
    
    model.add(Dense(600, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(300, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(100, activation=gelu))
    model.add(BatchNormalization())

    model.add(Dense(50, activation=gelu))
    model.add(BatchNormalization())
    
    model.add(Dense(6, activation='sigmoid')) # OUTPUT
    model.summary()



    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr, decay=lr_d),
                  metrics=['accuracy',QWK]
                  )
    
    hist = model.fit(kf_x_train, kf_y_train,
              batch_size=3000, epochs=1000,
              validation_data = [kf_x_val, kf_y_val],
              verbose=1, callbacks=[earlystopper,modelsaver,cosine_scheduler,swa])
    history_list.append(hist)


# In[ ]:


# epoch_num = 1000
# batch_size_num = 3000
# validation_split_nm =  0.2
# hist = model.fit(train_x, train_y, epochs=epoch_num, batch_size=batch_size_num, validation_split=validation_split_nm, callbacks=[earlystopper,modelsaver,cosine_scheduler, swa ])


# In[ ]:


# model.save_weights('model_history/mlp/'+MODEL_NAME+'_'+version+'_last.hdf5')


# In[12]:


index = 0 

dict_to_save = {}
for hist in history_list:
    history_file = 'model_history/mlp/history_{}_{}.txt'.format(version,index)
    for k, v in hist.history.items():
        dict_to_save.update({k: [np.format_float_positional(x) for x in hist.history[k]]})
    
    with open(history_file, 'w') as file:
        json.dump(dict_to_save, file)
    
    index += 1
    
    ep_max = len(hist.history['loss'])
    plt.plot(hist.history['loss'][:ep_max], label='loss')
    plt.plot(hist.history['val_loss'][:ep_max], label='val_loss')
    plt.legend()
    plt.savefig(f'model_history/mlp/history_{version}_fig_{str(index)}_loss.jpg')
    plt.show()
    plt.plot(hist.history['accuracy'][:ep_max], label='binary_accuracy')
    plt.plot(hist.history['val_accuracy'][:ep_max], label='val_binary_accuracy')
    plt.legend()
    plt.savefig(f'model_history/mlp/history_{version}_fig_{str(index)}_accuracy.jpg')
    plt.show()
    plt.plot(hist.history['QWK'][:ep_max], label='QWK')
    plt.plot(hist.history['val_QWK'][:ep_max], label='val_QWK')
    plt.legend()
    plt.savefig(f'model_history/mlp/history_{version}_fig_{str(index)}_QWK.jpg')
    plt.show()


# In[13]:


index = 0 

dict_to_save = {}
for hist in history_list:
    history_file = 'model_history/mlp/history_{}_{}.txt'.format(version,index)
    for k, v in hist.history.items():
        dict_to_save.update({k: [np.format_float_positional(x) for x in hist.history[k]]})
    
    with open(history_file, 'w') as file:
        json.dump(dict_to_save, file)
    
    index += 1
    
    ep_max = len(hist.history['loss'])
    plt.plot(hist.history['loss'][:ep_max], label='loss')
    plt.plot(hist.history['val_loss'][:ep_max], label='val_loss')
    plt.legend()
    plt.savefig(f'model_history/mlp/history_{version}_fig_{str(index-1)}_loss.jpg')
    plt.show()
    plt.plot(hist.history['accuracy'][:ep_max], label='binary_accuracy')
    plt.plot(hist.history['val_accuracy'][:ep_max], label='val_binary_accuracy')
    plt.legend()
    plt.savefig(f'model_history/mlp/history_{version}_fig_{str(index-1)}_accuracy.jpg')
    plt.show()
    plt.plot(hist.history['QWK'][:ep_max], label='QWK')
    plt.plot(hist.history['val_QWK'][:ep_max], label='val_QWK')
    plt.legend()
    plt.savefig(f'model_history/mlp/history_{version}_fig_{str(index-1)}_QWK.jpg')
    plt.show()


# In[ ]:


# y_predict = model.predict(train_x.iloc[:10])


# In[ ]:


# real = np.argmax(train_y[:10], axis=1)
# result = np.argmax(y_predict, axis=1)
# print(real)
# print(result)

# kappa_val = QWK(real,result)
# print(kappa_val)


# ## Predict

# In[ ]:


# model = load_model('../input/panda-mlp-model/MLP_V1_epoch_14_vloss_0.51.hdf5')
# model.summary()


# ## Submission 

# In[ ]:


# test = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')
# tmp_df = pd.DataFrame()
# preds = [[0] * 6] * len(test)
# if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):
#     """
#     test.csv 의 'image_id'를 참조하여 preprocess 함수에서 256x256x36 tile을 각 RGB 영역별로 분석
#     """
#     for i in range(len(test)):
#         test_datagen = DataGenPanda(
#                 imgs_path=DATA_PATH+'/test_images',
#                 df=test[i:i+1], 
#                 batch_size=1,#BATCH_SIZE,
#                 mode='predict', 
#                 shuffle=False, 
#                 aug=None, 
#                 seq_len=SEQ_LEN, 
#                 img_size=IMG_SIZE, 
#                 n_classes=6
#             )
#         Xt = test_datagen.__getitem__(0)
#         result_df = preprocess(Xt[0])
#         tmp_df = pd.concat([tmp_df,result_df])
#     tmp_df.reset_index(inplace=True,drop=True)
#     #display(tmp_df)
#     preds = model.predict(tmp_df)
#     print('preds done, total:', len(preds))
# else:
#     print('preds are zeros')
# test['isup_grade'] = np.argmax(preds, axis=1)
# test.drop('data_provider', axis=1, inplace=True)
# test.to_csv('submission.csv', index=False)
# print('submission saved')

