#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#GPU 사용량 할당, 필요한 만큼 사용하도록
import tensorflow as tf


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
        print(e)

# use_GB = 4 #사용할 용량 , 기가 단위
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*use_GB)])
#     except RuntimeError as e:
#         # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
#         print(e)


# In[ ]:


def quadratic_kappa_coefficient(y_true, y_pred):
    y_true = K.cast(y_true, "float32")
    n_classes = K.cast(y_pred.shape[-1], "float32")
    weights = K.arange(0, n_classes, dtype="float32") / (n_classes - 1)
    weights = (weights - K.expand_dims(weights, -1)) ** 2

    hist_true = K.sum(y_true, axis=0)
    hist_pred = K.sum(y_pred, axis=0)

    E = K.expand_dims(hist_true, axis=-1) * hist_pred
    E = E / K.sum(E, keepdims=False)

    O = K.transpose(K.transpose(y_true) @ y_pred)  # confusion matrix
    O = O / K.sum(O)

    num = weights * O
    den = weights * E

    QWK = (1 - K.sum(num) / K.sum(den))
    return QWK

def quadratic_kappa_loss(scale=2.0):
    def _quadratic_kappa_loss(y_true, y_pred):
        QWK = quadratic_kappa_coefficient(y_true, y_pred)
        loss = -K.log(K.sigmoid(scale * QWK))
        return loss
        
    return _quadratic_kappa_loss


# In[ ]:


import os
import matplotlib.image as img
import numpy as np
import cv2
import pandas as pd


# In[ ]:


from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.callbacks.callbacks import ModelCheckpoint, EarlyStopping


# In[ ]:


class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="sigmoid"):
        # 인풋 이미지의 차원과, 채널에 해당하는 축을 설정하여 모델을 초기화합니다
        # "channels_last"는 채널의 축이 마지막에 오는 것을 의미합니다
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # 만약 "channels_first"를 사용한다면, 인풋 이미지의 차원을
        # 그에 맞게 바꿔줍니다
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
            
        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # FC => RELU
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # 단일 라벨 분류는 *softmax* 활성화 함수를 사용합니다
        # 다중 라벨 분류는 *sigmoid* 활성화 함수를 사용합니다
        model.add(Dense(classes))
        model.add(Activation(finalAct))

        # 네트워크 아키텍처를 반환합니다
        return model


# In[ ]:


# matplotlib의 백엔드를 설정하여 그림이 백그라운드에서 저장될 수 있게합니다
import matplotlib

# 필요한 패키지들을 가져옵니다
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder #One-hot 인코더
# from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os


# In[ ]:


# 학습을 위해 에폭과 초기 학습률, 배치 사이즈, 그리고 이미지의 차원을 초기화합니다
EPOCHS = 40
INIT_LR = 1e-3
BS = 16
IMAGE_DIMS = (256, 256, 3)


# In[ ]:


train = pd.read_csv('/home/lab05/kaggle_dir/train.csv')
test = pd.read_csv('//home/lab05/kaggle_dir/test.csv')


# In[ ]:


train.head()


# In[ ]:


def get_batch_ids(sequence, batch_size): #0~10616 리스트와 , 배이사이즈 100
    sequence = list(sequence) # 리스트를 또 리스트해?
    random.shuffle(sequence) # sequence를 랜덤하게 섞는다
    batch = random.sample(sequence, batch_size) #100개 가져온다?
    return batch #매번 호출마다 랜덤하게 가져온다


# In[ ]:


input_shape = (256, 256, 3)
def get_image(image_location):
    #print(image_location)
    image = cv2.imread(image_location)    
    # input 사이즈로 이미지 리사이즈
    image = cv2.resize(image, dsize=(input_shape[0], input_shape[1]), interpolation=cv2.INTER_AREA)
    
    return image


# In[ ]:


def tile(img, sz=256, N=16):
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    return img


# In[ ]:


train['conv_image_path'] = ['/home/lab05/kaggle_dir/image_256_16_1/' +image_name +".png" for image_name in train['image_id']]
# test_df['image_path'] = [dir + 'train_images/' +image_name +".tiff" for image_name in test_df['image_id']]
# test_df['conv_image_path'] = [dir + 'conv_train_images/' +image_name +".jpg" for image_name in test_df['image_id']]


# In[ ]:


train.head()


# In[ ]:


# 'isup_grade'를 기준으로 라벨인코딩 진행
encoder = OneHotEncoder(handle_unknown = 'ignore')
encoder_labels = pd.DataFrame(encoder.fit_transform(train[['isup_grade']]).toarray())
#display(encoder_labels)

train= pd.merge(train, encoder_labels, left_index=True, right_index=True)
train.head(4)


# In[ ]:


def data_generator(data, batch_size): #train_df 일부와, 배치사이즈 값 100
    while True:
        data = data.reset_index(drop=True)
        indices = list(data.index) #RangeIndex(start=0, stop=10616, step=1), 0~10616 리스트

        batch_ids = get_batch_ids(indices, batch_size) #배치사이즈만큼 랜덤하게 호출
        batch = data.iloc[batch_ids]['conv_image_path']

        X = [get_image(x) for x in batch] # 배치에 저장된 만큼 이미지를 리스트 형식으로 담는다.
        Y = data[[0, 1, 2, 3, 4, 5]].values[batch_ids]

        # Convert X and Y to arrays
        X = np.array(X)
        Y = np.array(Y)

        yield X, Y


# In[ ]:


train_v, test = train_test_split(train, test_size=0.1, random_state=42)


# In[ ]:


model_checkpoint = ModelCheckpoint('model.h5', monitor = 'val_loss', verbose=0, save_best_only=True, save_weights_only=True)
early_stop = EarlyStopping(monitor='val_loss',patience=5,verbose=True)


# In[ ]:


# 다중 라벨 분류를 수행할 수 있도록 sigmoid 활성화 함수를
# 네트워크의 마지막 레이어로 설정합니다
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=6,
	finalAct="sigmoid")

# 옵티마이저를 초기화합니다
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)


# In[ ]:


# 각각의 결과 라벨을 독립적인 베르누이 분포로 취급하기 위해
# 범주형 교차 엔트로피 대신 이진 교차 엔트로피를 사용하여 모델을 컴파일합니다

opt = Adam(lr=1e-3, decay=INIT_LR / EPOCHS)
model.compile(loss=quadratic_kappa_loss(scale=2.0), optimizer=opt,metrics = ['accuracy',quadratic_kappa_coefficient])
 
# 네트워크를 학습시킵니다
print("[INFO] training network...")
H = model.fit_generator(generator = data_generator(train_v, BS),
                        validation_data = data_generator(test, BS),
                        steps_per_epoch=len(train) // BS,
                        validation_steps = 20,
                        epochs=EPOCHS, verbose=1,
                        callbacks =[model_checkpoint, early_stop])


# In[ ]:





# In[ ]:




