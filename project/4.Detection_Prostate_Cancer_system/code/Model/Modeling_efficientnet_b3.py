#!/usr/bin/env python
# coding: utf-8

# ## Original : [PANDA keras baseline](https://www.kaggle.com/vgarshin/panda-keras-baseline)

# ## Module import 

# In[15]:


#warning 메시지 제거
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd 
import json
import cv2

#tiff 이미지 호출
import skimage.io

#train/test 셋 정의
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#모델정의
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import efficientnet.tfkeras as efn
from sklearn.metrics import cohen_kappa_score
import albumentations as albu


# ## GPU activate 

# In[16]:


print('tensorflow version:', tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print('no gpus')


# ## Variable define

# In[17]:


DATA_PATH = '/home/lab05/kaggle_dir'
MODELS_PATH = '.'
IMG_SIZE = 128
SEQ_LEN = 9 # 16^2 = 256
BATCH_SIZE = 16
MDL_VERSION = 'v22'
SEED = 80


# ## Data prepare 

# In[18]:


#BackUP
#데이터 준비
class DataGenPanda(Sequence):
    #initialize
    def __init__(self, imgs_path, df, batch_size=32, 
                 mode='fit', shuffle=False, aug=None, 
                 seq_len=12, img_size=128, n_classes=6):
        self.imgs_path = imgs_path
        self.df = df
        self.shuffle = shuffle
        self.mode = mode
        self.aug = aug
        self.batch_size = batch_size
        self.img_size = img_size
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.side = int(seq_len ** .5)
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self, index):
        X = np.zeros((self.batch_size, self.side * self.img_size, self.side * self.img_size, 3), dtype=np.float32)
        imgs_batch = self.df[index * self.batch_size : (index + 1) * self.batch_size]['image_id'].values
        for i, img_name in enumerate(imgs_batch):
            #print(img_name)
            img_path = '{}/{}.png'.format(self.imgs_path, img_name)
            #print(img_path)
            img_patches = self.get_patches(img_path)
            
            X[i, ] = self.glue_to_one(img_patches)
        if self.mode == 'fit':
            y = np.zeros((self.batch_size, self.n_classes), dtype=np.float32)
            lbls_batch = self.df[index * self.batch_size : (index + 1) * self.batch_size]['isup_grade'].values
            for i in range(self.batch_size):
                y[i, lbls_batch[i]] = 1
            return X, y
        elif self.mode == 'predict':
            return X
        else:
            raise AttributeError('mode parameter error')
            
    def get_patches(self, img_path):
        num_patches = self.seq_len
        p_size = self.img_size
        img = skimage.io.MultiImage(img_path)[0] #/ 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        #print(img.shape)
        if self.aug:
            img = self.aug(image=img)['image'] 
        pad0, pad1 = (p_size - img.shape[0] % p_size) % p_size, (p_size - img.shape[1] % p_size) % p_size
        img = np.pad(
            img,
            [
                [pad0 // 2, pad0 - pad0 // 2], 
                [pad1 // 2, pad1 - pad1 // 2], 
                [0, 0]
            ],
            constant_values=1
        )
        img = img.reshape(img.shape[0] // p_size, p_size, img.shape[1] // p_size, p_size, 3)
        img = img.transpose(0, 2, 1, 3, 4).reshape(-1, p_size, p_size, 3)
        if len(img) < num_patches:
            img = np.pad(
                img, 
                [
                    [0, num_patches - len(img)],
                    [0, 0],
                    [0, 0],
                    [0, 0]
                ],
                constant_values=1
            )
        idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:num_patches]
        return np.array(img[idxs])
    def glue_to_one(self, imgs_seq):
        img_glue = np.zeros((self.img_size * self.side, self.img_size * self.side, 3), dtype=np.float32)
        #img_size * side 만큼 float32비트 형식으로 0으로 채움
        
        for i, ptch in enumerate(imgs_seq):
            x = i // self.side
            y = i % self.side
            img_glue[x * self.img_size : (x + 1) * self.img_size, 
                     y * self.img_size : (y + 1) * self.img_size, :] = ptch
        return img_glue


# In[19]:


def del_error_data(error_df,train):
    error_list = list(error_df['image_id'])
    for img_id in error_list:
        train.drop(train[train['image_id']==img_id].index,inplace=True)
        train.reset_index(inplace=True,drop=True)
    #train[train['image_id']=='0b6e34bf65ee0810c1a4bf702b667c88'].index
    #train.drop(train[train['image_id']=='0b6e34bf65ee0810c1a4bf702b667c88'].index)
    print('after train: ', train.shape, '| after unique ids:', sum(train['isup_grade'].value_counts()))
    print(f"------------------------")
    
    return train


# In[20]:


#error_df = pd.read_csv(DATA_PATH+'/PANDA_Suspicious_Slides.csv')
train = pd.read_csv(DATA_PATH+'/train_v2.csv')
print('original train: ', train.shape, '| original unique ids:', sum(train['isup_grade'].value_counts()))
print(f"------------------------")
#에러 데이터 제거
#train = del_error_data(error_df,train)

#train/test 분리
X_train, X_val = train_test_split(train, test_size=.2, stratify=train['isup_grade'], random_state=SEED)

lbl_value_counts = X_train['isup_grade'].value_counts()
print(f"isup_grade counts")
print(lbl_value_counts)
print(f"------------------------")

class_weights = {i: max(lbl_value_counts) / v for i, v in lbl_value_counts.items()}
print('classes weigths:')
print(class_weights)
print(f"------------------------")


# In[21]:


from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate
)


# In[22]:


#data augmentation 수행
# aug = albu.Compose(
#     [
#         albu.HorizontalFlip(p=.25),
#         albu.VerticalFlip(p=.25),
#         albu.ShiftScaleRotate(shift_limit=.1, scale_limit=.1, rotate_limit=20, p=.25)
#     ]
# )
aug = albu.Compose([
        albu.RandomRotate90(),
        albu.Flip(),
        albu.Transpose(),
        albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        albu.OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        #HueSaturationValue(p=0.3),
    ])


#train data
train_datagen = DataGenPanda(
    imgs_path=DATA_PATH+'/test_image',
    df=X_train, 
    batch_size=BATCH_SIZE,
    mode='fit', 
    shuffle=True, 
    aug=aug, 
    seq_len=SEQ_LEN, 
    img_size=IMG_SIZE, 
    n_classes=6
)

#test data
val_datagen = DataGenPanda(
    imgs_path=DATA_PATH+'/test_image', 
    df=X_val, 
    batch_size=BATCH_SIZE,
    mode='fit', 
    shuffle=False, 
    aug=None, 
    seq_len=SEQ_LEN, 
    img_size=IMG_SIZE, 
    n_classes=6
)

# #example data
# example_datagen = DataGenPanda(
#     imgs_path=DATA_PATH+'/train_images',
#     df=X_train[:10], 
#     batch_size=BATCH_SIZE,
#     mode='fit', 
#     shuffle=True, 
#     aug=aug, 
#     seq_len=SEQ_LEN, 
#     img_size=IMG_SIZE, 
#     n_classes=6
# )


# In[23]:


Xt, yt = train_datagen.__getitem__(0)
print('test X: ', Xt.shape)
print('test y: ', yt.shape)
fig, axes = plt.subplots(figsize=(18, 6), ncols=BATCH_SIZE)
for j in range(BATCH_SIZE):
    axes[j].imshow(Xt[j])
    axes[j].axis('off')
    axes[j].set_title('label {}'.format(np.argmax(yt[j, ])))
plt.show()


# ## Train Model 

# ## Use pretrain model
# - source : [efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5](https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5)

# In[24]:


#pretrain_model = 'model_history/efficientnet/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
pretrain_model = 'model_history/efficientnet/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
bottleneck = efn.EfficientNetB3(
    input_shape=(int(SEQ_LEN ** .5) * IMG_SIZE, int(SEQ_LEN ** .5) * IMG_SIZE, 3),
    weights=pretrain_model, 
    include_top=False, 
    pooling='avg'
)
bottleneck = Model(inputs=bottleneck.inputs, outputs=bottleneck.layers[-2].output)
model = Sequential()
model.add(bottleneck)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(.2))
model.add(Dense(512, activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(.2))
model.add(Dense(6, activation='sigmoid'))
model.summary()


# In[25]:


def kappa_score(y_true, y_pred):
    
    y_true=tf.math.argmax(y_true)
    y_pred=tf.math.argmax(y_pred)
    return tf.compat.v1.py_func(cohen_kappa_score ,(y_true, y_pred),tf.double)


# In[26]:


# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=Adam(lr=1e-3),
#     metrics=['categorical_accuracy',kappa_score]
# )
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=1e-5),
    metrics=['binary_crossentropy',kappa_score]
)


# In[ ]:


get_ipython().run_cell_magic('time', '', "previous_file = 'model_history/efficientnet/model_v22_epoch_12_vloss_0.35.h5'\nmodel_file = 'model_history/efficientnet/model_'+MDL_VERSION+'_epoch_{epoch:02d}_vloss_{val_loss:.2f}.h5'\nif True:\n    model.load_weights(previous_file)\n    print('model loaded')\nelse:\n    print('train from scratch')\nEPOCHS = 25\nearlystopper = EarlyStopping(\n    monitor='val_loss', \n    patience=10, \n    verbose=1,\n    mode='min'\n)\nmodelsaver = ModelCheckpoint(\n    model_file, \n    monitor='val_loss', \n    verbose=1, \n    save_best_only=True,\n    mode='min'\n)\nlrreducer = ReduceLROnPlateau(\n    monitor='val_loss',\n    factor=.1,\n    patience=5,\n    verbose=1,\n    min_lr=1e-7\n)\nhistory = model.fit_generator(\n    train_datagen,\n    validation_data=val_datagen,\n    class_weight=class_weights,\n    callbacks=[earlystopper, modelsaver, lrreducer],\n    epochs=EPOCHS,\n    verbose=1\n)")


# ## Save history 

# In[ ]:


history_file = '{}/history_{}.txt'.format(MODELS_PATH, MDL_VERSION)
dict_to_save = {}
for k, v in history.history.items():
    dict_to_save.update({k: [np.format_float_positional(x) for x in history.history[k]]})
with open(history_file, 'w') as file:
    json.dump(dict_to_save, file)
ep_max = EPOCHS
plt.plot(history.history['loss'][:ep_max], label='loss')
plt.plot(history.history['val_loss'][:ep_max], label='val_loss')
plt.legend()
plt.show()
# plt.plot(history.history['categorical_accuracy'][:ep_max], label='categorical_accuracy')
# plt.plot(history.history['val_categorical_accuracy'][:ep_max], label='val_categorical_accuracy')
plt.plot(history.history['binary_crossentropy'][:ep_max], label='binary_crossentropy')
plt.plot(history.history['val_binary_crossentropy'][:ep_max], label='val_binary_crossentropy')
plt.legend()
plt.show()
plt.plot(history.history['kappa_score'][:ep_max], label='kappa_score')
plt.plot(history.history['val_kappa_score'][:ep_max], label='val_kappa_score')
plt.legend()
plt.show()


# In[ ]:


get_ipython().run_cell_magic('time', '', "previous_file = 'model_history/efficientnet/model_v20_efnb3.h5'\nmodel_file = 'model_history/efficientnet/model_'+MDL_VERSION+'_epoch_{epoch:02d}_vloss_{val_loss:.2f}.h5'\nif False:\n    model.load_weights(previous_file)\n    print('model loaded')\nelse:\n    print('train from scratch')\nEPOCHS = 25\nearlystopper = EarlyStopping(\n    monitor='val_loss', \n    patience=10, \n    verbose=1,\n    mode='min'\n)\nmodelsaver = ModelCheckpoint(\n    model_file, \n    monitor='val_loss', \n    verbose=1, \n    save_best_only=True,\n    mode='min'\n)\nlrreducer = ReduceLROnPlateau(\n    monitor='val_loss',\n    factor=.1,\n    patience=5,\n    verbose=1,\n    min_lr=1e-7\n)\nhistory = model.fit_generator(\n    train_datagen,\n    validation_data=val_datagen,\n    class_weight=class_weights,\n    callbacks=[earlystopper, modelsaver, lrreducer],\n    epochs=EPOCHS,\n    verbose=1\n)")


# In[ ]:


history_file = '{}/history_{}.txt'.format(MODELS_PATH, MDL_VERSION)
dict_to_save = {}
for k, v in history.history.items():
    dict_to_save.update({k: [np.format_float_positional(x) for x in history.history[k]]})
with open(history_file, 'w') as file:
    json.dump(dict_to_save, file)
ep_max = EPOCHS
plt.plot(history.history['loss'][:ep_max], label='loss')
plt.plot(history.history['val_loss'][:ep_max], label='val_loss')
plt.legend()
plt.show()
# plt.plot(history.history['categorical_accuracy'][:ep_max], label='categorical_accuracy')
# plt.plot(history.history['val_categorical_accuracy'][:ep_max], label='val_categorical_accuracy')
plt.plot(history.history['binary_crossentropy'][:ep_max], label='binary_crossentropy')
plt.plot(history.history['val_binary_crossentropy'][:ep_max], label='val_binary_crossentropy')
plt.legend()
plt.show()
plt.plot(history.history['kappa_score'][:ep_max], label='kappa_score')
plt.plot(history.history['val_kappa_score'][:ep_max], label='val_kappa_score')
plt.legend()
plt.show()


# ## Submission 

# In[ ]:


test = pd.read_csv('{}/test.csv'.format(DATA_PATH))
preds = [[0] * 6] * len(test)
if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):
    subm_datagen = DataGenPanda(
        imgs_path='{}/test_images'.format(DATA_PATH), 
        df=test,
        batch_size=1,
        mode='predict', 
        shuffle=False, 
        aug=None, 
        seq_len=SEQ_LEN, 
        img_size=IMG_SIZE, 
        n_classes=6
    )
    preds = model.predict_generator(subm_datagen)
    print('preds done, total:', len(preds))
else:
    print('preds are zeros')
test['isup_grade'] = np.argmax(preds, axis=1)
test.drop('data_provider', axis=1, inplace=True)
test.to_csv('submission.csv', index=False)
print('submission saved')


# In[ ]:




