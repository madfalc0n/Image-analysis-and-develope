#!/usr/bin/env python
# coding: utf-8

# ## Original : [PANDA keras baseline](https://www.kaggle.com/vgarshin/panda-keras-baseline)

# In[ ]:


# !pip install ../input/panda-efficientnet/efficientnet-1.1.0-py3-none-any.whl


# ## Module import 

# In[1]:


#warning 메시지 제거
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd 
import json

#tiff 이미지 호출
import skimage.io

#train/test 셋 정의
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

import matplotlib.pyplot as plt

#모델정의
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras.backend as K

import efficientnet.tfkeras as efn

import albumentations as albu


# ## GPU activate 

# In[2]:


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
    
    
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)


# ## Variable define

# In[3]:


DATA_PATH = 'dataset/'
MODELS_PATH = '.'
IMG_SIZE = 256 
SEQ_LEN = 36 # 16^2 = 256
BATCH_SIZE = 4
MDL_VERSION = 'v25'
SEED = 80


# ## Data prepare 

# In[4]:


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
            img_path = '{}/{}.tiff'.format(self.imgs_path, img_name)
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
        img = skimage.io.MultiImage(img_path)[1] / 255
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


# In[5]:


def del_error_data(error_df,train):
    print('original train: ', train.shape, '| original unique ids:', sum(train['isup_grade'].value_counts()))
    print(f"------------------------")
    error_list = list(error_df['image_id'])
    for img_id in error_list:
        train.drop(train[train['image_id']==img_id].index,inplace=True)
        train.reset_index(inplace=True,drop=True)
    #train[train['image_id']=='0b6e34bf65ee0810c1a4bf702b667c88'].index
    #train.drop(train[train['image_id']=='0b6e34bf65ee0810c1a4bf702b667c88'].index)
    print('after train: ', train.shape, '| after unique ids:', sum(train['isup_grade'].value_counts()))
    print(f"------------------------")
    
    return train


# In[6]:


error_df = pd.read_csv(DATA_PATH + 'PANDA_Suspicious_Slides.csv')
train = pd.read_csv(DATA_PATH+'train.csv')

#에러 데이터 제거
train = del_error_data(error_df,train)

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


# In[ ]:


#data augmentation 수행
aug = albu.Compose(
    [
        albu.HorizontalFlip(p=.5),
        albu.VerticalFlip(p=.5),
        albu.Transpose(p=0.5),
        #albu.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.3)
    ]
)


#train data
train_datagen = DataGenPanda(
    imgs_path=DATA_PATH+'/train_images',
    df=X_train, 
    batch_size=BATCH_SIZE,#BATCH_SIZE
    mode='fit', 
    shuffle=True, 
    aug=aug, 
    seq_len=SEQ_LEN, 
    img_size=IMG_SIZE, 
    n_classes=6
)

#test data
val_datagen = DataGenPanda(
    imgs_path=DATA_PATH+'/train_images', 
    df=X_val, 
    batch_size=BATCH_SIZE,#BATCH_SIZE,
    mode='fit', 
    shuffle=False, 
    aug=None, 
    seq_len=SEQ_LEN, 
    img_size=IMG_SIZE, 
    n_classes=6
)


# In[ ]:


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

# In[12]:



pretrain_model = 'model_history/efficientnet/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
#pretrain_model = '../input/panda-efficientnet/efnb3_epoch3_vloss_2.49_vcal_acc_0.36.h5'
bottleneck = efn.EfficientNetB0(
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
# model.add(BatchNormalization())
# model.add(Dropout(.2))
model.add(Dense(512, activation='elu'))
model.add(BatchNormalization())
model.add(Dropout(.2))
model.add(Dense(6, activation='linear'))
model.summary()


# In[13]:


#qudratic weighted kappa score
def kappa_keras(y_true, y_pred):

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


# In[14]:


def kappa_score(y_true, y_pred):
    
    y_true=tf.math.argmax(y_true)
    y_pred=tf.math.argmax(y_pred)
    #qwk = cohen_kappa_score(y_true, y_pred,weights='quadratic')
    #print(f'qwk : {qwk}')
    return tf.compat.v1.py_func(cohen_kappa_score ,(y_true, y_pred),tf.double)


# In[15]:


model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=5e-4),
    #metrics=['binary_crossentropy',QuadraticWeightedKappa()]
    metrics=['binary_crossentropy',kappa_keras]
)


# In[ ]:


# model.load_weights('../input/panda-efficientnet/model_v7_epoch_28_vloss_1.93.h5')


# In[ ]:


get_ipython().run_cell_magic('time', '', "model_file = 'model_'+MDL_VERSION+'_epoch_{epoch:02d}_vloss_{val_loss:.1f}.h5'\n#model_file = 'model_'+MDL_VERSION+'_epoch__vloss_.h5'\nif False:\n    model = load_model(model_file)\n    print('model loaded')\nelse:\n    print('train from scratch')\nEPOCHS = 10\nearlystopper = EarlyStopping(\n    monitor='val_loss', \n    patience=10, \n    verbose=1,\n    mode='min'\n)\nmodelsaver = ModelCheckpoint(\n    model_file, \n    monitor='val_loss', \n    verbose=1, \n    save_best_only=True,\n    mode='min'\n)\nlrreducer = ReduceLROnPlateau(\n    monitor='val_loss',\n    factor=.1,\n    patience=5,\n    verbose=1,\n    min_lr=5e-7\n)\nhistory = model.fit_generator(\n    train_datagen,\n    #steps_per_epoch= 3,\n    validation_data=val_datagen,\n    #validation_steps = 3,\n    class_weight=class_weights,\n    callbacks=[earlystopper, modelsaver, lrreducer],\n    epochs=EPOCHS,\n    verbose=1\n)")


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
plt.plot(history.history['binary_crossentropy'][:ep_max], label='binary_crossentropy')
plt.plot(history.history['val_binary_crossentropy'][:ep_max], label='val_binary_crossentropy')
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




