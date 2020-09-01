#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.getcwd())


# In[2]:


import os
from functools import reduce
import zipfile
import pandas as pd
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm


# In[3]:


TRAIN = '/home/lab03/kaggle_project/dataset/train_images/'
# TRAIN = '/home/lab05/kaggle_dir/test/'
# MASKS = '../input/prostate-cancer-grade-assessment/train_label_masks/'
SAVE = '/home/lab05/kaggle_dir/test_image'
# OUT_TRAIN = 'train.zip'


# In[4]:


def tile(img, sz=128,N=16):
    count = 0
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                 constant_values=255)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
#     img = img.reshape(img.shape[0], -1).sum(axis=-1)
#     for i in range(len(img)):
#         if img[i] < 48000000 :
#             count += 1
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    

#     idxs = len(idxs)
    img = img[idxs]
    return img


# In[4]:


try:
    if not(os.path.isdir(SAVE)):
        os.makedirs(os.path.join(SAVE))
except OSError as e:
    print("error")
    
    


# In[5]:


names=[name[:-5] for name in os.listdir(TRAIN)]
for name in names :
    print(name)


# In[6]:


x_tot,x2_tot = [],[]
names=[name[:-5] for name in os.listdir(TRAIN)]
a = 0
# with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:
for name in names:
    print(a)
    a+=1
    
    image = skimage.io.MultiImage(os.path.join(TRAIN,name+'.tiff'))[1]        
#     image = tile(img)
#     image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
#                          cv2.vconcat([image[4], image[5], image[6], image[7]]), 
#                          cv2.vconcat([image[8], image[9], image[10], image[11]]),
#                          cv2.vconcat([image[12], image[13], image[14], image[15]])] 
#                          )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("/home/lab05/kaggle_dir/test_image/"+f'{name}.png', image)
    
    if a == 100 :
        break
    
            


# In[4]:




image = skimage.io.MultiImage('/home/lab03/kaggle_project/dataset/train_images/000920ad0b612851f8e01bcc880d9b3d.tiff')[0]        
#     image = tile(img)
#     image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
#                          cv2.vconcat([image[4], image[5], image[6], image[7]]), 
#                          cv2.vconcat([image[8], image[9], image[10], image[11]]),
#                          cv2.vconcat([image[12], image[13], image[14], image[15]])] 
#                          )
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("/home/lab05/kaggle_dir/test/123.png", image)

            

