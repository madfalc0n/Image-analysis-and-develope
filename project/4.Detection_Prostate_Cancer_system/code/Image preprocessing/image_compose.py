#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
from functools import reduce
import cv2
import pandas as pd
import skimage.io
import numpy as np
import matplotlib.pyplot as plt


# In[16]:


SAVE = '/home/lab05/kaggle_dir/test'
TRAIN = '/home/lab03/kaggle_project/dataset/train_images/'
MASK = '/home/lab03/kaggle_project/dataset/train_label_masks/'


BLACK = (0,) * 3
GRAY = (200,) * 3
WHITE = (255,) * 3
RED = (255, 0, 0)

SIZE = 128
N = 12


# In[17]:


def tile(img, sz=128,N=12):
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


# In[18]:


train = pd.read_csv('/home/lab05/kaggle_dir/train.csv')
test = pd.read_csv('/home/lab05/kaggle_dir/test.csv')


# In[19]:


print(len(train))


# In[20]:


del_image = pd.read_csv('/home/lab05/kaggle_dir/PANDA_Suspicious_Slides.csv')
a = del_image['image_id']
b = list(a)


# In[21]:


for i in range(len(train)):
    if train['image_id'][i] in b:
        train.drop(i,inplace=True)
train.reset_index(drop=False,inplace=True)


# In[8]:


len(train)


# In[22]:


#원본데이터 증식

index = 2

names = train[train['isup_grade'] == index]
names = list(names['image_id'])

print(names[0])


# In[25]:


#원본데이터 증식

index = 2

names = train[train['isup_grade'] == index]
names = list(names['image_id'])


img = skimage.io.MultiImage(os.path.join(TRAIN,names[0]+'.tiff'))[-1]
# img = cv2.flip(img, 0)       

cv2.imwrite("/home/lab05/kaggle_dir/test/1231.png", img)
         

        
    
        
    

            


# In[16]:


#원본데이터 증식

index = 2
while index < 6:
    names = train[train['isup_grade'] == index]
    names = list(names['image_id'])
#     names=[name[:-5] for name in os.listdir(TRAIN)]
#     index += 1
    a = 0
        # with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:
    #     for i in range(0,1000,2):
    for i in range(len(names)):
        

        a+=1
        print(a)
        if a < len(names)//2 :

            img = skimage.io.MultiImage(os.path.join(TRAIN,names[i]+'.tiff'))[-1]
            img = cv2.flip(img, 0)       

            cv2.imwrite("/home/lab05/kaggle_dir/test_image/"+f'{i}_{index}_v2.png', img)
            new_data = {'image_id':f'{i}_{index}_v2','isup_grade':f'{index}'}
            train = train.append(new_data,ignore_index =True)
        else :
            img = skimage.io.MultiImage(os.path.join(TRAIN,names[i]+'.tiff'))[-1]
            img = cv2.flip(img, 1)       

            cv2.imwrite("/home/lab05/kaggle_dir/test_image/"+f'{i}_{index}_v2.png', img)
            new_data = {'image_id':f'{i}_{index}_v2','isup_grade':f'{index}'}
            train = train.append(new_data,ignore_index =True)
    
    index += 1


        
    
        
    

            


# In[ ]:


index = 3
while index < 6:
    names = train[train['isup_grade'] == index]
    names = list(names['image_id'])
#     names=[name[:-5] for name in os.listdir(TRAIN)]
    index += 1
    a = 0
        # with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:
    #     for i in range(0,1000,2):
    for i in range(len(names)):

        a+=1

        img = skimage.io.MultiImage(os.path.join(TRAIN,names[i]+'.tiff'))[-1]    
    #         img2 = skimage.io.MultiImage(os.path.join(TRAIN,names[i+2]+'.tiff'))[-1]
        image = tile(img)
    #         image2 = tile(img2)
        image = cv2.hconcat([cv2.vconcat([image[11], image[10], image[9]]), 
                             cv2.vconcat([image[8], image[7], image[6]]), 
                             cv2.vconcat([image[5], image[4], image[3]]), 
                             cv2.vconcat([image[2], image[1], image[0]]),
                            ])

    #         image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
    #                              cv2.vconcat([image2[0], image2[1], image2[2], image2[3]]), 
    #                              cv2.vconcat([image[4], image[5], image[6], image[7]]),
    #                              cv2.vconcat([image2[4], image2[5], image2[6], image2[7]])])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image.shape)
        cv2.imwrite("/home/lab09/kaggle_dir/image_128_2_12/"+f'{i}_{index}_v2.png', image)
        new_data = {'image_id':f'{i}_{index}_v2','isup_grade':f'{index}'}
        train = train.append(new_data,ignore_index =True)
        
        
    
        
    

            


# In[ ]:





# In[ ]:


index = 0
while index < 6:
    names = train[train['isup_grade'] == index]
    names = list(names['image_id'])
#     names=[name[:-5] for name in os.listdir(TRAIN)]
    index += 1
    a = 0
        # with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:
    #     for i in range(0,1000,2):
    for i in range(len(names)):

        a+=1

        img = skimage.io.MultiImage(os.path.join(TRAIN,names[i]+'.tiff'))[-1]    
    #         img2 = skimage.io.MultiImage(os.path.join(TRAIN,names[i+2]+'.tiff'))[-1]
        image = tile(img)
    #         image2 = tile(img2)
        image = cv2.hconcat([cv2.vconcat([image[0], image[4], image[8]]), 
                             cv2.vconcat([image[1], image[5], image[9]]), 
                             cv2.vconcat([image[2], image[6], image[10]]), 
                             cv2.vconcat([image[3], image[7], image[11]]),
                            ])

    #         image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
    #                              cv2.vconcat([image2[0], image2[1], image2[2], image2[3]]), 
    #                              cv2.vconcat([image[4], image[5], image[6], image[7]]),
    #                              cv2.vconcat([image2[4], image2[5], image2[6], image2[7]])])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image.shape)
        cv2.imwrite("/home/lab09/kaggle_dir/image_128_2_12/"+f'{i}_{index}_v3.png', image)
        new_data = {'image_id':f'{i}_{index}_v3','isup_grade':f'{index}'}
        train = train.append(new_data,ignore_index =True)

    
        
    

            


# In[ ]:


index = 0
while index < 6:
    names = train[train['isup_grade'] == index]
    names = list(names['image_id'])
#     names=[name[:-5] for name in os.listdir(TRAIN)]
    index += 1
    a = 0
        # with zipfile.ZipFile(OUT_TRAIN, 'w') as img_out:
    #     for i in range(0,1000,2):
    for i in range(len(names)):

        a+=1

        img = skimage.io.MultiImage(os.path.join(TRAIN,names[i]+'.tiff'))[-1]    
    #         img2 = skimage.io.MultiImage(os.path.join(TRAIN,names[i+2]+'.tiff'))[-1]
        image = tile(img)
    #         image2 = tile(img2)
        image = cv2.hconcat([cv2.vconcat([image[8], image[4], image[0]]), 
                             cv2.vconcat([image[9], image[5], image[1]]), 
                             cv2.vconcat([image[10], image[6], image[2]]), 
                             cv2.vconcat([image[11], image[7], image[3]]),
                            ])

    #         image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
    #                              cv2.vconcat([image2[0], image2[1], image2[2], image2[3]]), 
    #                              cv2.vconcat([image[4], image[5], image[6], image[7]]),
    #                              cv2.vconcat([image2[4], image2[5], image2[6], image2[7]])])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image.shape)
        cv2.imwrite("/home/lab09/kaggle_dir/image_128_2_12/"+f'{i}_{index}_v4.png', image)
        new_data = {'image_id':f'{i}_{index}_v4','isup_grade':f'{index}'}
        train = train.append(new_data,ignore_index =True)

    
        
    

            


# In[ ]:




