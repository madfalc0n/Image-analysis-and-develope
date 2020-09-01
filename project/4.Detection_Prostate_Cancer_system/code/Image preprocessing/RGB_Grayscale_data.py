#!/usr/bin/env python
# coding: utf-8

# ## Module import 

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cv2
import collections
from collections import OrderedDict

#사이킷런
import skimage.io
from skimage import color

#detectron 2
import torch, torchvision
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from tqdm.notebook import tqdm


# ## Define Variable

# In[2]:


PATH = 'dataset/'
DATA_PATH = 'dataset/train_images/'
MASK_DATA_PATH = 'dataset/train_label_masks/'
SEED = 2020


# ## Define Function

# In[3]:


#특정 색상을 변경해줌
def transform_img(ori_img,asis_channel,tobe_channel):
    """
    ori_img : 색상변경을 적용할 원본이미지(3채널의 rgb 또는 bgr 이미지)
    asis_channel : 변경하고자 하는 색상 ex: [0,0,0]
    tobe_channel : 변경할 색상 ex: [255,255,255]
    [0,0,0] 에 해당되는 색상을 [255,255,255]로 변경해줌
    """
    cpy_img = np.copy(ori_img)
    bgr_threshold = asis_channel
    # BGR 제한값과 같으면 256으로 변경
    thresholds = (ori_img[:,:,0] == bgr_threshold[0])& 
                (ori_img[:,:,1] == bgr_threshold[1])& 
                (ori_img[:,:,2] == bgr_threshold[2])
    cpy_img[thresholds] = tobe_channel
    
    return cpy_img


# In[4]:


# IMAGE R, G, B, GRAY 영역별로 분리해주는 함수

def preproc(org_list,color_list):
    result_dict = {}
    cnt_list = collections.Counter(org_list)
    max_val = np.max(list(cnt_list.values())) + 1
    for i in range(256):
        if cnt_list[i] == 0:
            result_dict[color_list + str(i)] = [0]
        else:
            result_dict[color_list + str(i)] =  [np.round(cnt_list[i] / max_val,9)]
    return result_dict

# RGB 이미지 전처리, 각 채널 별로 분리
def RGB_preproc(rgb_image):
    color_list = ['R_','G_','B_']
    color_df = pd.DataFrame()
    
    #흰색영역을 걸러내기위한 타입변환
    rgb_image = rgb_image.astype('uint16')
    #흰색영역 추출
    blue_threshold = 250
    green_threshold = 250
    red_threshold = 250
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    # BGR 제한값과 같으면 256으로 변경, 하얀영역 제거
    thresholds_1 = (rgb_image[:,:,0] >= bgr_threshold[0])&
                    (rgb_image[:,:,1] >= bgr_threshold[1])& 
                    (rgb_image[:,:,2] >= bgr_threshold[2])
    rgb_image[thresholds_1] = [256,256,256]
    
    # BGR 제한값과 같으면 256으로 변경, 완전 검은 영역 제거
#     thresholds_2 = (rgb_image[:,:,0] <= 50) \
#                 & (rgb_image[:,:,1] <= 50) \
#                 & (rgb_image[:,:,2] <= 50)
#     rgb_image[thresholds_2] = [256,256,256]

    #b,g,r 영역으로 분리
    b,g,r = cv2.split(rgb_image)

    #blue, green, red 분포별로 추출
    mhb = b.ravel() #blue
    mhb = mhb[mhb != 256]
    
    mhg = g.ravel() #green
    mhg = mhg[mhg != 256]
    
    mhr = r.ravel() #red
    mhr = mhr[mhr != 256]
    
    #plt.hist(mhb , 256, [0,255], color='b')
    hist_mhb = np.histogram(mhb, bins=range(256))
    #print(np.sum(hist_mhb[0][:50]), np.sum(hist_mhb[0][50:]))
    #plt.hist(mhg, 256, [0,255], color='g')
    hist_mhg = np.histogram(mhg, bins=range(256))
    #print(np.sum(hist_mhg[0][:50]), np.sum(hist_mhg[0][50:]))
    #plt.hist(mhr, 256, [0,255], color='r')
    hist_mhr = np.histogram(mhr, bins=range(256))
    #print(np.sum(hist_mhr[0][:50]), np.sum(hist_mhr[0][50:]))
    #plt.show()
    
    #검은색 마킹영역이 많으면 비어있는 데이터 프레임 반환
    if np.sum(hist_mhb[0][:50]) > np.sum(hist_mhb[0][50:])and np.sum(hist_mhg[0][:50])> np.sum(hist_mhg[0][50:])and np.sum(hist_mhr[0][:50])> np.sum(hist_mhr[0][50:]):
        
        return color_df
    
    #추출된 r,g,b 값 list에 저장
    total_color = [mhb,mhg,mhr]

    #임시 데이터 프레임 생성
    tmp_df = pd.DataFrame()
    for number in range(3):
        #preproc 함수에서 각 color 별 0 ~ 255에 대한 count 값(0~80000)들을 0 ~ 1로 변경
        color_dict = preproc(total_color[number],color_list[number])
        tmp_df2 = pd.DataFrame(color_dict)
        tmp_df = pd.concat([tmp_df, tmp_df2], axis=1)
    
    return tmp_df
    
def GRAY_preproc(gray_image):

    color_list = ['GRAY_']
    color_df = pd.DataFrame()
    
    gray_image = color.rgb2gray(gray_image)*255
    gray_image = gray_image.astype(np.int32)
    gray_ravel = gray_image.ravel()

    #print(gray_ravel)
    
    #흰색영역 제거
    non_255 = gray_ravel[gray_ravel != 255]
    
    hist_gray = np.histogram(non_255, bins=range(256))
    #검은색 마킹영역이 더 많으면 return -1
    if np.sum(hist_gray[0][:50]) > np.sum(hist_gray[0][50:]):
        return color_df
#     print(np.sum(hist_gray[0][:50]), np.sum(hist_gray[0][50:]))
#     plt.hist(non_255, 256, [0,255], color='black')
#     plt.show()
    
    total_color = [non_255]
    tmp_df = pd.DataFrame()
    color_dict = preproc(total_color[0],color_list[0])
    tmp_df2 = pd.DataFrame(color_dict)
    tmp_df = pd.concat([tmp_df, tmp_df2], axis=1)
    
    return tmp_df


# ## Data Load 

# In[5]:


train = pd.read_csv(PATH + 'train.csv')
#train.head(1)

# ISUP 1 이상인 애들만 추출
# 'radboud' 공급사의 데이터(분포 0~4)를 사용, Karolinska 는 분포가 0~2 밖에없음, 
radboud_isup = (train['data_provider'] == 'radboud') & (train['isup_grade'] >= 0)
radboud_df = train[radboud_isup]
radboud_df.reset_index(inplace=True,drop=True)
display(radboud_df.head(10))
# display(train.head(1))


# In[12]:


#테스트용
print(radboud_df.iloc[2594]['image_id'])
tmp_mask_image_file = MASK_DATA_PATH + radboud_df.iloc[2594]['image_id'] + '_mask.tiff'
tmp_mask_image_f = skimage.io.MultiImage(tmp_mask_image_file)[1]
plt.imshow(tmp_mask_image_f)


# In[13]:


result_df = pd.DataFrame()
for start in tqdm(range(2594,3870)):
    #이미지 번호
    image_index = start

    #이미지 호출위한 변수 설정
    image_file = DATA_PATH + radboud_df.iloc[image_index]['image_id'] + '.tiff'
    mask_image_file = MASK_DATA_PATH + radboud_df.iloc[image_index]['image_id'] + '_mask.tiff'

    #원본이미지 호출
    image = skimage.io.MultiImage(image_file)[1]

    #마스크이미지 호출, R채널에만 이미지 값이 있음
    try:
        mask_image = skimage.io.MultiImage(mask_image_file)[1]
    except IndexError as ie:
        print(f'{mask_image_file} : mask image not found.', ie)
        continue
    mask_cnt_list = list(np.unique(mask_image))
    mask_label = radboud_df['isup_grade'][image_index]
    image_provider = radboud_df['data_provider'][image_index]

    # Mask gleson score 값
    #print(f'mask unique : {np.unique(mask_image)}')

    # # image show
    # fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    # fig.suptitle(f"Provider : {image_provider}, ISUP_Grade : {mask_label}", fontsize=16)
    # #오리지날IMAGE
    # ax[0].imshow(image)
    # ax[0].set_title(f'Original Image')
    # #Mask IMAGE
    # ax[1].imshow(mask_image*30)
    # ax[1].set_title(f'Mask Image')
    # plt.show()
    # Gleason score 별로 show

    for i,value in enumerate(mask_cnt_list):
        #print(f"gleason score : {value}")

        # 마스크 이미지 copy
        cpy_mask_image = mask_image.copy()
        cov_mask_image = np.zeros_like(image)
        # 마스크 이미지에서 해당 gleason_score(value)값일 경우 1로 아닌경우 0으로 변경 , 
        thresholds_1 = (cpy_mask_image[:,:,0] != value)
        cov_mask_image[thresholds_1] = [0,0,0]
        thresholds_2 = (cpy_mask_image[:,:,0] == value)
        cov_mask_image[thresholds_2] = [1,1,1]

        # mask에 원본이미지 곱함 -> 값이 1이므로 원본이미지가 출력
        cov_mask_image = cov_mask_image * image 
        cov_mask_image = transform_img(cov_mask_image,[0, 0, 0],[255, 255, 255])


        # RGB 별로 전처리 
        rgb_img = np.copy(cov_mask_image)
        RGB_df = RGB_preproc(rgb_img)
        #display(RGB_df)

        # GRAY 전처리
        gray_img = np.copy(cov_mask_image)
        GRAY_df = GRAY_preproc(gray_img)
        #display(GRAY_df)

        if len(RGB_df) == 0 or len(GRAY_df) == 0:
            #print('black')
            continue

        rgb_gray_df = pd.concat([RGB_df,GRAY_df] , axis = 1)
        rgb_gray_df['image_id'] = radboud_df['image_id'][start]+'_'+str(value)
        rgb_gray_df['org_image_id'] = radboud_df['image_id'][start]
        rgb_gray_df['gleason_score'] = value
        result_df = pd.concat([result_df,rgb_gray_df])

        # 출력
        #ax[i].imshow(cov_mask_image)
        #ax[i].set_title(f'Gleason Score : {value}')
        #cv2.imwrite('test_tmp'+str(value)+'.png',cov_mask_image)

result_df.reset_index(inplace=True,drop=True)
result_df.to_csv('train_merge_rgb_gray_20200616_7.csv', index=False)

