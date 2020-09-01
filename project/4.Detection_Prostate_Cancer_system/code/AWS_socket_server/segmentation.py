#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import detectron2
import torch, torchvision
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage import color

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode


# ## Variable Setting 

# In[2]:


PATH = '/home/lab03/kaggle_project/dataset/'
DATA_PATH = '/home/lab03/kaggle_project/dataset/train_images/'
MASK_DATA_PATH = '/home/lab03/kaggle_project/dataset/train_label_masks/'
PNG_PATH = '/home/lab03/kaggle_project/dataset/train_images_png/'
SEED = 2020


# ## Model config 

# In[3]:


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 


# In[4]:


def Predict_prostate(original_image):
    #gray sacle로 변환 및 전처리
    gray_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    thr, gray_image_2 = cv2.threshold(gray_image, 240, 255, cv2.THRESH_TRUNC)

    # segmentation 예측을 위한 차원 확장
    gray_image_2 = np.expand_dims(gray_image_2,axis=2)
    gray_image_2 = gray_image_2*np.ones((1,1,3)).astype('uint8')
    #print(gray_image_2.shape)

    #zero_image - (x,x,3)
    zero_image = np.ones_like(gray_image)
    zero_image = np.where(zero_image==1, 0, zero_image)
    #print(zero_image.shape)

    value = 31000
    for i in range(2):
        print(f'INDEX : {i}')
        model_info = 'model_iter_'+str(value)

        print(f'load model weights : Training Iteration {value} ')
        model_path = '/home/lab03/kaggle_project/dataset/segmentation_data/output'
        #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_info+".pth")        
        cfg.MODEL.WEIGHTS = model_path + model_info+".pth"        
        predictor = DefaultPredictor(cfg)

        # 결과 예측
        print('Predict Image')
        outputs = predictor(gray_image_2)
        #print(len(outputs["instances"].pred_masks))

        # GPU 처리를 CPU로 변환 후 예측한 마스크 값을 넘파이 형식으로 추출
        mask = outputs['instances'].to('cpu').pred_masks.numpy()

        # 마스크 값들 중 'False'인 항목은 0으로 , 'True'인 항목은 1로 변경
        mask = np.where(mask==False, 0, mask)
        mask = np.where(mask==True, 1, mask)    

        for i in range(len(mask)):
            # or 연산, 나뉘어져 있는 마스크영역을, zero_image 한곳에 모음
            zero_image = np.bitwise_or(zero_image,mask[i])

        print('Predict complete!')
        print('---------------------------------------')
        value += 5000


    # 마스크 영역 표시할 이미지(cpy_image) 생성
    cpy_image = np.copy(original_image)
    #마스크 좌표 생성, 'cv2.RETR_EXTERNAL'옵션을 이용해 가장 바깥쪽 라인만 생성
    # contour 옵션은 https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html
    contours, hierarchy = cv2.findContours(zero_image.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        cv2.drawContours(cpy_image, [cnt], 0, [0,255,255], 3)
    
    return cpy_image


# In[6]:

def seg(filepath):

    #image 호출, 소켓프로그래밍 통해 받을 예정
    #image_name = 'ffe06afd66a93258f8fabdef6044e181'
    #original_image = cv2.imread(PNG_PATH+image_name+'.png')
    original_image = cv2.imread(filepath)

    result_img = Predict_prostate(original_image)
    
    #cv2.imwrite('complete.png',result_img)
    return result_img 


# In[ ]:





# In[ ]:





# In[ ]:



    

