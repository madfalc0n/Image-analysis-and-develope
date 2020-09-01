#!/usr/bin/env python
# coding: utf-8

# # Detectron2: Prostate Cancer Instance Segmentation
# <img src="https://dl.fbaipublicfiles.com/detectron2/Detectron2-Logo-Horz.png" width="500">

# In[ ]:

# detectron2 설치
# install dependencies: (use cu100 because colab is on CUDA 10.0)
!pip install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html 
!pip install cython pyyaml==5.1
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
import torch, torchvision
torch.__version__
!gcc --version
# opencv is pre-installed on colab
# install detectron2:
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html


# ## Get data

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

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


# In[2]:


!wget https://github.com/TannerGilbert/Detectron2-Train-a-Instance-Segmentation-Model/raw/master/microcontroller_segmentation_data.zip
!unzip microcontroller_segmentation_data.zip
get_ipython().system('ls')


# In[3]:


get_ipython().system("ls 'Prostate_cancer_data'")


# Detectron2에 Data 등록

import os
import numpy as np
import json
from detectron2.structures import BoxMode

def get_prostate_dicts(directory):
    #classes = ['3', '4', '5']
    classes = ['5']
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        record["file_name"] = filename+'.png'
        record["height"] = img_anns["height"]
        record["width"] = img_anns["width"]
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                #"category_id": classes.index(anno['label']),
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "test"]:
    DatasetCatalog.register("prostate_" + d, lambda d=d: get_prostate_dicts('Prostate_cancer_data/' + d))
    #MetadataCatalog.get("prostate_" + d).set(thing_classes=['3', '4', '5'])
    MetadataCatalog.get("prostate_" + d).set(thing_classes=['5'])
prostate_metadata = MetadataCatalog.get("prostate_train")


# 데이터 등록여부 확인
import random
dataset_dicts = get_prostate_dicts("Prostate_cancer_data/train")


# In[6]:


for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    print(d["file_name"])
    v = Visualizer(img[:, :, ::-1], metadata=prostate_metadata, scale=1.0)
    v = v.draw_dataset_dict(d)
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    cv2.imwrite('example_img.png',v.get_image()[:, :, ::-1])
    plt.show()


# 모델 Training

# In[7]:


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


# In[ ]:

# model 학습
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("prostate_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

# model 학습2
value = 1000
for index in range(6):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("prostate_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = 'output/model_iter_'+str(value)+'.pth'
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    value += 5000
    os.rename('output/model_final.pth', 'output/model_iter_'+str(value)+'.pth')
    os.rename('output/metrics.json', 'output/metrics_iter_'+str(value)+'.json')
    


# 학습모델 호출
model_info = 'model_iter_11000'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_info+".pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.DATASETS.TEST = ("prostate_test", )
predictor = DefaultPredictor(cfg)


# test 데이터 등록
from detectron2.utils.visualizer import ColorMode
dataset_dicts_test = get_prostate_dicts('Prostate_cancer_data/test')

# 학습모델 가중치별 예측
value = 1000
for i in range(11):
    model_info = 'model_iter_'+str(value)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_info+".pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.DATASETS.TEST = ("prostate_test", )
    predictor = DefaultPredictor(cfg)

    im = cv2.imread('Prostate_cancer_data/test/b2dca2c953d270bc78478df8cf5ddb62.png')
    tmp_str = model_info+'_b2dca2c953d270bc78478df8cf5ddb62.png'
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=prostate_metadata, 
                   scale=0.8, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    cv2.imwrite('result/'+tmp_str, cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()
    value += 5000

