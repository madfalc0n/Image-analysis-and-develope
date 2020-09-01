#!/usr/bin/env python
# coding: utf-8

# # Visualization: PANDA 16x128x128 tiles

# ### This notebook:
# 
# - is based on [PANDA 16x128x128 tiles][original]. Thank you [@iafoss][iafoss] for sharing it with us.
# - visualizes which tiles are selected in iafoss's approach.
# 
# [iafoss]: https://www.kaggle.com/iafoss
# [original]: https://www.kaggle.com/iafoss/panda-16x128x128-tiles

# In[10]:


import os
from functools import reduce

import pandas as pd
import skimage.io
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


# INPUT_DIR = "../input/prostate-cancer-grade-assessment"
TRAIN_DIR = '/home/lab03/kaggle_project/dataset/train_images/'
# MASK_DIR = f"{INPUT_DIR}/train_label_masks"

BLACK = (0,) * 3
GRAY = (200,) * 3
WHITE = (255,) * 3
RED = (255, 0, 0)

SIZE = 128
N = 16


# In[12]:


train = pd.read_csv('/home/lab05/kaggle_dir/train.csv')
train.head()


# # Load image

# In[13]:


def imread(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: '{path}'")

    return skimage.io.MultiImage(path)


def imshow(
    img,
    title=None,
    show_shape=True,
    figsize=(8, 8)
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.grid("off")
    ax.set_xticks([])
    ax.set_yticks([])

    if show_shape:
        ax.set_xlabel(f"Shape: {img.shape}", fontsize=16)
        
    if title:
        ax.set_title(title, fontsize=16)

    return ax


# In[32]:


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


# In[36]:


img_id = "0018ae58b01bdadc8e347995b69f99aa"
image = skimage.io.MultiImage(os.path.join(TRAIN_DIR, f"{img_id}.tiff"))[-1]        
image = tile(image)
image = cv2.hconcat([cv2.vconcat([image[0], image[1], image[2], image[3]]), 
                     cv2.vconcat([image[4], image[5], image[6], image[7]]), 
                     cv2.vconcat([image[8], image[9], image[10], image[11]]),
                     cv2.vconcat([image[12], image[13], image[14], image[15]])] 
                     )
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imshow(image, "Original image")
    
 


# In[15]:


import cv2
img_id = "0018ae58b01bdadc8e347995b69f99aa"
img_org = imread(os.path.join(TRAIN_DIR, f"{img_id}.tiff"))[1]
# img_org = image = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
print(img_org.shape)
imshow(img_org, "Original image")


# # Padding
# 
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html

# In[17]:


H, W = img_org.shape[:2]
pad_h = (SIZE - H % SIZE) % SIZE
pad_w = (SIZE - W % SIZE) % SIZE

print("pad_h:", pad_h)
print("pad_w", pad_w)


# In[19]:


padded_vis = np.pad(
    img_org,
    [[pad_h // 2, pad_h - pad_h // 2],
     [pad_w // 2, pad_w - pad_w // 2],
     [0, 0]],
    constant_values=GRAY[0],  # use GRAY for visualization.
)

imshow(padded_vis, "Padded image")


# In[20]:


padded = np.pad(
    img_org,
    [[pad_h // 2, pad_h - pad_h // 2],
     [pad_w // 2, pad_w - pad_w // 2],
     [0, 0]],
    constant_values=WHITE[0],
)


# In[21]:


N_ROWS = padded.shape[0] // SIZE
N_COLS = padded.shape[1] // SIZE

print("N_ROWS :", N_ROWS)
print("N_COLS :", N_COLS)
print("N_TILES:", N_ROWS * N_COLS)


# # Create tiles
# 
# - https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
# - https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html

# In[22]:


reshaped = padded.reshape(
    padded.shape[0] // SIZE,
    SIZE,
    padded.shape[1] // SIZE,
    SIZE,
    3,
)
transposed = reshaped.transpose(0, 2, 1, 3, 4)
tiles = transposed.reshape(-1, SIZE, SIZE, 3)

print("reshaped.shape  :", reshaped.shape)
print("transposed.shape:", transposed.shape)
print("tiles.shape     :", tiles.shape)


# # Visualize tiles

# In[24]:


def merge_tiles(tiles, funcs=None):
    """
    If `funcs` specified, apply them to each tile before merging.
    """
    return np.vstack([
        np.hstack([
            reduce(lambda acc, f: f(acc), funcs, x) if funcs else x
            for x in row
        ])
        for row in tiles
    ])


def draw_borders(img):
    """
    Put borders around an image.
    """
    ret = img.copy()
    ret[0, :] = GRAY   # top
    ret[-1, :] = GRAY  # bottom
    ret[:, 0] = GRAY   # left
    ret[:, -1] = GRAY  # right
    return ret


# In[25]:


imshow(merge_tiles(transposed, [draw_borders]), "Tiles")


# # Select tiles

# In[26]:


import re
sums = tiles.reshape(tiles.shape[0], -1).sum(axis=-1)


highlight = lambda x: "color: {}".format("red" if x != sums.max() else "black")
pd.DataFrame(sums.reshape(N_ROWS, N_COLS)).style.applymap(highlight)


# In[27]:


idxs_selected = np.argsort(sums)[:N]
idxs_selected


# # Visuzalize selected tiles

# In[28]:


def fill_tiles(tiles, fill_func):
    """
    Fill each tile with another array created by `fill_func`.
    """
    return np.array([[fill_func(x) for x in row] for row in tiles])


def make_patch_func(true_color, false_color):
    def ret(x):
        """
        Retunrs a color patch. The color will be `true_color` if `x` is True otherwise `false_color`.
        """
        color = true_color if x else false_color
        return np.tile(color, (SIZE, SIZE, 1)).astype(np.uint8)

    return ret


# In[29]:


mask = np.isin(np.arange(len(sums)), idxs_selected).reshape(N_ROWS, N_COLS)
mask = fill_tiles(mask, make_patch_func(WHITE, BLACK))
mask = merge_tiles(mask, [draw_borders])

imshow(mask, "Selected tiles")


# In[30]:


mask = np.isin(np.arange(len(sums)), idxs_selected).reshape(N_ROWS, N_COLS)
mask = fill_tiles(mask, make_patch_func(RED, WHITE))
mask = merge_tiles(mask, [draw_borders])

with_mask = np.ubyte(0.7 * padded + 0.3 * mask)

imshow(with_mask, "Selected tiles")


# In[ ]:




