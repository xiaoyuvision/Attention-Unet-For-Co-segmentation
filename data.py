# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:42:26 2019

@author: Administrator
"""

"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def default_loader(id, img_root,mask_root,resize_shape):
    img = cv2.imread(os.path.join(img_root,'{}.BMP').format(id[0]))
    img=cv2.resize(img,resize_shape)
    mask = cv2.imread(os.path.join(mask_root,'{}.png').format(id[0]), cv2.IMREAD_GRAYSCALE)
    
    mask=cv2.resize(mask,resize_shape)
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))
    
    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    #mask[mask>=0.5] = 1
    #mask[mask<=0.5] = 0
    #mask = abs(mask-1)
    ######################################################################
    
    
    img_c = cv2.imread(os.path.join(img_root,'{}.BMP').format(id[1]))
    img_c=cv2.resize(img_c,resize_shape)
    mask_c = cv2.imread(os.path.join(mask_root,'{}.png').format(id[1]), cv2.IMREAD_GRAYSCALE)
    
    mask_c=cv2.resize(mask_c,resize_shape)
    img_c = randomHueSaturationValue(img_c,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))
    
    img_c, mask_c = randomShiftScaleRotate(img_c, mask_c,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img_c, mask_c = randomHorizontalFlip(img_c, mask_c)
    img_c, mask_c = randomVerticleFlip(img_c, mask_c)
    img_c, mask_c = randomRotate90(img_c, mask_c)
    
    mask_c = np.expand_dims(mask_c, axis=2)
    img_c = np.array(img_c, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask_c = np.array(mask_c, np.float32).transpose(2,0,1)/255.0
    
    
    
    
    
    
    return img,img_c,mask,mask_c

class ImageFolder(data.Dataset):

    def __init__(self,img_root,mask_root,resize_shape):
        self.img_root=img_root
        self.mask_root=mask_root
        self.resize_shape=resize_shape
        ids_1=set()
        ids_2=set()
        imgs_id=os.listdir(img_root)
        masks_id=os.listdir(mask_root)
        for i in imgs_id:
            if i[0]!='.':
                ids_1.add(i.split('.')[0])
        for j in masks_id:
            if j[0]!='.':
                ids_2.add(j.split('.')[0])
        ids_all_x=list(ids_1.intersection(ids_2))
        ids_all_y=ids_all_x
        np.random.shuffle(ids_all_x)
        
        
        self.ids = list(zip(ids_all_x,ids_all_y))
#        print('.......',ids)
        self.loader = default_loader

    def __getitem__(self, index):
        id = self.ids[index]
        img1,img2,mask1,mask2 = self.loader(id, self.img_root,self.mask_root,self.resize_shape)
        img1 = torch.Tensor(img1)
        mask1 = torch.Tensor(mask1)
        img2 = torch.Tensor(img2)
        mask2 = torch.Tensor(mask2)
        
        return img1,img2,mask1,mask2

    def __len__(self):
        return len(self.ids)
