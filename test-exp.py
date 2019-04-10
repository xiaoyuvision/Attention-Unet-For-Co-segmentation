# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:08:26 2019

@author: Administrator
"""
import cv2
from module import Net

import numpy as np
import torch
import torch.autograd.Variable as V


model= Net().cuda()
model.load_state_dict(torch.load('myweights.pth'))

img1=cv2.imread('traindata/imgs/YL1_2019_03_19_11_00_18_269_1122_ImgMatting.BMP')

img2=cv2.imread('traindata/imgs/YL1_2019_03_19_13_39_59_828_6128_ImgMatting.BMP')
shape=(768,768)
img1=cv2.resize(img1,shape).transpose(2,0,1)
    
    
img1 = np.array(img1, np.float32)/255.0 * 3.2 - 1.6

img2=cv2.resize(img2,shape).transpose(2,0,1)
    
    
img2 = np.array(img2, np.float32)/255.0 * 3.2 - 1.6
img1 = V(torch.Tensor(img1).cuda()).unsqueeze(0)

img2 = V(torch.Tensor(img2).cuda()).unsqueeze(0)
out1,out2 = model(img1,img2)
out1=out1.squeeze().cpu().data.numpy()
out2=out2.squeeze().cpu().data.numpy()

cv2.imwrite('mask1.png',out1*255)
cv2.imwrite('mask1.png',out2*255)
