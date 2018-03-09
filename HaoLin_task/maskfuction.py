# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 07:09:44 2018

@author: Viktor Young
"""

import os
import cv2
from utils.probeImage import probeImage
from PIL import Image,  ImageDraw
import numpy as np
import pickle
import glob
import PIL.ImageOps
from skimage import data,img_as_float
import matplotlib.pyplot as plt
img_path = glob.glob(os.path.join("sample_mask/0ea221716cf13710214dcd331a61cea48308c3940df1d28cfc7fd817c83714e1.png"))[0]
coor_masks = pickle.load( open( "sample_mask/x1y1_x2y2_masks.p", "rb" ) )
def mask(img_path,coor_masks):
    xys = coor_masks["coor"]
    masks = coor_masks["mask"]
    img = Image.open(img_path).convert("RGB")
    img_arr = np.asarray(img, order='F')
    img_new=img_arr
    img_new[:,:,:]=255
    overall=img_arr
    overall[:,:,:]=0
    mask_sum=[]
    for i in range (0,len(xys)):
        #produce new mask
        print('processing maks'+str(i))
        img_new=img_arr
        img_new[:,:,:]=255
        bbox = xys[i]
        x1, y1, x2, y2 = bbox
        mask = masks[i]
        mask=abs(mask-1)
        img_new[x1:x2,y1:y2]=img_new[x1:x2,y1:y2]*mask    
        binary_mask=img_as_float(img_new)
        binary_mask=abs(binary_mask-1)
        t=img_as_float(overall)
        t=abs(t-1)
        test=t+ binary_mask
        test[test<2]=0
        test=test/2  
        binary_mask=binary_mask-test 
        mask_sum.append(np.array(binary_mask))  
        overall=overall+abs(img_new-255)
        plt.imshow(binary_mask)
    print('the overall mask:')
    plt.imshow(overall)
    
    return mask_sum
maskall=mask(img_path,coor_masks)