#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:24:31 2018

@author: zkq
"""

import os
import numpy as np
from scipy import ndimage, misc
import imageio
import matplotlib.pyplot as plt
import csv
import pickle

datapath = 'data/science/train'

imagepath = os.path.join(datapath, 'image')
csvpath = os.path.join(datapath, 'stage1_train_labels.csv')

filelist = os.listdir(imagepath)
imginfo = dict()
for imageid in filelist:
    thisimgpath = os.path.join(imagepath, imageid, 'images')
    img = imageio.imread(os.path.join(thisimgpath, os.listdir(thisimgpath)[0]))
    imginfo[imageid] = {'size': img.shape, 'roi': []}

with open(csvpath, 'r') as f:
    reader = csv.reader(f)
    imgmasklist = list(reader)
    for imgmaskline in imgmasklist:
        imageid = imgmaskline[0]
        if (imageid == 'ImageId'):
            continue
        ptstr = imgmaskline[1].split(' ')
        ptnum = np.array([int(a) for a in ptstr])
        ptnum = ptnum.reshape([len(ptnum)//2, 2])
        
        ptlistx = []
        ptlisty = []
        imgshape = imginfo[imageid]['size']
        for n1 in ptnum:
            p1 = [(n1[0]-1)//imgshape[0], (n1[0]-1)%imgshape[0]]
            p2 = [(n1[0]+n1[1]-2)//imgshape[0], (n1[0]+n1[1]-2)%imgshape[0]]
            ptlistx.append(p1[0])
            ptlistx.append(p2[0])
            ptlisty.append(p1[1])
            ptlisty.append(p2[1])
            xmin = min(ptlistx)
            xmax = max(ptlistx)
            ymin = min(ptlisty)
            ymax = max(ptlisty)
            roi = [xmin, ymin, xmax, ymax]
            imginfo[imageid]['roi'].append(roi)

f = open(os.path.join(datapath, 'stage1_train_labels_pickle.txt'), 'wb')
pickle.dump(imginfo, f)
f.close
        
        

