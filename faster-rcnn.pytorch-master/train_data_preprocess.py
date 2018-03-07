#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:24:31 2018

@author: zkq
"""

import os
import numpy as np
from scipy import ndimage, misc, sparse
#import imageio
import matplotlib.pyplot as plt
import csv
import pickle
from datasets.science import science
from model.utils.net_utils import vis_detections
import cv2

datapath = 'data/science/train'

imagepath = os.path.join(datapath, 'image')
csvpath = os.path.join(datapath, 'stage1_train_labels.csv')

filelist = os.listdir(imagepath)
filelist.sort()
imginfo = dict()

saveimg = True
savepath = "save/science/"

for imageid in filelist:
    thisimgpath = os.path.join(imagepath, imageid, 'images')
    img = misc.imread(os.path.join(thisimgpath, os.listdir(thisimgpath)[0]))
    imginfo[imageid] = {'size': img.shape, 
                        'boxes': [], 
                        'gt_classes': [],
                        'gt_ishard': [],
                        'gt_overlaps': [],
                        'flipped': False,
                        'seg_areas': []}

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
        imginfo[imageid]['boxes'].append(roi)
        imginfo[imageid]['gt_classes'].append(1)
        imginfo[imageid]['gt_ishard'].append(False)
        imginfo[imageid]['gt_overlaps'].append([0.0, 1.0])
        imginfo[imageid]['seg_areas'].append((xmax - xmin + 1) * (ymax - ymin + 1))

roidb = [None] * len(filelist)
for ii in range(len(filelist)):
    roidb[ii] = dict()
    roidb[ii]['boxes']       = np.array(imginfo[filelist[ii]]['boxes'], dtype=np.uint16)
    roidb[ii]['gt_classes']  = np.array(imginfo[filelist[ii]]['gt_classes'], dtype=np.int32)
    #roidb[ii]['gt_ishard']   = np.array(imginfo[filelist[ii]]['gt_ishard'])
    roidb[ii]['gt_overlaps'] = sparse.csr_matrix(np.array(imginfo[filelist[ii]]['gt_overlaps'], dtype=np.float32))
    roidb[ii]['flipped'] = imginfo[filelist[ii]]['flipped']
    #roidb[ii]['seg_areas']   = np.array(imginfo[filelist[ii]]['seg_areas'])

f = open(os.path.join(datapath, 'stage1_train_labels_pickle.txt'), 'wb')
pickle.dump(roidb, f)
f.close

if (saveimg):
    dataset = science('train')
    roidb = dataset.gt_roidb()
    for ii in range(len(roidb)):
        impath = dataset.image_path_at(ii)
        im = cv2.imread(impath)
        if im.shape[2] == 4:
            im = im[:,:,0:3]
        #im = im[:,:,::-1]
        nbox = len(roidb[ii]['boxes'])
        dets = np.zeros([nbox, 5])
        dets[:, 0:4] = roidb[ii]['boxes']
        dets[:, 4] = 1
        im = vis_detections(im, '', dets)
        cv2.imwrite(savepath + dataset.image_id_at(ii) + '.png', im)
        
        
        
        
        

