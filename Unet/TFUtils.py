#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 05:11:10 2018

@author: Niny
"""

import tensorflow as tf
import numpy as np
import scipy.misc as misc
from six.moves import urllib
import scipy
from PIL import Image

import os

#IMAGE_SIZE=256
#IMAGE_HEIGHT=1944//2
#IMAGE_WIDTH=2580//2
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

recordPath = os.getcwd()+'/'


#Save images
def save_imgs(test_num, label_batch, pred_batch, itr):
    
    s_path = recordPath+'imgs/'
    if not os.path.exists(s_path):
        os.mkdir(s_path)
        
    savepath = s_path +'results'+str(itr)+'/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    label_img_mat = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    pred_img_mat = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    for i in range (IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            if label_batch[i, j] == 0:
                label_img_mat[i, j, 0] = label_img_mat[i, j, 1] =  label_img_mat[i, j, 2] = 128  # backgroud Gray
            if label_batch[i, j] == 1:
                label_img_mat[i, j, 0] = 255
                label_img_mat[i, j, 1] = 69
                label_img_mat[i, j, 2] = 0     # liver Red
                
            if pred_batch[i, j] == 0:
                pred_img_mat[i, j, 0] = pred_img_mat[i, j, 1] =  pred_img_mat[i, j, 2] = 128  # backgroud Gray
            if pred_batch[i, j] == 1:
                pred_img_mat[i, j, 0] = 255
                pred_img_mat[i, j, 1] = 69
                pred_img_mat[i, j, 2] = 0    # liver Red
    label_img_mat=np.uint8(label_img_mat)            
    pred_img_mat=np.uint8(pred_img_mat) 
#    scipy.misc.imsave(recordPath + 'imgs/' + '%d-mask.jpg' % (test_num), label_img_mat)
#    scipy.misc.imsave(recordPath + 'imgs/' + '%d-pred.jpg' % (test_num), pred_img_mat)
    
    scipy.misc.imsave(savepath + '%d-mask.bmp' % (test_num), label_img_mat)
    scipy.misc.imsave(savepath + '%d-pred.bmp' % (test_num), pred_img_mat)


#Accuracy_Measure
def rgb2Bin(img):
    res = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for i in range (IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            if img[i, j ,0] == 128 and img[i, j ,1] == 128 and img[i, j ,2] == 128:
                res[i,j] = 0
            if img[i, j ,0] == 255 and img[i, j ,1] == 69 and img[i, j ,2] == 0:
                res[i,j] = 1
    return res
    

def Accuracy_Measure(itr):
    print("Accuracy_Measure..........")
    res_path = recordPath+'imgs/'+'results'+str(itr)+'/'
    file_names = os.listdir(res_path)
    file_names.sort()
    res_num = len(file_names)
    
    gt_label = 0
    pr_label = 0
    TP = 0
#    TN = 0
    
    for img_i in range(0, res_num, 2):
        label_batch = np.array(Image.open(res_path + file_names[img_i]))
        pred_batch = np.array(Image.open(res_path + file_names[img_i+1]))
        
        label = rgb2Bin(label_batch)
        pred = rgb2Bin(pred_batch)
              
        gt_label = gt_label + np.count_nonzero(label == 1)
        pr_label = pr_label + np.count_nonzero(pred == 1)
        
        label_bool = (label == 1)
        pred_bool = (pred == 1)
        common = np.logical_and(label_bool, pred_bool)
        TP = TP + np.count_nonzero(common == True)
        
#        label_gro = (label == 0)
 #       pred_gro = (pred == 0)
  #      common2 = np.logical_and(label_gro, pred_gro)
   #     TN = TN + np.count_nonzero(common2 == True)
        #allpix = IMAGE_SIZE*IMAGE_SIZE*res_num//2
        
        
    dice_coe = 2*TP/(gt_label + pr_label)
    #MIOU = TP/(gt_label + pr_label-TP)
    #Pixel_Acc = (TP + TN)/allpix
    
  
    print("DSC:", dice_coe)
   # print("MIOU:", MIOU)
    #print("PA:", Pixel_Acc)
   # print("GroundTruth_label", gt_label)
    #print("Predict", pr_label)
   # print("labPred", TP)
    return dice_coe

