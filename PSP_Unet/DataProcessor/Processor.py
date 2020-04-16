# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 09:32:13 2018

@author: Niny
"""


import numpy as np
import os
#import tensorflow as tf
import tensorflow as tf
from PIL import Image
import cv2


#SP_WIDTH = 48
#SP_HEIGHT = 36
#WIDTH = 1296
#HEIGHT = 972

WIDTH = 800
HEIGHT = 600
SP_WIDTH = 160
SP_HEIGHT = 150



def contrast_normalization(image, min_divisor=1e-3):
    """
    Data normalization
    
     output = (input-mean)/Standard_deviation
    
    """
    mean = image.mean()
    std = image.std()
    if std < min_divisor:
        std = min_divisor
    return (image - mean) / std

def data_argument(image, mode, dx, dy ,theta):

    if mode == 'False':
        res = image
        
    if mode=='Mirror':
        #Im = Image.fromarray(image)
        #res = Im.transpose(Image.FLIP_LEFT_RIGHT)
        res = cv2.flip(image, 1)
        res = np.array(res)
          
    if mode == 'Rot90':
        rows,cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        res = cv2.warpAffine(image,M,(cols,rows))
        res = np.array(res)
    
    if mode == 'Rot-90':
        rows,cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
        res = cv2.warpAffine(image,M,(cols,rows))
        res = np.array(res)
        
    if mode == 'Displa':
        tran = np.float32([[1,0,dx],[0,1,dy]])
        rows,cols = image.shape[:2]
        res = cv2.warpAffine(image,tran,(cols,rows))
        res = np.array(res)
    if mode =='Rotate':
        rows,cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)
        res = cv2.warpAffine(image,M,(cols,rows))
        res = np.array(res)
    return res

def Intensity(img):
    img = img.astype('float64')
    res = (img[...,0]+img[...,1]+img[...,2])/3
    return res

def CLAHE(imgs):
    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(8,8))
    res = clahe.apply(imgs)
    return res


def image_splite(image, cols, rows):
    img={}
    for i in range(cols):
        img[i]={}
        for j in range(rows):
            img[i][j]=image[:,i*SP_HEIGHT:(i+1)*SP_HEIGHT,j*SP_WIDTH:(j+1)*SP_WIDTH,:]
    return img
    
def image_merge(image, cols, rows,chan):
    img_batch = np.zeros([1,HEIGHT,WIDTH,chan]).astype('uint8')
    for i in range(cols):
        for j in range(rows):
            img_batch[0,i*SP_HEIGHT:(i+1)*SP_HEIGHT,j*SP_WIDTH:(j+1)*SP_WIDTH,:]=image[i][j]          
    return img_batch
    

def down_sample(image):
    res = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    return res