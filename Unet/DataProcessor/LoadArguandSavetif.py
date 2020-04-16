# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:40:38 2018

@author: Niny
"""

import random
import numpy as np
#import nibabel as nib
import os
#import tensorflow as tf
import tensorflow as tf
from PIL import Image
import cv2

#from Augmentation import contrast_normalization, data_argument
#from Processor import image_splite, image_merge, down_sample, data_argument,contrast_normalization
import Processor as pr

#WIDTH = 2580//2
#HEIGHT = 1944//2
#COLS = 4
#ROWS = 5
#SP_WIDTH = 258
#SP_HEIGHT = 243
WIDTH = 800
HEIGHT = 600
COLS = 4
ROWS = 5
SP_WIDTH = 160
SP_HEIGHT = 150

TRAIN, TEST = 'TRAIN', 'TEST'


train_inpath = 'M:/DATA/Cells/PSPTrain'
test_inpath = 'M:/DATA/Cells/PSPTest'
outpath = 'M:/Codes/Cells/160_150/'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_sub_folders(folder):
    return [sub_folder for sub_folder in os.listdir(folder) if os.path.isdir(os.path.join(folder, sub_folder))]

def data_trainsave(input_dir, output_dir):
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    

    filename='TrainSet.tfrecord'
    Trans = ['False', 'Mirror', 'Rot90', 'Rot-90', 'Displa','Rotate']
    dx=0
    dy=0
    theta=0
    writer = tf.python_io.TFRecordWriter(os.path.join(output_dir+filename))
    for itr in range(2):
        if itr>6:
            trans_mode = Trans[5]
        elif itr>4 and itr<=6:
            trans_mode = Trans[4]
        elif itr<=4:
            trans_mode = Trans[itr]      
        print('____________________'+trans_mode+'___________________________')
        for folder in os.listdir(input_dir):
            #img_path = folder+'/Image/'
            #lbl_path = folder+'/Label/'
            for sub_folder in os.listdir(input_dir+'/'+folder+'/'):
                #savenum=0
                #savename = filename+str(savenum)+'.tfrecord'
                #writer = tf.python_io.TFRecordWriter(os.path.join(output_dir+savename))
                img_folder = input_dir+'/'+folder+'/'+sub_folder+'/Image/'
                lbl_folder = input_dir+'/'+folder+'/'+sub_folder+'/Label/'
                print('****************'+sub_folder+'***********************')
                for imgname in os.listdir(img_folder):
                    lblname=imgname[:-3]+'bmp'
                    print(imgname+","+lblname)
                    img=cv2.imread(img_folder+imgname)
                    lbl=cv2.imread(lbl_folder+lblname)
                    #imgs =  np.dot(img[...,:3], [0.299, 0.587, 0.114])
                    imgs = pr.Intensity(img)
                    lbls =  np.dot(lbl[...,:3], [0.299, 0.587, 0.114])//255
                    lbls = lbls.astype('uint8')
                
                
                    img_batch = pr.down_sample(imgs).astype('float64')
                    lbl_batch = pr.down_sample(lbls)
                    
                    img_batch_norm=pr.contrast_normalization(img_batch)
                    if trans_mode=='Displa':
                        dx = random.randint(-40,40)
                        dy = random.randint(-40,40)
                        print('dx:'+str(dx)+'...dy:'+str(dy))
                    if trans_mode=='Rotate':
                        theta = random.randint(-50,50)
                        print('theta:'+str(dx))
                    img_all_batch = pr.data_argument(img_batch_norm, trans_mode, dx, dy, theta)
                    lbl_all_batch = pr.data_argument(lbl_batch, trans_mode, dx, dy, theta)

                    img_all = img_all_batch.reshape([1,HEIGHT,WIDTH,1])
                    lbl_all = lbl_all_batch.reshape([1,HEIGHT,WIDTH,1])
                
                    img_sp={}
                    lbl_sp={}
                    img_sp = pr.image_splite(img_all,COLS,ROWS)
                    lbl_sp = pr.image_splite(lbl_all,COLS,ROWS)
                
                    for col in range(COLS):
                        for row in range(ROWS):
                            print('~~~~~~~~'+str(col)+'~~~'+str(row)+'~~~~~~~~')
                            img_save_batch = img_sp[col][row]
                            lbl_save_batch = lbl_sp[col][row]
                        
                            img_save = img_save_batch.reshape([SP_HEIGHT,SP_WIDTH,1])
                            lbl_save = lbl_save_batch.reshape([SP_HEIGHT,SP_WIDTH,1]).astype('uint8')

                        
                            example = tf.train.Example(features=tf.train.Features(feature={
                                    'img_raw': _bytes_feature(img_save.tostring()),
                                    'gt_raw': _bytes_feature(lbl_save.tostring())}))
                            writer.write(example.SerializeToString())
    writer.close()
            #savenum=savenum+1
    
def data_testsave(input_dir, output_dir):
    filename='TestSet.tfrecord'
    writer = tf.python_io.TFRecordWriter(os.path.join(output_dir+filename))
    for folder in os.listdir(input_dir):
        print('****************'+folder+'***********************')
        img_path = input_dir+'/'+folder+'/Image/'
        lbl_path= input_dir+'/'+folder+'/Label/'
        for imgname in os.listdir(img_path):
            lblname=imgname[:-3]+'bmp'
            print(imgname+","+lblname)
            img=cv2.imread(img_path+imgname)
            lbl=cv2.imread(lbl_path+lblname)
            #imgs =  np.dot(img[...,:3], [0.299, 0.587, 0.114])
            imgs = pr.Intensity(img)
            lbls =  np.dot(lbl[...,:3], [0.299, 0.587, 0.114])//255
            lbls = lbls.astype('uint8')
        
            img_batch = pr.down_sample(imgs).astype('float64')
            lbl_batch = pr.down_sample(lbls)
            img_batch_norm=pr.contrast_normalization(img_batch)
        
            example = tf.train.Example(features=tf.train.Features(feature={
                    'img_raw': _bytes_feature(img_batch_norm.tostring()),
                    'gt_raw': _bytes_feature(lbl_batch.tostring())}))
            writer.write(example.SerializeToString())
    writer.close()
                
    
    
#data_trainsave(train_inpath, outpath)
data_testsave(test_inpath, outpath)