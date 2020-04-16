# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:01:50 2018

@author: mnlist
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:15:33 2018


@author: Niny
"""

from DataProcessor import Processor as data_pro

import tensorflow as tf 
import numpy as np
from scipy import misc
import scipy

import TestUtils as testutils
#from UNets import Unet_Res as net
from Nets import Psp_Unet as net

import os
import cv2
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


recordPath = os.getcwd()+'/'


# WIDTH = 1290
# HEIGHT = 972
# ROWS = 5
# COLS = 4

# SP_WIDTH = 258
# SP_HEIGHT = 243

WIDTH = 800
HEIGHT = 600
COLS = 4
ROWS = 5
SP_WIDTH = 160
SP_HEIGHT = 150

IMAGE_CHANNEL = 1 


print('Test.py os.getcwd: ', os.getcwd())

#img_path = os.getcwd()+'/TestImage/Paper/HCT-116/'
#test_save_path = os.getcwd()+'/TestRes/Papertest3/HCT-116/'
#img_path = os.getcwd()+'/TestImage/Paper/HT-29/'
#test_save_path = os.getcwd()+'/TestRes/Papertest3/HT-29/'
#img_path = os.getcwd()+'/TestImage/Paper/MCF-7/'
#test_save_path = os.getcwd()+'/TestRes/Papertest3/MCF-7/'
#img_path = os.getcwd()+'/TestImage/Paper/A549/'
#test_save_path = os.getcwd()+'/TestRes/Papertest3/A549/'
#img_path = os.getcwd()+'/TestImage/Paper/NCI-H23/'
#test_save_path = os.getcwd()+'/TestRes/Papertest3/NCI-H23/'
#img_path = os.getcwd()+'/TestImage/Paper/MDA-MB-231/'
#test_save_path = os.getcwd()+'/TestRes/Papertest3/MDA-MB-231/'

#img_path = os.getcwd()+'/TestImage/0916/DMSO/Dayx/'
#test_save_path = os.getcwd()+'/TestRes/0919/DMSO/Dayx/'
#img_path = os.getcwd()+'/TestImage/1127/NCIH23/Day10/'
#test_save_path = os.getcwd()+'/TestRes/1127/NCIH23/Dayx/'
img_path = os.getcwd()+'/TestImage/1203/04/MCF7/'
test_save_path = os.getcwd()+'/TestRes/1203/04/MCF7/'


#
#modelname = os.getcwd()+'/modInva/model-277500'
modelname = os.getcwd()+'/modNp/model-162500'
#modelname = os.getcwd()+'/modNone/model-500000'



def main(agrv = None):
    start = time.time()
    is_training = tf.placeholder(tf.bool, name = 'is_train') #BN: istraining
    
    image = tf.placeholder(tf.float32, shape = [None, SP_HEIGHT, SP_WIDTH, IMAGE_CHANNEL], name = 'input_img')    

#    logits, pred = net.u_net(image, is_training)
    logits, pred = net.PSPUnet(image, is_training)
    
        
    print ('setup session...')
    print("Setting up dataset reader")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)

    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep=50)
    sess.run(tf.global_variables_initializer())
    
    saver.restore(sess,modelname)
    
    img_folder = img_path
    test_itr=0
    for imgname in os.listdir(img_folder):
        test_itr = test_itr+1 
        print('path is :', img_folder+imgname)
        test_img_batch = testutils.read_img(img_folder+imgname)

        test_img_sp = data_pro.image_splite(test_img_batch,COLS,ROWS)
        pred_sp = {}
        logits_sp = {}
        for col in range(COLS):
            pred_sp[col]={}
            logits_sp[col] = {}
            for row in range(ROWS):
                img_test = test_img_sp[col][row]
                test_feed = {image: img_test, is_training: False}
                
                test_pred_logits, pred_image = sess.run([logits, pred], feed_dict = test_feed)
                logits_sp[col][row] = test_pred_logits
                pred_sp[col][row] = pred_image


        pred_all = data_pro.image_merge(pred_sp, COLS, ROWS, 1)
        pred_batch = np.squeeze(pred_all)
        print("test_itr:",test_itr)
        testutils.saveImage(img_folder+imgname, pred_batch, imgname, test_save_path)

        # pred_batch = np.squeeze(pred_all)

        # pred_tosave = np.reshape(pred_batch, [HEIGHT,WIDTH])
        # print("test_itr:",test_itr)

        # utils.save_Test(imgname[:-4], pred_tosave, test_save_path)
    end = time.time()
    elapse = end - start
    print('elapse time is :', elapse)
    
if __name__ == '__main__':
	tf.app.run()


