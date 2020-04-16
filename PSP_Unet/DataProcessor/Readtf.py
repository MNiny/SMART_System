#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 22:00:22 2018

@author: Niny
"""

'''
read from .raws
'''
import os
import scipy.io as sio
#import tensorflow as tf
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.python.lib.io import file_io as file_io


cwd = os.getcwd()
print(cwd)

recordPath = os.getcwd()+'/Data/'



WIDTH = 800
HEIGHT = 600
SP_WIDTH = 160
SP_HEIGHT = 150



def read_traintfrcd(filename):
#    file_queue = tf.train.string_input_producer([filename],num_epochs=1)
    file_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
        
    features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                    'img_raw': tf.FixedLenFeature([], tf.string),
                    'gt_raw': tf.FixedLenFeature([], tf.string),
                    })
    
    with tf.variable_scope('decoder'):
        #image = tf.decode_raw(features['img_raw'], tf.uint8)
        image = tf.decode_raw(features['img_raw'], tf.float64)
        ground_truth = tf.decode_raw(features['gt_raw'], tf.uint8)
        
        
    with tf.variable_scope('image'):
        # reshape and add 0 dimension (would be batch dimension)
        image = tf.cast(tf.reshape(image, [SP_HEIGHT,SP_WIDTH,1]), tf.float32)
        #image = tf.cast(tf.reshape(image, [SP_HEIGHT,SP_WIDTH,3]), tf.float32)
        #image = tf.expand_dims(image, axis = 2)
    with tf.variable_scope('ground_truth'):
        # reshape
        ground_truth = tf.cast(tf.reshape(ground_truth, [SP_HEIGHT,SP_WIDTH,1]), tf.float32)
        #ground_truth = tf.expand_dims(ground_truth, axis = 2)
    return image,ground_truth



def read_testtfrcd(filename):
    file_queue = tf.train.string_input_producer([filename])
#    file_queue = tf.train.string_input_producer([filename],num_epochs=1)
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
        
    features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                    'img_raw': tf.FixedLenFeature([], tf.string),
                    'gt_raw': tf.FixedLenFeature([], tf.string),
                    })
    
    with tf.variable_scope('decoder'):
        #image = tf.decode_raw(features['img_raw'], tf.uint8)
        image = tf.decode_raw(features['img_raw'], tf.float64)
        ground_truth = tf.decode_raw(features['gt_raw'], tf.uint8)
        
        
    with tf.variable_scope('image'):
        # reshape and add 0 dimension (would be batch dimension)
        #image = tf.cast(tf.reshape(image, [HEIGHT,WIDTH,3]), tf.float32)
        image = tf.cast(tf.reshape(image, [HEIGHT,WIDTH,1]), tf.float32)
        #image = tf.expand_dims(image, axis = 2)
    with tf.variable_scope('ground_truth'):
        # reshape
        ground_truth = tf.cast(tf.reshape(ground_truth, [HEIGHT,WIDTH,1]), tf.float32)
        #ground_truth = tf.expand_dims(ground_truth, axis = 2)
    return image,ground_truth


def read_and_decord(mode):
    if mode=='Train':
        filename = recordPath+'TrainSet.tfrecord'
        image,ground_truth = read_traintfrcd(filename)
    if mode=='Test':
        filename = recordPath+'TestSet.tfrecord'
        image,ground_truth = read_testtfrcd(filename)
        
   
    return image,ground_truth


    

if __name__ == '__main__':
	#createTrainRecord()

	# img_data shape: [batch_size, depths, row(height), cols(width)] ,
	# need to add channels =1 at the end of cols(width) to 5D tensor
    
    
    image,label= read_and_decord('Test')
    
    img_batch,label_batch = tf.train.batch([image,label], batch_size = 1, capacity=1)
    init = tf.global_variables_initializer()
    local=tf.local_variables_initializer()
    
    sess=tf.Session() 
    sess.run([init,local])
    sum=0
    threads = tf.train.start_queue_runners(sess)
    for i in range(500000):
        sum=sum+1
        x,y= sess.run([img_batch, label_batch])
        print(x.shape)
        print(y.shape)
        #print (s)
        print(sum)
		#np.savetxt('xx32.csv', x[0, 39, :, :], delimiter=',', fmt = '%.4f' )










