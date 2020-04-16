#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 04:58:35 2018

@author: Niny
"""

import tensorflow as tf
from .Layers import convolution_layer, deconvolution_layer, maxpool

FLAGS = tf.app.flags.FLAGS

In_Channels=1
NUM_OF_CLASS = 2


"""
2Dunet 
image_size=256*256
input_batch=[batch_size,256,256,1]

Batch_norm
strid=2 convolution replaced max_pooling
"""

def u_net(in_tensor, is_training=True):
    n_channels=64
    
    with tf.variable_scope('encoder_level_1'):
        with tf.variable_scope('conv_1'):
            c11 = convolution_layer(in_tensor, [3, 3, In_Channels, n_channels], [1, 1, 1, 1])
            c11_relu = tf.nn.relu(c11_bn)

        with tf.variable_scope('conv_2'):
            c12 = convolution_layer(c11_relu, [3, 3, n_channels, n_channels], [1, 1, 1, 1])
            c12_relu = tf.nn.relu(c11_bn)

        with tf.variable_scope('down_sample'):
            c13 = tf.nn.relu(maxpool(c12_relu))
            
   
    with tf.variable_scope('encoder_level_2'):
        with tf.variable_scope('conv_1'):
            c21 = convolution_layer(c13, [3,3,n_channels, n_channels*2],[1,1,1,1])
            c21_relu = tf.nn.relu(c21)

        with tf.variable_scope('conv_2'):
            c22 = convolution_layer(c21_relu, [3,3,n_channels*2, n_channels*2],[1,1,1,1])
            c22_relu = tf.nn.relu(c22)

        with tf.variable_scope('down_sample'):
            c23 = tf.nn.relu(maxpool(c22_relu))
            
    
    with tf.variable_scope('encoder_level_3'):
        with tf.variable_scope('conv_1'):
            c31 = convolution_layer(c23, [3,3,n_channels*2, n_channels*4],[1,1,1,1])
            c31_relu = tf.nn.relu(c31)
            
        with tf.variable_scope('conv_2'):
            c32 = convolution_layer(c31_relu, [3,3,n_channels*4, n_channels*4],[1,1,1,1])
            c32_relu = tf.nn.relu(c32)

        with tf.variable_scope('down_sample'):
            c33 = tf.nn.relu(maxpool(c32_relu))
     
        
    with tf.variable_scope('encoder_level_4'):
        with tf.variable_scope('conv_1'):
            c41 = convolution_layer(c33, [3,3,n_channels*4, n_channels*8],[1,1,1,1])
            c41_relu = tf.nn.relu(c41)
            
        with tf.variable_scope('conv_2'):
            c42 = convolution_layer(c41_relu, [3,3,n_channels*8, n_channels*8],[1,1,1,1])
            c42_relu = tf.nn.relu(c42)

        with tf.variable_scope('down_sample'):
            c43 = tf.nn.relu(maxpool(c42_relu))
    
    
    with tf.variable_scope('encoder_decoder_level_5'):
        with tf.variable_scope('conv_1'):
            c51 = convolution_layer(c43, [3,3,n_channels*8, n_channels*16],[1,1,1,1])
            c51_relu = tf.nn.relu(c51)
            
        with tf.variable_scope('conv_2'):
            c52 = convolution_layer(c51_relu, [3,3,n_channels*16, n_channels*16],[1,1,1,1])
            c52_relu = tf.nn.relu(c52)
        
        with tf.variable_scope('up_sample'):
            c53 = deconvolution_layer(c52_relu, [2,2,n_channels*8, n_channels*16], tf.shape(c43), [1,2,2,1])
            c53_relu = tf.nn.relu(c53)
            
            
    with tf.variable_scope('decoder_level_4'):
        d4 = tf.concat((c53_relu, c42_relu), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d41 = convolution_layer(d4, [3,3,n_channels*16, n_channels*8],[1,1,1,1])
            d41_relu = tf.nn.relu(d41)
            
        with tf.variable_scope('conv_2'):
            d42 = convolution_layer(d41_relu, [3,3,n_channels*8, n_channels*8],[1,1,1,1])
            d42_relu = tf.nn.relu(d42)
        
        with tf.variable_scope('up_sample'):
            d43 = deconvolution_layer(d42_relu, [2,2,n_channels*4, n_channels*8], tf.shape(c33), [1,2,2,1])
            d43_relu = tf.nn.relu(d44)
            
    with tf.variable_scope('decoder_level_3'):
        d3 = tf.concat((d43_relu, c32_relu), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d31 = convolution_layer(d3, [3,3,n_channels*8, n_channels*4],[1,1,1,1])
            d31_relu = tf.nn.relu(d31)
        
        with tf.variable_scope('conv_2'):
            d32 = convolution_layer(d31_relu, [3,3,n_channels*4, n_channels*4],[1,1,1,1])
            d32_relu = tf.nn.relu(d32)

            
        with tf.variable_scope('up_sample'):
            d33 = deconvolution_layer(d32_relu, [2,2,n_channels*2, n_channels*4], tf.shape(c22), [1,2,2,1])
            d33_relu = tf.nn.relu(d33)
        
            
    with tf.variable_scope('decoder_level_2'):
        d2 = tf.concat((d33_relu, c22_relu), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d21 = convolution_layer(d2, [3,3,n_channels*4, n_channels*2],[1,1,1,1])
            d21_relu = tf.nn.relu(d21)
        
        with tf.variable_scope('conv_2'):
            d22 = convolution_layer(d21_relu, [3,3,n_channels*2, n_channels*2],[1,1,1,1])
            d22_relu = tf.nn.relu(d22)
        
        with tf.variable_scope('up_sample'):
            d23 = deconvolution_layer(d22_relu, [2,2,n_channels, n_channels*2], tf.shape(c11), [1,2,2,1])
            d23_relu = tf.nn.relu(d23)
            
            
    with tf.variable_scope('decoder_level_1'):
        d1 = tf.concat((d23_relu, c11_relu), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d11 = convolution_layer(d1, [3,3,n_channels*2, n_channels],[1,1,1,1])
            d11_relu = tf.nn.relu(d11)
        with tf.variable_scope('conv_2'):
            d12 = convolution_layer(d11_relu, [3, 3, n_channels * 2, n_channels], [1, 1, 1, 1])
            d12_relu = tf.nn.relu(d12)


    with tf.variable_scope('output_layer'):
        logits = convolution_layer(d12_relu, [1,1,n_channels,NUM_OF_CLASS], [1,1,1,1])
        
    
    annotation_pred = tf.expand_dims(tf.argmax(logits, axis = 3, name = 'prediction'), axis= 3)
    
    return logits, annotation_pred
            
            

            
            
            
            
            
            
            