#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 04:58:35 2018

@author: Niny
"""

import tensorflow as tf
from .Layers_Dia import convolution_layer, deconvolution_layer, batch_norm

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
dilation_rate1 = [1,1,1,1]
dilation_rate2 = [1,2,2,1]
dilation_rate4 = [1,4,4,1]

def u_net(in_tensor, is_training=True):
    n_channels=64
    
    with tf.variable_scope('encoder_level_0'):
        c0 = tf.nn.relu(convolution_layer(in_tensor, [3, 3, In_Channels, n_channels], [1, 1, 1, 1]))
    
    with tf.variable_scope('encoder_level_1'):
        with tf.variable_scope('conv_1'):
            c11 = convolution_layer(in_tensor, [3, 3, In_Channels, n_channels], [1, 1, 1, 1], dilation_rate1)
            c11_bn = batch_norm(c11,is_training)
            c11_relu = tf.nn.relu(c11_bn)
            
        c1 = c11_relu + c0
        
        with tf.variable_scope('down_sample'):
            c12 = tf.nn.relu(convolution_layer(c1, [2,2, n_channels, n_channels*2], [1,2,2,1]))
            
   
    with tf.variable_scope('encoder_level_2'):
        with tf.variable_scope('conv_1'):
            c21 = convolution_layer(c12, [3,3,n_channels*2, n_channels*2], [1,1,1,1], dilation_rate1)
            c21_bn = batch_norm(c21,is_training)
            c21_relu = tf.nn.relu(c21_bn)

        with tf.variable_scope('conv_2'):
            c22 = convolution_layer(c21_relu, [3,3,n_channels*2, n_channels*2], [1,1,1,1], dilation_rate1)
            c22_bn = batch_norm(c22,is_training)
            c22_relu = tf.nn.relu(c22_bn)
        
        c2 = c22_relu + c12
        
        with tf.variable_scope('down_sample'):
            c23 = tf.nn.relu(convolution_layer(c2, [2,2, n_channels*2, n_channels*4], [1,2,2,1]))
            
    
    with tf.variable_scope('encoder_level_3'):
        with tf.variable_scope('conv_1'):
            c31 = convolution_layer(c23, [3,3,n_channels*4, n_channels*4], [1,1,1,1], dilation_rate1)
            c31_bn = batch_norm(c31,is_training)
            c31_relu = tf.nn.relu(c31_bn)
            
        with tf.variable_scope('conv_2'):
            c32 = convolution_layer(c31_relu, [3,3,n_channels*4, n_channels*4], [1,1,1,1], dilation_rate1)
            c32_bn = batch_norm(c32,is_training)
            c32_relu = tf.nn.relu(c32_bn)
            
        with tf.variable_scope('conv_3'):
            c33 = convolution_layer(c32_relu, [3,3,n_channels*4, n_channels*4], [1,1,1,1], dilation_rate1)
            c33_bn = batch_norm(c33,is_training)
            c33_relu = tf.nn.relu(c33_bn)
        
        c3 = c33_relu + c23
        
        with tf.variable_scope('down_sample'):
            c34 = tf.nn.relu(convolution_layer(c3, [2,2, n_channels*4, n_channels*8], [1,2,2,1]))
     
        
    with tf.variable_scope('encoder_level_4'):
        with tf.variable_scope('conv_1'):
            c41 = convolution_layer(c34, [3,3,n_channels*8, n_channels*8], [1,1,1,1], dilation_rate1)
            c41_bn = batch_norm(c41,is_training)
            c41_relu = tf.nn.relu(c41_bn)
            
        with tf.variable_scope('conv_2'):
            c42 = convolution_layer(c41_relu, [3,3,n_channels*8, n_channels*8],[1,1,1,1], dilation_rate2)
            c42_bn = batch_norm(c42,is_training)
            c42_relu = tf.nn.relu(c42_bn)
            
        with tf.variable_scope('conv_3'):
            c43 = convolution_layer(c42_relu, [3,3,n_channels*8, n_channels*8],[1,1,1,1], dilation_rate2)
            c43_bn = batch_norm(c43,is_training)
            c43_relu = tf.nn.relu(c43_bn)
         
        c4 = c43_relu + c34
        
        with tf.variable_scope('down_sample'):
            c44 = tf.nn.relu(convolution_layer(c4, [2,2, n_channels*8, n_channels*16], [1,2,2,1]))
    
    
    with tf.variable_scope('encoder_decoder_level_5'):
        with tf.variable_scope('conv_1'):
            c51 = convolution_layer(c44, [3,3,n_channels*16, n_channels*16],[1,1,1,1], dilation_rate1)
            c51_bn = batch_norm(c51,is_training)
            c51_relu = tf.nn.relu(c51_bn)
            
        with tf.variable_scope('conv_2'):
            c52 = convolution_layer(c51_relu, [3,3,n_channels*16, n_channels*16],[1,1,1,1], dilation_rate4)
            c52_bn = batch_norm(c52,is_training)
            c52_relu = tf.nn.relu(c52_bn)
        
        with tf.variable_scope('conv_3'):
            c53 = convolution_layer(c52_relu, [3,3,n_channels*16, n_channels*16],[1,1,1,1], dilation_rate4)
            c53_bn = batch_norm(c53,is_training)
            c53_relu = tf.nn.relu(c53_bn)
        
        c5 = c53_relu + c44
        
        with tf.variable_scope('up_sample'):
            c54 = deconvolution_layer(c5, [2,2,n_channels*8, n_channels*16], tf.shape(c43), [1,2,2,1])
            c54_relu = tf.nn.relu(c54)
            
            
    with tf.variable_scope('decoder_level_4'):
        d4 = tf.concat((c54_relu, c43_relu), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d41 = convolution_layer(d4, [3,3,n_channels*16, n_channels*8],[1,1,1,1], dilation_rate1)
            d41_bn = batch_norm(d41,is_training)
            d41_relu = tf.nn.relu(d41_bn)
            
        with tf.variable_scope('conv_2'):
            d42 = convolution_layer(d41_relu, [3,3,n_channels*8, n_channels*8],[1,1,1,1], dilation_rate2)
            d42_bn = batch_norm(d42,is_training)
            d42_relu = tf.nn.relu(d42_bn)
            
        with tf.variable_scope('conv_3'):
            d43 = convolution_layer(d42_relu, [3,3,n_channels*8, n_channels*8],[1,1,1,1], dilation_rate2)
            d43_bn = batch_norm(d43,is_training)
            d43_relu = tf.nn.relu(d43_bn)
        
        da4 = d43_relu + c54_relu
        
        with tf.variable_scope('up_sample'):
            d44 = deconvolution_layer(da4, [2,2,n_channels*4, n_channels*8], tf.shape(c33), [1,2,2,1])
            d44_relu = tf.nn.relu(d44)
            
    with tf.variable_scope('decoder_level_3'):
        d3 = tf.concat((d44_relu, c33_relu), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d31 = convolution_layer(d3, [3,3,n_channels*8, n_channels*4],[1,1,1,1], dilation_rate1)
            d31_bn = batch_norm(d31,is_training)
            d31_relu = tf.nn.relu(d31_bn)
        
        with tf.variable_scope('conv_2'):
            d32 = convolution_layer(d31_relu, [3,3,n_channels*4, n_channels*4],[1,1,1,1], dilation_rate1)
            d32_bn = batch_norm(d32,is_training)
            d32_relu = tf.nn.relu(d32_bn)
            
        with tf.variable_scope('conv_3'):
            d33 = convolution_layer(d32_relu, [3,3,n_channels*4, n_channels*4],[1,1,1,1], dilation_rate1)
            d33_bn = batch_norm(d33,is_training)
            d33_relu = tf.nn.relu(d33_bn)
            
        da3 = d33_relu + d44_relu
            
        with tf.variable_scope('up_sample'):
            d34 = deconvolution_layer(da3, [2,2,n_channels*2, n_channels*4], tf.shape(c22), [1,2,2,1])
            d34_relu = tf.nn.relu(d34)
        
            
    with tf.variable_scope('decoder_level_2'):
        d2 = tf.concat((d34_relu, c22_relu), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d21 = convolution_layer(d2, [3,3,n_channels*4, n_channels*2],[1,1,1,1], dilation_rate1)
            d21_bn = batch_norm(d21,is_training)
            d21_relu = tf.nn.relu(d21_bn)
        
        with tf.variable_scope('conv_2'):
            d22 = convolution_layer(d21_relu, [3,3,n_channels*2, n_channels*2],[1,1,1,1], dilation_rate1)
            d22_bn = batch_norm(d22,is_training)
            d22_relu = tf.nn.relu(d22_bn)
            
        da2 = d22_relu + d34_relu
        
        with tf.variable_scope('up_sample'):
            d23 = deconvolution_layer(da2, [2,2,n_channels, n_channels*2], tf.shape(c11), [1,2,2,1])
            d23_relu = tf.nn.relu(d23)
            
            
    with tf.variable_scope('decoder_level_1'):
        d1 = tf.concat((d23_relu, c11_relu), axis = -1)
        
        with tf.variable_scope('conv_1'):
            d11 = convolution_layer(d1, [3,3,n_channels*2, n_channels],[1,1,1,1], dilation_rate1)
            d11_bn = batch_norm(d11,is_training)
            d11_relu = tf.nn.relu(d11_bn)
        da1 = d11_relu + d23_relu
        
    with tf.variable_scope('output_layer'):
        logits = convolution_layer(da1, [1,1,n_channels,NUM_OF_CLASS], [1,1,1,1])
        
    
    annotation_pred = tf.expand_dims(tf.argmax(logits, axis = 3, name = 'prediction'), axis= 3)
    
    return logits, annotation_pred
            
            

            
            
            
            
            
            
            