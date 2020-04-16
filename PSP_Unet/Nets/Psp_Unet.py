# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:39:26 2018

@author: Niny
"""
import tensorflow as tf
from .PSPLayers import convolution_layer, deconvolution_layer, batch_norm, maxpool, Bottleneck

FLAGS = tf.app.flags.FLAGS

NUM_OF_CLASS = 2


"""
2Dpsp-unet
input_batch=[batch_size,w,h,1]

Batch_norm
strid=2 convolution replaced max_pooling
"""



def PSPUnet(in_tensor, is_training):
    
    dilation_rate1 = [1, 1, 1, 1]
    dilation_rate2 = [1, 2, 2, 1]
    dilation_rate4 = [1, 4, 4, 1]
    
    n_channels =64
    with tf.variable_scope('conv0'):
        c01 = tf.nn.relu(batch_norm(convolution_layer(in_tensor, [5, 5, 1, n_channels//2], [1, 1, 1, 1]), is_training, 'conv01'))
        c02 = tf.nn.relu(batch_norm(convolution_layer(c01, [3, 3, n_channels//2, n_channels], [1, 1, 1, 1]), is_training, 'conv02'))
    
    with tf.variable_scope('maxpool'):
        pool = maxpool(c02)
    
    with tf.variable_scope('conv1'):
        c1 = Bottleneck(pool, n_channels, n_channels*2, dilation_rate1, is_training, scope='c1')
        c11 = Bottleneck(c1, n_channels*2, n_channels*2, dilation_rate1, is_training, scope='c11')
        c12 = Bottleneck(c11, n_channels*2, n_channels*2, dilation_rate1, is_training, scope='c12')
       
    with tf.variable_scope('conv2'):
        c2 = Bottleneck(c12, n_channels*2, n_channels*4, dilation_rate1, is_training, scope='c2', stride=2)
        c21 = Bottleneck(c2, n_channels*4, n_channels*4, dilation_rate1, is_training, scope='c21')
        c22 = Bottleneck(c21, n_channels*4, n_channels*4, dilation_rate1, is_training, scope='c22')
        c23 = Bottleneck(c22, n_channels*4, n_channels*4, dilation_rate1, is_training, scope='c23')
        
    with tf.variable_scope('conv3'): 
        c3 = Bottleneck(c23, n_channels*4, n_channels*8, dilation_rate1, is_training, scope='c3', stride=2)
        c31 = Bottleneck(c3, n_channels*8, n_channels*8, dilation_rate2, is_training, scope='c31')
        c32 = Bottleneck(c31, n_channels*8, n_channels*8, dilation_rate2, is_training, scope='c32')
        c33 = Bottleneck(c32, n_channels*8, n_channels*8, dilation_rate2, is_training, scope='c33')
        c34 = Bottleneck(c33, n_channels*8, n_channels*8, dilation_rate2, is_training, scope='c34')
        c35 = Bottleneck(c34, n_channels*8, n_channels*8, dilation_rate2, is_training, scope='c35')
        c36 = Bottleneck(c35, n_channels*8, n_channels*8, dilation_rate2, is_training, scope='c36')
    
    with tf.variable_scope('conv4'): 
        c4 = Bottleneck(c36, n_channels*8, n_channels*16, dilation_rate1, is_training, scope='c4', stride=2)
        c41 = Bottleneck(c4, n_channels*16, n_channels*16, dilation_rate4, is_training, scope='c41')
        c42 = Bottleneck(c41, n_channels*16, n_channels*16, dilation_rate4, is_training, scope='c42')
        c43 = Bottleneck(c42, n_channels*16, n_channels*16, dilation_rate4, is_training, scope='c43')
        c44 = Bottleneck(c43, n_channels*16, n_channels*16, dilation_rate4, is_training, scope='c44')
        c45 = Bottleneck(c44, n_channels*16, n_channels*16, dilation_rate4, is_training, scope='c45')
        c46 = Bottleneck(c45, n_channels*16, n_channels*16, dilation_rate4, is_training, scope='c46')
        c47 = Bottleneck(c46, n_channels*16, n_channels*16, dilation_rate4, is_training, scope='c47')
        c48 = Bottleneck(c47, n_channels*16, n_channels*16, dilation_rate4, is_training, scope='c48')
        
    with tf.variable_scope('deconv4'):
        up4 = tf.nn.relu(batch_norm(deconvolution_layer(c48, [2, 2, n_channels*8, n_channels*16], tf.shape(c36), [1, 2, 2, 1]), is_training, 'de4'))
        con4 = tf.concat((up4, c36), axis = -1)
        d4 = Bottleneck(con4, n_channels*16, n_channels*8, dilation_rate1, is_training, scope='d41')
        d41 = Bottleneck(d4, n_channels*8, n_channels*8, dilation_rate4, is_training, scope='d4')
        d42 = Bottleneck(d41, n_channels*8, n_channels*8, dilation_rate4, is_training, scope='d42')
        d43 = Bottleneck(d42, n_channels*8, n_channels*8, dilation_rate4, is_training, scope='d43')
        d44 = Bottleneck(d43, n_channels*8, n_channels*8, dilation_rate4, is_training, scope='d44')
        d45 = Bottleneck(d44, n_channels*8, n_channels*8, dilation_rate4, is_training, scope='d45')
        d46 = Bottleneck(d45, n_channels*8, n_channels*8, dilation_rate4, is_training, scope='d46')
        d47 = Bottleneck(d46, n_channels*8, n_channels*8, dilation_rate4, is_training, scope='d47')
        d48 = Bottleneck(d47, n_channels*8, n_channels*8, dilation_rate4, is_training, scope='d48')
        
    with tf.variable_scope('deconv3'):
        up3 = tf.nn.relu(batch_norm(deconvolution_layer(d48, [2, 2, n_channels*4, n_channels*8], tf.shape(c23), [1, 2, 2, 1]), is_training, 'de3'))
        con3 = tf.concat((up3, c23), axis = -1)
        d3 = Bottleneck(con3, n_channels*8, n_channels*4, dilation_rate1, is_training, scope='d3')
        d31 = Bottleneck(d3, n_channels*4, n_channels*4, dilation_rate2, is_training, scope='d31')
        d32 = Bottleneck(d31, n_channels*4, n_channels*4, dilation_rate2, is_training, scope='d32')
        d33 = Bottleneck(d32, n_channels*4, n_channels*4, dilation_rate2, is_training, scope='d33')
        d34 = Bottleneck(d33, n_channels*4, n_channels*4, dilation_rate2, is_training, scope='d34')
        d35 = Bottleneck(d34, n_channels*4, n_channels*4, dilation_rate2, is_training, scope='d35')
        d36 = Bottleneck(d35, n_channels*4, n_channels*4, dilation_rate2, is_training, scope='d36')
    
    with tf.variable_scope('deconv2'):
        up2 = tf.nn.relu(batch_norm(deconvolution_layer(d36, [2, 2, n_channels*2, n_channels*4], tf.shape(c12), [1, 2, 2, 1]), is_training, 'de2'))
        con2 = tf.concat((up2, c12), axis = -1)
        d2 = Bottleneck(con2, n_channels*4, n_channels*2, dilation_rate1, is_training, scope='d2')
        d21 = Bottleneck(d2, n_channels*2, n_channels*2, dilation_rate1, is_training, scope='d21')
        d22 = Bottleneck(d21, n_channels*2, n_channels*2, dilation_rate1, is_training, scope='d22')
        d23 = Bottleneck(d22, n_channels*2, n_channels*2, dilation_rate1, is_training, scope='d23')
        
    with  tf.variable_scope('deconv1'):
        up1 = tf.nn.relu(batch_norm(deconvolution_layer(d23, [2, 2, n_channels, n_channels*2], tf.shape(c02), [1, 2, 2, 1]), is_training, 'de1'))
        con1 = tf.concat((up1, c02), axis = -1)
        d1 = Bottleneck(con1, n_channels*2, n_channels, dilation_rate1, is_training, scope='d1')
        d11 = Bottleneck(d1, n_channels, n_channels, dilation_rate1, is_training, scope='d11')
        d12 = Bottleneck(d11, n_channels, n_channels, dilation_rate1, is_training, scope='d12')
       
    with  tf.variable_scope('finalconv'): 
        logits = convolution_layer(d12, filter=[1, 1, n_channels, NUM_OF_CLASS], strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1])
        
    
    annotation_pred = tf.expand_dims(tf.argmax(logits, axis = 3, name = 'prediction'), dim= 3)
    
    return logits, annotation_pred 


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.Debug:
        # print(len(var_list))
        for grad, var in grads:
            tf.summary.histogram(grad, var)
            
    return optimizer.apply_gradients(grads)

  