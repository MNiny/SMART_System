# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:45:59 2018

@author: Niny
"""

import tensorflow as tf
import numpy as np
import scipy.misc as misc
from six.moves import urllib
import scipy

import os



def get_conv_var(filter):
    #normal distribution
    std=0.01  #tf.sqrt(3. / (tf.reduce_prod(filter[:2]) * tf.reduce_sum(filter[2:])))
    init_value = tf.truncated_normal(filter, mean = 0.0, stddev=std)
    w = tf.Variable(initial_value = init_value, name = 'weight')

    return w

def convolution_layer(in_tensor, filter, strides,  padding = 'SAME', dilations = [1,1,1,1]):
    assert len(filter) == 4 #[height, width, in_channels, out_channels]
    assert len(strides)==4 #match input dimensions [batch, inheight, inwidth, inchannels]
    assert padding in ['VALID', 'SAME']
    
    w = get_conv_var(filter)
    b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
    conv = tf.nn.conv2d(in_tensor, w, strides, padding, dilations=dilations)
    bias=conv+b
    
    return bias



def batch_norm(in_tensor, is_training, scope):
    bn = tf.contrib.layers.batch_norm(in_tensor, decay=0.997, epsilon=1e-4, is_training=is_training, scope=scope)
    return bn



def maxpool(in_tensor):
    return tf.nn.max_pool(in_tensor, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],padding = 'SAME')

def deconvolution_layer(in_tensor, filter, output_shape, strides, padding='SAME'):
    assert len(filter) == 4  # [height, width, out_channels, in_channels]
    assert len(strides) == 4  # must match input dimensions [batch, height, width, in_channels]
    assert padding in ['VALID', 'SAME']
    
    w = get_conv_var(filter)
    b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')
    deconvolution = tf.nn.conv2d_transpose(in_tensor, w, output_shape, strides, padding)
    return deconvolution + b


def Bottleneck(in_tensor, in_channel, out_channel, dilations, is_training, scope, stride = 1):
    dim = False
    
    if in_channel == out_channel:
        dim = False
    else:
        dim = True
        
    with tf.variable_scope('conv_in_block'):
        conv1 = tf.nn.relu(batch_norm(convolution_layer(in_tensor, filter=[1, 1, in_channel, out_channel//4], strides=[1, 1, 1, 1], dilations=dilations), is_training, scope = scope+'bn1'))

        conv2 = tf.nn.relu(batch_norm(convolution_layer(conv1, filter=[3, 3, out_channel//4, out_channel//4], strides=[1,stride, stride, 1], dilations=dilations), is_training, scope = scope+'bn2'))

        conv3 = tf.nn.relu(batch_norm(convolution_layer(conv2, filter=[1, 1, out_channel//4, out_channel], strides=[1, 1, 1, 1], dilations=dilations), is_training, scope = scope+'bn3'))

    with tf.variable_scope('shortcut_connection'):
        if dim is True:
            x = convolution_layer(in_tensor, [1, 1, in_channel, out_channel],[1, stride, stride, 1])
        else:
            x = in_tensor

        add = tf.nn.relu(tf.add_n([conv3,x]))
    
    return add

