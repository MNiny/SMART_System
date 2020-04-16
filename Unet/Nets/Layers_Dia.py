#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 05:11:31 2018

@author: Niny
"""

import tensorflow as tf
import numpy as np
import scipy.misc as misc
from six.moves import urllib
import scipy


import Unet_Res_Dia

import os

IMAGE_SIZE=128
IMAGE_OF_NUM=64
inchannels=1

def get_conv_var(filter):
    #normal distribution
    std=0.01  #tf.sqrt(3. / (tf.reduce_prod(filter[:2]) * tf.reduce_sum(filter[2:])))
    init_value = tf.truncated_normal(filter, mean = 0.0, stddev=std)
    w = tf.Variable(initial_value = init_value, name = 'weight')

    return w


def batch_norm(in_tensor, is_training):
    bn = tf.contrib.layers.batch_norm(in_tensor, decay=0.997, epsilon=1e-4, is_training=is_training)
    return bn

def convolution_layer(in_tensor, filter, strides, dilations = [1,1,1,1], padding = 'SAME'):
    assert len(filter) == 4 #[height, width, in_channels, out_channels]
    assert len(strides)==4 #match input dimensions [batch, inheight, inwidth, inchannels]
    assert padding in ['VALID', 'SAME']
    
    w = get_conv_var(filter)
    b = tf.Variable(tf.constant(1.0, shape=[filter[-1]]), name='biases')
    conv = tf.nn.conv2d(in_tensor, w, strides, padding, dilations=dilations)
    bias=conv+b
    
    return bias



def deconvolution_layer(in_tensor, filter, output_shape, strides, padding='SAME'):
    assert len(filter) == 4  # [height, width, out_channels, in_channels]
    assert len(strides) == 4  # must match input dimensions [batch, height, width, in_channels]
    assert padding in ['VALID', 'SAME']
    
    w = get_conv_var(filter)
    b = tf.Variable(tf.constant(1.0, shape=[filter[-2]]), name='biases')
    deconvolution = tf.nn.conv2d_transpose(in_tensor, w, output_shape, strides, padding)
    return deconvolution + b


