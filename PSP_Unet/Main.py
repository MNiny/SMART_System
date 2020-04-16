#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:15:33 2018


@author: Niny
"""

from DataProcessor import Readtf as read_mat
from DataProcessor import Processor as data_pro

import tensorflow as tf 
import numpy as np
from scipy import misc
import time
import TFUtils as utils
#from UNets import Unet_Res_Dia as net
from Nets import Psp_Unet as net
import TestUtils as testutils

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf.app.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.app.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.app.flags.DEFINE_bool('Debug', "False", "Debug mode: True/ False")
tf.app.flags.DEFINE_string('Mode', "Train", "Mode Train/ Test/ Visualize")
tf.app.flags.DEFINE_string("mod_dir","mod/model-162500","path to model")
tf.app.flags.DEFINE_string("img_dir","TestImage/","test image path")
FLAGS = tf.app.flags.FLAGS

recordPath = os.getcwd()+'/'


WIDTH = 800
HEIGHT = 600
COLS = 4
ROWS = 5
SP_WIDTH = 160
SP_HEIGHT = 150



test_epoch = 2500


Epoch = 500
batch_sz=8
Train_NUM = 3200

MAX_ITERATION = 520000

IMAGE_CHANNEL = 1 

TEST_RAW =25
img_path = os.getcwd()+FLAGS.img_dir
modelname = os.getcwd()+FLAGS.mod_dir
test_save_path = os.getcwd()+'/TestRes/'

def Train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.Debug:
        # print(len(var_list))
        for grad, var in grads:
            tf.summary.histogram(grad, var)
            
    return optimizer.apply_gradients(grads)


def main(agrv = None):
	#FLAGS.para_name
    if FLAGS.Mode == 'Train':
        #keep_probability = tf.placeholder(tf.float32, name = "keep_prob") #dropout: keep_probability
        is_training = tf.placeholder(tf.bool, name = 'is_train') #BN: istraining
       
        image = tf.placeholder(tf.float32, shape = [None, SP_HEIGHT, SP_WIDTH, IMAGE_CHANNEL], name = 'input_img')    
        annotation = tf.placeholder(tf.int32, shape = [None, SP_HEIGHT, SP_WIDTH, 1], name = 'annotation')
    
        Testloss = tf.placeholder(tf.float32, name = 'Test_loss')
    
        logits, pred = net.PSPUnet(image, is_training)
    
    
        tf.summary.image("image", image, max_outputs = batch_sz)
        tf.summary.image("groud_truth", tf.cast(annotation, tf.uint8), max_outputs = batch_sz)
        tf.summary.image("pred_annotation", tf.cast(pred, tf.uint8), max_outputs = batch_sz)
        #softmax cross entropy loss
        loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                               labels = tf.squeeze(annotation, axis=[3]),
                                                               name = 'entropy_loss'))

        DSC_batch = tf.placeholder(tf.float32, name = 'Dice_coeff')

        trainable_var = tf.trainable_variables()
        if FLAGS.Debug:
            for var in trainable_var:
                tf.summary.histogram(var.op.name, var)
                tf.add_to_collection('reg_loss', tf.nn.l2_loss(var))
    
        #BN: update moving_mean&moving_variance
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = Train(loss, trainable_var)
    
    #    loss_train_op = tf.summary.scalar('training_loss', loss)
        tf.summary.scalar('training_loss', loss)
        summary_op = tf.summary.merge_all()
    
        loss_test_op = tf.summary.scalar('test_loss',Testloss)
        DSC_op = tf.summary.scalar('Dice_coefficient', DSC_batch)
        print("setting up image reader...")
        image_batch_a, label_batch_a = read_mat.read_and_decord('Train')
        
        img_train_batch, label_train_batch = tf.train.shuffle_batch([image_batch_a, label_batch_a],
																	batch_size=batch_sz, capacity=batch_sz*2, min_after_dequeue=batch_sz)

        print (img_train_batch.shape)
        print ('setup session...')

    
        print("Setting up dataset reader")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(config=config)

        print("Setting up Saver...")
        saver = tf.train.Saver(max_to_keep=50)
        #filename = 'E:/maning/Cells/mod/model-87500'
        #saver.restore(sess,filename)
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir,sess.graph)
        sess.run(tf.global_variables_initializer())
        threads = tf.train.start_queue_runners(sess)
        test_i = -1

        for itr in range (MAX_ITERATION):
            #img_batch shape:[batch_size, depth, height, width, chanl], type: ndarray
            img_batch, l_batch = sess.run([img_train_batch, label_train_batch ])
            
            print (itr, img_batch.shape)
            feed = {image: img_batch, annotation: l_batch, is_training: True}
            sess.run(train_op, feed_dict = feed)

            train_loss_print, summary_str= sess.run([loss,summary_op],  feed_dict = feed)
            print (train_loss_print)
            summary_writer.add_summary(summary_str, itr)


            if itr%test_epoch == 0 or itr%5000 == 0: 
                saver.save(sess, './mod/model', global_step=itr)
            elif itr == (MAX_ITERATION - 1):
                saver.save(sess, './mod/model', global_step=itr)
    
            ############################# test test test test test test test test test test#####################################
            if (itr!=0 and itr%test_epoch == 0) or itr == (MAX_ITERATION - 1):
                #overfittest
                test_i = test_i+1
                print("train finish! start test~")
                
                test_img_data_a, test_label_data_a = read_mat.read_and_decord('Test')
                
                test_img_train_batch, test_label_train_batch = tf.train.batch([test_img_data_a, test_label_data_a],batch_size=1, capacity=1)
                
                threads = tf.train.start_queue_runners(sess)
                Testl = 0.0
                for test_itr in range (TEST_RAW): 
                    test_img_batch, test_l_batch = sess.run([test_img_train_batch, test_label_train_batch])
                    test_img_sp = data_pro.image_splite(test_img_batch,COLS,ROWS)
                    test_lbl_sp = data_pro.image_splite(test_l_batch,COLS,ROWS)
                    pred_sp = {}
                    for col in range(COLS):
                        pred_sp[col]={}
                        for row in range(ROWS):
                            img_test = test_img_sp[col][row]
                            lbl_test = test_lbl_sp[col][row]
                            test_feed = {image: img_test, annotation: lbl_test, is_training: False}
                            
                            test_pred_logits, pred_image, testloss = sess.run([logits, pred, loss], feed_dict = test_feed)
                            pred_sp[col][row] = pred_image
                            Testl = Testl+testloss
                    #pred_image = sess.run(tf.argmax(input=test_pred_logits,axis=3))
                    
                    pred_all = data_pro.image_merge(pred_sp, COLS, ROWS, 1)
                    
                    label_batch = np.squeeze(test_l_batch)
                    pred_batch = np.squeeze(pred_all)
                    #label_batch_tp = np.transpose(label_batch, (0, 2, 1))
                    label_tosave = np.reshape(label_batch, [HEIGHT,WIDTH])
                    #pred_batch_tp = np.transpose(pred_batch, (0, 2, 1))
                    pred_tosave = np.reshape(pred_batch, [HEIGHT,WIDTH])
                    print("test_itr:",test_itr)
                    # tep  = test_pred_annotation[0, 30, :, 0]
                    #np.savetxt('pred30.csv', tep, delimiter=',')
                    #np.savetxt('dice_smi_co.csv',test_dice_coe, delimiter=',')
                    utils.save_imgs(test_itr, label_tosave, pred_tosave, itr)
                Testl = Testl/TEST_RAW
                test_summary_str = sess.run(loss_test_op,  feed_dict = {Testloss:Testl})
                print (test_i,':',Testl)
                summary_writer.add_summary(test_summary_str, test_i)
                
                
#                Dise similarity coefficient
                DSC = utils.Accuracy_Measure(itr)
                DSC_Summary_str = sess.run(DSC_op,  feed_dict = {DSC_batch:DSC})
                print (test_i,':',DSC)
                summary_writer.add_summary(DSC_Summary_str, test_i)

                
    elif FLAGS.Mode == 'Visualize':
        start = time.time()
        is_training = tf.placeholder(tf.bool, name = 'is_train') #BN: istraining
    
        image = tf.placeholder(tf.float32, shape = [None, SP_HEIGHT, SP_WIDTH, IMAGE_CHANNEL], name = 'input_img')    

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


        end = time.time()
        elapse = end - start
        print('elapse time is :', elapse)
        print ('finished!')


if __name__ == '__main__':
	tf.app.run()
