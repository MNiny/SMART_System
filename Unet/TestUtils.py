#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 05:11:10 2018

@author: Niny
"""

import tensorflow as tf
import numpy as np
import scipy.misc as misc
from six.moves import urllib
import scipy
import cv2
from DataProcessor import Processor as pr
import xlwt
import math

from scipy.interpolate import spline, UnivariateSpline, Akima1DInterpolator, PchipInterpolator
import matplotlib.pyplot as plt

import os

#IMAGE_SIZE=256
# IMAGE_HEIGHT=972
# IMAGE_WIDTH=1290

IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

recordPath = os.getcwd()+'/'

'''
Verification
input:img,lbl
output:pred,groungtruth
'''
#read images
def read_img_lbl(imgname,lblname):
    
    img=cv2.imread(imgname)
    lbl=cv2.imread(lblname)
    imgs = pr.Intensity(img)
    lbls = np.dot(lbl[...,:3], [0.299, 0.587, 0.114])//255
    lbls = np.uint8(lbls)
        
    img_batch = pr.down_sample(imgs)
    img_batch = np.float64(img_batch)
    lbl_batch = pr.down_sample(lbls)
    
    img_batch_norm = pr.contrast_normalization(img_batch)
    
    img_all = img_batch_norm.reshape([1,IMAGE_HEIGHT,IMAGE_WIDTH,1])
    lbl_all = lbl_batch.reshape([1,IMAGE_HEIGHT,IMAGE_WIDTH,1])
    
    return img_all, lbl_all

#Save images
def save_Veri(test_num, label_batch, pred_batch, savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    label_img_mat = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    pred_img_mat = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    for i in range (IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            if label_batch[i, j] == 0:
                label_img_mat[i, j, 0] = label_img_mat[i, j, 1] =  label_img_mat[i, j, 2] = 128 
            if label_batch[i, j] == 1:
                label_img_mat[i, j, 0] = 255
                label_img_mat[i, j, 1] = 69
                label_img_mat[i, j, 2] = 0     
                
            if pred_batch[i, j] == 0:
                pred_img_mat[i, j, 0] = pred_img_mat[i, j, 1] =  pred_img_mat[i, j, 2] = 128  
            if pred_batch[i, j] == 1:
                pred_img_mat[i, j, 0] = 255
                pred_img_mat[i, j, 1] = 69
                pred_img_mat[i, j, 2] = 0    
    label_img_mat=np.uint8(label_img_mat)            
    pred_img_mat=np.uint8(pred_img_mat) 
    
    scipy.misc.imsave(savepath + '\\' + test_num + '-mask.bmp', label_img_mat)
    scipy.misc.imsave(savepath + '\\' + test_num + '-pred.bmp', pred_img_mat)



'''
Test
input:img
output:pred
'''
#read images
def read_img(imgname):
    
    img=cv2.imread(imgname)
    
    imgs = pr.Intensity(img)
    
    img_batch = pr.down_sample(imgs)
    img_batch = np.float64(img_batch)
    
    
    img_batch_norm = pr.contrast_normalization(img_batch)
    
    img_all = img_batch_norm.reshape([1,IMAGE_HEIGHT,IMAGE_WIDTH,1])
    
    return img_all

def contour_ratio(max_contour):
    '''
    计算圆度
    '''
    if type(max_contour) is str:
        return None, None, None, None, None, None
    else:
        max_contour = np.squeeze(max_contour)
        ma = max_contour.max(axis=0)
        mi = max_contour.min(axis=0)
        width, height = ma - mi
        if width < height:
            radio = width / height
        else:
            radio = height / width
        # print(radio)
        area = cv2.contourArea(max_contour)
        return radio, area

def contourCenter_impr(max_contour):
    '''
    利用质心找最小二乘圆
    @var min: 最小半径
    @var max: 最大半径
    @var cX: 质心X轴位置
    @var cY: 质心Y轴位置
    @var maxpt: 距离质心最远点的坐标
    @var minpt: 距离质心最近点的坐标
    '''
    # print("M = cv2.moments(max_contour):", type(max_contour))
    if type(max_contour) is str:
        return None, None, None, None, None, None
    else:
        M = cv2.moments(max_contour)
        cX = M["m10"] / M["m00"]
        cY = M["m01"] / M["m00"]
        centor_point = [cX, cY]
        max, min = 0, 1600
        for pt in max_contour:
            dist = np.linalg.norm(pt[0] - centor_point)
            if dist > max:
                max = dist   
                maxpt = pt[0]
            if dist < min:
                min = dist
                minpt = pt[0]
        return min, max, cX, cY, maxpt, minpt


def saveImage(imagePath, pred_batch, imgname, test_save_path):
    if not os.path.exists(test_save_path):
        os.mkdir(test_save_path)
    # src = cv2.imread(imagePath)
    src = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8), -1)
    dim = (800, 604)
    src = cv2.resize(src, dim)
    pred_tosave = cv2.resize(pred_batch, dim)
    

    sobelImage = sobel(pred_tosave)

    #show_image('sobelImage', sobelImage)

    sortContour = findContours(sobelImage)
    if sortContour == None:
        return None

    #选取最优轮廓
    postContour = sortContour[0]
    if len(sortContour) > 1:
        ratio0, area0 = contour_ratio(sortContour[0])
        ratio1, area1 = contour_ratio(sortContour[1])
        if area1/area0 > 0.8 and ratio0 < ratio1:
            postContour = sortContour[1]
    
    Rmin, Rmax, cX, cY, maxpt, minpt = contourCenter_impr(postContour)

    #在原图上勾画轮廓
    cv2.line(src, (int(cX), int(cY)), tuple(maxpt), (255, 255, 0), 1)
    cv2.line(src, (int(cX), int(cY)), tuple(minpt), (255, 255, 0), 1)
    cv2.circle(src, (int(cX), int(cY)), int(Rmax), (255, 255, 0), 1)
    cv2.circle(src, (int(cX), int(cY)), int(Rmin), (255, 255, 0), 1)
    cv2.circle(src, (int(cX), int(cY)), 3, (255, 255, 0), -1)

    cv2.drawContours(src, postContour, -1, (0, 255,  0), 1)
    # cv2.imwrite(test_save_path + imgname, src)
    
    img_save_path=test_save_path +'img/'
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)
    cv2.imencode('.tiff', src)[1].tofile(img_save_path + imgname)

    #在白板上勾画轮廓
    whiteBoard = np.zeros([604, 800, 3], np.uint8) + 255
    cv2.line(whiteBoard, (int(cX), int(cY)), tuple(maxpt), (255, 255, 0), 1)
    cv2.line(whiteBoard, (int(cX), int(cY)), tuple(minpt), (255, 255, 0), 1)
    cv2.circle(whiteBoard, (int(cX), int(cY)), int(Rmax), (255, 255, 0), 1)
    cv2.circle(whiteBoard, (int(cX), int(cY)), int(Rmin), (255, 255, 0), 1)
    cv2.circle(whiteBoard, (int(cX), int(cY)), 3, (255, 255, 0), -1)
    cv2.drawContours(whiteBoard, postContour, -1, (0, 0,  255), 1)

    newBoardName = imgname.replace('cap', 'circle')
    # cv2.imwrite(test_save_path + imgname, src)
    
    dec_save_path=test_save_path+ 'dec/'
    if not os.path.exists(dec_save_path):
        os.mkdir(dec_save_path)
    cv2.imencode('.tiff', whiteBoard)[1].tofile(dec_save_path + newBoardName)
    
    Red_per,Standard_perimeter ,Equi_diameter = Redundant_perimeter(postContour)
    polar = RectToPolar(postContour,cX,cY,Equi_diameter/2)
    polar_r_int,polar_theta_int = Interpolator(polar,Standard_perimeter)
    
#    save_data_d(test_save_path,postContour,cX,cY,imgname,Rmin)
    save_data_txt(test_save_path, imgname, Red_per, Standard_perimeter)
    save_data_p(test_save_path, polar_r_int,polar_theta_int, imgname)

def save_data_txt(test_save_path, imgname, Red_per, Standard_perimeter):
     data_save_path = test_save_path+'Data_d/'
     if not os.path.exists(data_save_path):
         os.mkdir(data_save_path)
     f1 = open(data_save_path+imgname[:-3]+'txt','w')
     f1.write(str(Red_per))
     f1.close()
     
#     f1 = open(data_save_path+imgname[:-4]+'per.txt','w')
#     f1.write(str(Standard_perimeter))
#     f1.close()

def save_data_d(test_save_path, data,cX,cY,imgname,Rmin):
    data_save_path = test_save_path+'Data_d/'
    if not os.path.exists(data_save_path):
      os.mkdir(data_save_path) 
    #
    wb = xlwt.Workbook()
    ws = wb.add_sheet('111')
    for i in range(data.shape[0]):
        x=data[i][0][0]
        y=data[i][0][1]
        d_R=((x-cX)*(x-cX)+(y-cY)*(y-cY))**0.5-Rmin
        ws.write(i,0,d_R)
    points_save_path = data_save_path + 'Points/'
    if not os.path.exists(points_save_path):
        os.mkdir(points_save_path)
    wb.save(points_save_path+imgname[:-3]+'xls')
    
#    #
#    wb2 = xlwt.Workbook()
#    ws2 = wb2.add_sheet('111')
#    for i in range(data.shape[0]-1):
#        x1=data[i][0][0]
#        y1=data[i][0][1]
#        x2=data[i+1][0][0]
#        y2=data[i+1][0][1]
#        
#        radian=((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))**0.5/Equi_radius*360
#        ws2.write(i,0,radian)
#        
#    radians_save_path=data_save_path+'Radians/'
#    if not os.path.exists(radians_save_path):
#        os.mkdir(radians_save_path)
#    wb2.save(radians_save_path+imgname[:-3]+'xls')
    
#def save_data_p(test_save_path, data,imgname):
#    data_save_path = test_save_path+'Data_d/'
#    if not os.path.exists(data_save_path):
#      os.mkdir(data_save_path) 
#    #
#    wbp = xlwt.Workbook()
#    wsp = wbp.add_sheet('111')
#    for i in range(data.shape[0]):
#        r=data[i][0]
#        theta=data[i][1]
#        wsp.write(i,0,r)
#        wsp.write(i,1, theta)
#        
#    points_save_path = data_save_path + 'Polars/'
#    if not os.path.exists(points_save_path):
#        os.mkdir(points_save_path)
#    wbp.save(points_save_path+imgname[:-3]+'xls')
    
    
def save_data_p(test_save_path, data_r,data_theta,imgname):
    data_save_path = test_save_path+'Data_d/'
    if not os.path.exists(data_save_path):
      os.mkdir(data_save_path) 
    #
    wbp = xlwt.Workbook()
    wsp = wbp.add_sheet('111')
    for i in range(data_r.shape[0]):
        r_int=data_r[i]
        theta_int=data_theta[i]
        wsp.write(i,0,r_int)
        wsp.write(i,1, theta_int)
        
    points_save_path = data_save_path + 'Polars_Int/'
    if not os.path.exists(points_save_path):
        os.mkdir(points_save_path)
    wbp.save(points_save_path+imgname[:-3]+'xls')
    
    
    
    
def Redundant_perimeter(contour):
    area = cv2.contourArea(contour)
    equi_diameter = np.sqrt(4*area/np.pi)
    perimeter = cv2.arcLength(contour,True)
    Red_per=(perimeter-(np.pi*equi_diameter))/(np.pi*equi_diameter)
    return Red_per, np.pi*equi_diameter,equi_diameter



def RectToPolar(data,Cx,Cy,sta_R):
    
    polar=[]
    
    for i in range(data.shape[0]):
        xi=data[i][0][0]
        yi=data[i][0][1]
        
        x=xi-Cx
        y=yi-Cy
        
        r=pow((pow(x,2)+pow(y,2)),0.5)
        if x==0:
            if(y>=0):
                theta = np.pi/2
            else:
                theta = -np.pi/2
        else:
            if x>0:
                theta = math.atan(y/x)
            if x<0:
                theta = math.atan(y/x)+np.pi
        theta +=np.pi/2 
        polar.append([r,theta*sta_R])
    
    return np.array(polar)

def Interpolator(polar,Standard_perimeter):
        r=[]
        theta=[]
        polat_t = polar[np.lexsort(polar.T)]
        for i in range(polat_t.shape[0]):
            r.append(polat_t[i][0])
            theta.append(polat_t[i][1])
        r=np.array(r)
        theta=np.array(theta)
        
        theta_smooth = np.linspace(min(theta), max(theta),int(Standard_perimeter*2))
#        bi = Akima1DInterpolator(theta,r)
        bi = PchipInterpolator(theta,r)
        r_smooth = bi(theta_smooth)
        
        return r_smooth,theta_smooth
            



#Save images
def save_Test(test_num, pred_batch,savepath):
    
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    pred_img_mat = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    for i in range (IMAGE_HEIGHT):
        for j in range(IMAGE_WIDTH):
            if pred_batch[i, j] == 0:
                pred_img_mat[i, j, 0] = pred_img_mat[i, j, 1] =  pred_img_mat[i, j, 2] = 128 
            if pred_batch[i, j] == 1:
                pred_img_mat[i, j, 0] = 255
                pred_img_mat[i, j, 1] = 69
                pred_img_mat[i, j, 2] = 0    
    pred_img_mat=np.uint8(pred_img_mat) 
    scipy.misc.imsave(savepath + test_num + '-pred.bmp', pred_img_mat)



def sobel(image):
    '''
    sobel算子提取边缘
    '''
    if image.ndim > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # todo 增加几个参数 http://blog.csdn.net/sunny2038/article/details/9170013
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))
    sobelcombine = cv2.bitwise_or(sobelx,sobely)
    return sobelcombine

def findContours(image):
    '''
    findContours提取轮廓，根据面积大小进行排序
    http://blog.csdn.net/mokeding/article/details/20153325
    '''
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return None
    else:
        sort_contour = sorted(contours, key=cv2.contourArea, reverse=True)
    return sort_contour


def show_image(window_name, show_image):
    '''
    show_image
    '''
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, show_image)
    cv2.waitKey(0)