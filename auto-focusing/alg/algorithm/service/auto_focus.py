# -*- coding:utf-8 -*-
'''
Created on 2018年01月02日
@author: huanhuan
'''
import cv2
import numpy as np
import math
import datetime

class IMageError:
    '''
    图像错误代码
    '''
    IMage_Ok = 'IMage_Ok'
    IMage_None = 'IMage_None'
    IMage_Error = 'IMage_Error'
    IMage_List_Ok = 'IMage_List_Ok'
    IMage_List_Empyt = 'IMage_List_Empyt'
    IMage_List_Error = 'IMage_List_Error'


class AutoFocus(object):

    @staticmethod
    def tenengrad(image):
        '''
        Tenengrad梯度方法利用Sobel算子分别计算水平和垂直方向的梯度，
        同一场景下梯度值越高，图像越清晰.
        :return: errcode, data
        '''
        try:
            _, _, c = image.shape
            if c > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

            absX = cv2.convertScaleAbs(sobelx)  # 转回uint8
            absY = cv2.convertScaleAbs(sobely)
            absX = np.square(absX)  
            absY = np.square(absY)
            sobelcombine = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            mean_gray = np.sum(sobelcombine)
            return IMageError.IMage_Ok, mean_gray

        except Exception as err:
            return IMageError.IMage_Error, None

    @staticmethod
    def laplacian(image):

        '''
        Laplacian梯度是一种求图像梯度的方法，图像越清晰，边缘梯度越大
        :return: errcode, data
        '''
        try:
            if image.ndim > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap = np.uint8(np.absolute(lap))
            mean_gray = np.mean(lap)

            return IMageError.IMage_Ok, mean_gray

        except Exception as err:
            return IMageError.IMage_Error, None

    @staticmethod
    def variance(image):
        '''
         :return: errcode, data
        '''
        try:
            if image.ndim > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            mean, stddev = cv2.meanStdDev(gray)  # 均值和标准差
            w, h = gray.shape
            st = 0.0
            for i in range(w):
                for j in range(h):
                    st += (gray[i][j] - mean[0][0]) ** 2
            st1 = st / (w * h)  # 方差

            return IMageError.IMage_Ok, st1

        except Exception as err:
            return IMageError.IMage_Error, None

    @staticmethod
    def sml(image, step=2):
        '''
        改进的拉普拉斯算子
        :return: errcode, data
        '''
        try:
            if image.ndim > 2:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            w, h = gray.shape
            ML_list = []
            for i in range(step, w - step):
                for j in range(step, h - step):
                    G = abs(2 * gray[i][j] - gray[i - step][j] - gray[i + step][j]) + abs(
                        2 * gray[i][j] - gray[i][j - step] - gray[i][j + step])
                    ML_list.append(G)
            L0 = sum(ML_list)  # 可设置阈值T，大于阈值T的参加汇总

            return IMageError.IMage_Ok, L0

        except Exception as err:
            return IMageError.IMage_Error, None



if __name__ == "__main__":
    image = cv2.imread(r'H:\snap_picture\20180313155450.bmp')
    image = cv2.resize(image, (100, 300))
    AutoFocus.grayEntropy(image)
    print("auto_focus_service")
