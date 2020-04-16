# -*- coding:utf-8 -*-
'''
Created on 2018年01月02日
@author: huanhuan
'''
import cv2
import numpy as np
import json
import math
from skimage import data,filters,segmentation,measure,morphology,color
import sys
sys.path.append(r'./')

Not_Read_Img = cv2.imread(r'')
class Color:
    COLOR_BLUE = (255, 0,  0)
    COLOR_RED  = (0, 0, 255)
    COLOR_GREEN  = (0, 255, 0)
    COLOR_SKY_BLUE = (255, 255, 0)

class IMageError:
    '''
    图像错误代码
    '''
    IMage_read_Ok = 'IMage_read_Ok'
    IMage_None = 'IMage_None'
    IMage_Area_None = 'no aim area'


class SphereDetection(object):

    def __init__(self, image_path=None, image_data=None, param_file=None, method=None):
        self.method = method
        if not image_path:
            self.image = image_data
            
        else:
            # self.image = cv2.imread(image_path)
            self.image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

        self.param_file = param_file
        if self.param_file == None:
            self.param_file = {
                "blockSize":11,
                "C":7,
                "stdDevScale":1.4,
                "ksize":5,
                "X":8,
                "Y":8,
                "N":5
            } 

    def im_read(self):
        if not isinstance(self.image, np.ndarray):
            return IMageError.IMage_None, None
        else:
            return IMageError.IMage_read_Ok, self.image

    def resize_by_value(self, image, width=400, height=300, inter=cv2.INTER_AREA):
        '''
        resize_by_value: 按固定长宽缩放
        '''
        # 初始化缩放比例，并获取图像尺寸
        dim = None
        (h, w) = image.shape[:2]
        # 如果宽度和高度均为0，则返回原图
        if width is None and height is None:
            return image
        # 宽度是0
        if width is None:
            # 则根据高度计算缩放比例
            r = height / float(h)
            dim = (int(w * r), height)
        # 如果高度为0
        else:
            # 根据宽度计算缩放比例
            r = width / float(w)
            dim = (width, int(h * r))
        # 缩放图像
        resized = cv2.resize(image, dim, interpolation=inter)
        # 返回缩放后的图像
        return resized

    def changing(self, image, type='gray'):
        '''
        rbg2gray
        type = 'gray' or 'hsv'
        '''
        if type == 'gray':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif type == 'hsv':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif type =='real':
            return image
        elif type =='max':  # 取每个通道的最大值
            B, G, R = cv2.split(image)
            MAX = B.copy()
            for i in range(B.shape[0]):
                for j in range(B.shape[1]):
                    MAX[i, j] = max(B[i, j], G[i, j], R[i, j])
            return MAX
        return None

    def binary_threshold(self, src, method):
        '''
        blockSize, C
        mean_threshold(二值化)
        src 为输入图像；blockSize: b的值；C 为从均值中减去的常数，用于得到阈值；
        return 返回二值化的图像
        '''
        blockSize = self.param_file['blockSize']
        C = self.param_file['C']

        if method == 'otsu':
            threshold, imgOtsu = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return imgOtsu

        elif method == 'adaptive':
            _src = src
            if src.ndim == 3:
                _src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, blockSize, C)

        elif method == 'adaptive_mean':
            return cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)


    def medianBlur(self, image):
        '''
            ksize
            medianBlur 中值滤波
            非线性滤波，对消除椒盐噪声很有用
        '''
        ksize = self.param_file['ksize']
        dst = cv2.medianBlur(image, ksize)
        return dst

    def open(self, image):
        '''
         X, Y
        :param image:
        :param (X,Y): 核函数大小，单细胞球时（6,6）、多细胞球时（8, 7）效果较好。
        :return:
        '''
        X = self.param_file['X']
        Y = self.param_file['Y']
        kernel = np.ones((X, Y), np.uint8)   
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        return opening

    def sobel(self, image):
        '''
        sobel
        '''
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

        sobelx = np.uint8(np.absolute(sobelx))
        sobely = np.uint8(np.absolute(sobely))
        sobelcombine = cv2.bitwise_or(sobelx,sobely)
        return sobelcombine

    def findContours(self, image):
        '''
        findContours
        '''
        _, contours, _ = cv2.findContours(image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            return IMageError.IMage_Area_None
        else:
            sort_contour = sorted(contours, key=cv2.contourArea, reverse=True)
        return sort_contour

    def contour_du(self, max_contour):
        '''
        改进方法，计算圆度
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

    def contourCenter_impr(self, image, max_contour):
        '''
        改进方法，利用质心找最小二乘圆
        '''

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

    def moMents(self, sort_contour):
        '''
            Calculate the moments of each area
        '''
        center_list = []
        for i in range(len(sort_contour)):
            M = cv2.moments(sort_contour[i])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centor_point = [i, cX, cY]
            center_list.append(centor_point)

        return center_list

    def counter_area(self, contour):
        area = cv2.contourArea(contour)
        #  Equivalent Diameter 与轮廓物体面积相等的圆的直径
        equi_diameter = np.sqrt(4 * area / np.pi)
        return area, equi_diameter

    def show_image(self, window_name, show_image):
        '''
        show_image
        '''
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, show_image)
        cv2.waitKey(0)

    def save_image(self, filename, img):
        '''
        show_image
        '''
        cv2.imencode('.tiff', img)[1].tofile(filename)
        cv2.destroyAllWindows()

    def drawContours(self, image, contours, cidx=-1):
        '''
            drawContours
        '''
        image_contour = cv2.drawContours(image, contours, cidx, Color.COLOR_GREEN, 1)
        return image_contour

    def write(self, path=None, imged=None):
        if imged is not None:
            cv2.imencode('.tiff', imged)[1].tofile(path)
            

    def equalize_hist(self, image):
        if image.ndim == 3:
            im_r = cv2.equalizeHist(image[:,:,0])
            im_g = cv2.equalizeHist(image[:,:,1])
            im_b = cv2.equalizeHist(image[:,:,2])
            im = cv2.merge([im_r, im_g, im_b])
            return im
        else:
            return cv2.equalizeHist(image[:,:,0])


if __name__ == "__main__":

    pass
