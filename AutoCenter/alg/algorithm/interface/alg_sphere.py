# -*- coding:utf-8 -*-

import cv2

import numpy as np
import json
import math
import datetime
import sys

sys.path.append(r'./')
import alg.algorithm.service.sphere_detection as sphere_detection

class IMageError:
    '''
    图像错误代码
    '''
    IMage_read_Ok = 'IMage_read_Ok'
    IMage_None = 'IMage_None'
    IMage_No_Aim = "IMage_read_Ok but Don't the coutour"
    IMage_No_Any_Aim = "IMage_read_Ok but Don't any coutour"
    IMage_Have_Aim = "IMage_read_Ok and have the coutour"

class AlgSphers(object):

    def __init__(self, image_path=None, image_data=None, field='bright_1', context={}):
        '''
        :param param_file: 调用图像处理参数列表
        '''

        self.field = field
        if self.field == 'bright_1':
            self.param_file = {"blockSize": 11, "C": 7, "stdDevScale": 1.4, "ksize": 15, "X": 8, "Y": 8, "N": 5}
            self.show, self.gray_type, self.binary_method = False, 'gray', 'adaptive'
        elif self.field == 'center':
            self.param_file = {"blockSize": 17, "C": 3, "stdDevScale": 1.4, "ksize": 5, "X": 7, "Y": 7, "N": 5}
            self.show, self.gray_type, self.binary_method = False, 'gray', 'otsu'

        self.service = sphere_detection.SphereDetection(image_path, image_data, self.param_file)
        self.image = self.service.image
        self.Rmax = 0
        self.Rmin = 0
        self.resi = self.service.resize_by_value(self.image)
        # 设备上下文
        self.context = context

    def img_process(self, field, show=False):

        img_code, image = self.service.im_read()
        self.show = show
        if img_code == IMageError.IMage_read_Ok and self.field == 'bright_1':
            resizeByValue = self.service.resize_by_value(self.image, 800, 604)
            changing = self.service.changing(resizeByValue, self.gray_type)
            if self.show:
                self.service.show_image('changing', changing)
            binary_threshold = self.service.binary_threshold(changing, self.binary_method)
            if self.show:
                self.service.show_image('binary_threshold', binary_threshold)
            open_img = self.service.open(binary_threshold)
            if self.show:
                self.service.show_image('open', open_img)
            sobel_img = self.service.sobel(open_img)
            if self.show:
                self.service.show_image('sobel', sobel_img)
            sort_contour = self.service.findContours(sobel_img)

        elif img_code == IMageError.IMage_read_Ok and self.field == 'center':
            resizeByValue = self.service.resize_by_value(self.image, 800, 604)

            bilateral_img = cv2.bilateralFilter(resizeByValue, 40, 75, 75)
            changing = self.service.changing(bilateral_img, self.gray_type)

            if self.show:
                self.service.show_image('changing', changing)
            binary_threshold = self.service.binary_threshold(changing, self.binary_method)
            if self.show:
                self.service.show_image('binary_threshold', binary_threshold)
            medianBlur_img = self.service.medianBlur(binary_threshold)
            if self.show:
                self.service.show_image('medianBlur', medianBlur_img)

            sobel_img = self.service.sobel(medianBlur_img)
            if self.show:
                self.service.show_image('sobel', sobel_img)
            sort_contour = self.service.findContours(sobel_img)

        if sort_contour == None:
            return 0, 0, self.image.width, self.image.height

        else:
            post_contour = sort_contour[0]
            if len(sort_contour) > 1:
                ratio0, area0 = self.service.contour_du(sort_contour[0])
                ratio1, area1 = self.service.contour_du(sort_contour[1])
                if area1/area0 > 0.8 and ratio0 < ratio1:
                    post_contour = sort_contour[1]

            resizeByValue = self.service.resize_by_value(self.image, 800, 604)
            self.Rmin, self.Rmax, cX, cY, maxpt, minpt = self.service.contourCenter_impr(resizeByValue, post_contour)

            return cX, cY, self.image.width, self.image.height


if __name__ == "__main__":
    pass