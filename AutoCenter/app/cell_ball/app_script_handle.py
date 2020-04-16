# -*- coding: utf-8 -*-

# 添加命令
#   self.check      ：添加check入口
#   def _check_xxx  : 定义check函数
#

import os
import re
import sys
import cv2
import numpy as np
import time
import datetime
import json
import threading 

sys.path.append(r'./')

import app.cell_ball.app_optics_process as app_optics_process
import app.cell_ball.ui.tools.img_list as img_list
import alg.algorithm.interface.alg_sphere as alg_sphere
import alg.algorithm.service.sphere_detection as sphere_detection

optics_process = app_optics_process.optics_process

mutex = threading.Lock()


class Interpreter:

    ret_list = []

    def __init__(self, callback):
        self.save_image_path = ""


    def sc_detection(self, flag='bright_1', center=False):
        '''
        @dec 拍摄图像. \
        @val times: 连拍次数
        '''

        img_list.b_process_task_run = True
        img_list.b_process = True
        while not img_list.b_prepare:
            time.sleep(0.002)   
        img_list.b_prepare = False  

        # Get the CCD video stream data
        im = img_list.list_for_process.pop(0)
        alg = alg_sphere.AlgSphers(None, im, flag)

        x, y, width, height = alg.img_process(field=flag, show=False)
        img_list.b_process = False
        return x, y, width, height

    def ssc_move_everywhere_xy(self, x:int, y:int):

        movement_control.tiny_move_to(x, y)
        return "ssc_tiny_move_to, test", None


    def _off_2_center(self, ratio):

        x, y, width, height= self.sc_detection('center', True)
        if x == 0 :
            return False
        else:
            pix_offset_x = x - (width/2)
            pix_offset_y = y - (height/2)
            offset_x = pix_offset_x * ratio
            offset_Y = pix_offset_y * ratio
            self.ssc_move_everywhere_xy(int(offset_x), int(offset_Y))
            return True

                
    def auto_center(self, pixel_to_step_ratio):
        ret = self._off_2_center(pixel_to_step_ratio)


if __name__ == "__main__":

    interpreter = Interpreter(None)## self.callback

