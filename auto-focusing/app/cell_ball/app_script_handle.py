# -*- coding: utf-8 -*-

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
import alg.algorithm.service.auto_focus as auto_focus
import app.cell_ball.ui.tools.img_list as img_list

optics_process = app_optics_process.optics_process
OPTICS_ERROR_CODE = app_optics_process.OpticsProcessErrorCode

mutex = threading.Lock()


class Interpreter:

    def __init__(self, callback):
        
        self.save_image_path = ""


    def sc_get_definition(self):
        '''
        @dec Gets the definition of the current image

        '''

        img_list.b_process_task_run = True
        img_list.b_process = True
        while not img_list.b_prepare:
            time.sleep(0.002)   
        img_list.b_prepare = False 
        # Get the CCD video stream data
        im = img_list.list_for_process.pop(0)

        error_code, fixture = auto_focus.AutoFocus.tenengrad(im)
        img_list.list_for_process.append(im)    
        img_list.b_process = False
        img_list.b_process_task_run = False
        time.sleep(0.050)
        return "", fixture
    
    def rough_autofocus(self, times, step, z_state, z_now_pos, z_max):
        '''
        粗对焦过程
        '''
        count = 0
        max_fixture = 0
        max_index = 0
        while count < times:
            if z_state:
                step_status = optics_process._find_step_status()
                if z_max != 0 and step_status['z'] + step > z_max:
                    break
                self.ssc_z_move_up(step)
                z_now_pos += step
            else:
                self.ssc_z_move_down(step)
                z_now_pos -= step
            errcode, fixture = self.sc_get_definition()

            if fixture != None:
                if max_fixture < fixture:
                    max_fixture = fixture
                    max_index = count
                if count - max_index > 4:
                    break
                count = count + 1
                time.sleep(0.010)
            else:
                count = count + 1
                time.sleep(0.010)

    def fine_autofocus(self, times, step, z_state, z_now_pos, z_max):
        '''
        细对焦过程
        '''
        count = 0
        max_fixture = 0
        max_index = 0
        while count < times:
            if z_state:
                step_status = optics_process._find_step_status()
                if z_max != 0 and step_status['z'] + step > z_max:
                    break
                self.ssc_z_move_up(step)
                z_now_pos += step
            else:
                self.ssc_z_move_down(step)
                z_now_pos -= step
            errcode, fixture = self.sc_get_definition()

            if fixture != None:
                if max_fixture < fixture:
                    max_fixture = fixture
                    max_index = count
                if count - max_index > 4:
                    break
                count = count + 1
                time.sleep(0.010)
            else:
                count = count + 1
                time.sleep(0.010)
        #回到最优焦面
        if z_state:
            self.ssc_z_move_down(step*5)
        else:
            self.ssc_z_move_up(step*5)

    def sc_autofocus_fine(self, times:int, z_max:int, z_up_step:int):
        z_state = True
        
        reduce_step = int(z_up_step / 4)
        #先粗对焦，单步步长每次递减1/5
        while z_up_step >= reduce_step:
            step_status = optics_process._find_step_status()
            z_pos = int(step_status['z'])
            self.rough_autofocus(times, z_up_step, z_state, z_pos, z_max)
            z_up_step -= reduce_step
            z_state = not z_state
        #然后进行细对焦
        step_status = optics_process._find_step_status()
        z_pos = int(step_status['z'])
        self.fine_autofocus(times, int(reduce_step/2), z_state, z_pos, z_max)
        step_status = optics_process._find_step_status()
        return step_status['z']


    def ssc_z_move_up(self, step:int):
        '''
            @dec SYSTEM.  Control the z axis to go up.
        '''

        err_code, temp_data = optics_process.evt_optics_z_move_forward_by_step(step)

        if err_code == OPTICS_ERROR_CODE.OK:
            return "", None
        else:
            return "ssc_z_move_up, error", None

    def ssc_z_move_down(self, step:int):
        '''
            @dec SYSTEM.  Control the z axis to go down.
        '''

        err_code, temp_data = optics_process.evt_optics_z_move_backward_by_step(step)

        if err_code == OPTICS_ERROR_CODE.OK:
            return "", None
        else:
            return "ssc_z_move_down, error", None

    def ssc_soft_reset_z(self, start_pos, fine_mode=True):
        step_status = optics_process._find_step_status()  # Get the current position
        steps = step_status['z'] - start_pos

        if steps == 0:
            return "", None
        if steps > 0 :
            return self.ssc_z_move_down(steps)   # Control z axis motor movement
        else:
            return self.ssc_z_move_up(-steps)


    def auto_focus(self, start_pos, z_max, single_step=3000, max_times=60):
        '''
            Autofocus function, need to adapt to 3D culture imaging system.
        '''
        self.ssc_soft_reset_z(start_pos)
        time.sleep(0.15)
        z_step = self.sc_autofocus_fine(max_times, z_max, single_step)
        return z_step


if __name__ == "__main__":

    interpreter = Interpreter(None)## self.callback
