import logging
import os
import time
from ctypes import *
from tkinter import messagebox

import clr
import cv2
import numpy as np
from PIL import Image, ImageTk
from utils.utils import thread_execute

logger = logging.getLogger(__name__)


class StageController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.m30_device = None
        # self.m30_event = m30_event
        # self.m30_param = m30_param
        self.x_pos, self.y_pos = 0, 0
        self.curAcc = 5
        self.curVel = 2
        self._x, self._y = 0, 0
        logger.debug(f"Initialization done.")

    @thread_execute
    def connect(self):
        self.con_stat = "CONNECTING"
        self.m30_device = 1
        time.sleep(2.5)  # wait statements are important to allow settings to be sent to the device
        self.con_stat = "CONNECTED"

    @thread_execute
    def disconnect(self):
        self.m30_device = None
        self.con_stat = "NOT CONNECTED"

    @thread_execute
    def update_position(self):
        if self.m30_device and self.con_stat == "CONNECTED":
            self.x_pos = self._x
            self.y_pos = self._y

    @thread_execute
    def set_postion(self, new_x, new_y):
        if self.m30_device and self.con_stat == "CONNECTED":
            dst_x = self._x - new_x
            dst_y = self._y - new_y
            dst = np.sqrt(dst_x**2 + dst_y**2)
            time.sleep(0.25*dst)
            self._x = new_x
            self._y = new_y

    @thread_execute
    def home(self):
        if self.m30_device and self.con_stat == "CONNECTED":
            self.set_postion(0, 0)

    def get_status(self):
        self.update_position()
        return {
            "connection": self.con_stat,
            "x_pos": self.x_pos,
            "y_pos": self.y_pos,
        }
