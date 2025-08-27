import logging
import os
import time
from ctypes import *
from tkinter import messagebox

import clr
import cv2
import numpy as np
from PIL import Image, ImageTk

from devices.TC300_COMMAND_LIB import *
from utils.consts import LabJackConsts
from utils.utils import thread_execute

logger = logging.getLogger(__name__)


class LabjackController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.height = 0.0
        self.labjack = None
        logger.debug(f"Initialization done.")

    @thread_execute
    def connect(self):
        self.con_stat = "CONNECTING"
        time.sleep(2.1)

        self.con_stat = "CONNECTED"

    @thread_execute
    def disconnect(self):
        self.labjack = None
        self.con_stat = "NOT CONNECTED"

    @thread_execute
    def update_height(self):
        pass

    @thread_execute
    def set_height(self, value):
        if value < LabJackConsts.MIN_POS or LabJackConsts.MAX_POS < value:
            # TODO: proper logging needs to implemented
            err_msg = f"Requested z-value = {value:.3f} for labjack is outside the range "
            err_msg += f"({LabJackConsts.MIN_POS} to {LabJackConsts.MAX_POS})"
            print(err_msg)
        elif self.labjack and self.con_stat == "CONNECTED":
            self.height = value

    @thread_execute
    def home(self):
        pass

    @thread_execute
    def move_relative(self, value):
        self.update_height()
        time.sleep(0.1)
        new_height = self.height + value
        self.set_height(new_height)

    def get_status(self):
        self.update_height()
        return {
            "connection": self.con_stat,
            "height": self.height,
        }
