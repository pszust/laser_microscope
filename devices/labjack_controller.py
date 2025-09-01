import logging
import os
import time
from ctypes import *
from tkinter import messagebox

import clr
import cv2
import numpy as np
from PIL import Image, ImageTk
from pylablib.devices import Thorlabs

from utils.consts import LabJackConsts
from utils.utils import thread_execute

logger = logging.getLogger(__name__)

STEPS2MM = 1228800


class LabjackController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.height = 0.0
        self.device = None
        logger.debug(f"Initialization done.")

    @thread_execute
    def connect(self):
        self.con_stat = "CONNECTING"
        self.device = Thorlabs.KinesisMotor("49499304")
        time.sleep(0.25)  # wait statements are important to allow settings to be sent to the device
        logger.info(f"Connected to labjack")
        self.con_stat = "CONNECTED"

    @thread_execute
    def disconnect(self):
        self.device = None
        self.con_stat = "NOT CONNECTED"

    @thread_execute
    def update_height(self):
        if self.device and self.con_stat == "CONNECTED":
            self.height = round(self.device.get_position() / STEPS2MM, 5)

    @thread_execute
    def set_height(self, value):
        if not self.device.is_moving():
            if value < LabJackConsts.MIN_POS or LabJackConsts.MAX_POS < value:
                err_msg = f"Requested z-value = {value:.3f} for labjack is outside the range "
                err_msg += f"({LabJackConsts.MIN_POS} to {LabJackConsts.MAX_POS})"
                logger.warning(err_msg)
            elif self.device and self.con_stat == "CONNECTED":
                self.device.move_to(int(value * STEPS2MM))
        else:
            logger.warning("Labjack device is currently moving, command ignored")

    @thread_execute
    def home(self):
        self.device.home()

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
