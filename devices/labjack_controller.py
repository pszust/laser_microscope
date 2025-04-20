import os
import time
from tkinter import messagebox

import cv2
import numpy as np
from utils.utils import thread_execute
from PIL import Image, ImageTk


class LabjackController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.height = 0.0

    @thread_execute
    def connect(self):
        self.con_stat = "CONNECTING"
        time.sleep(2)  # Simulate delay
        self.con_stat = "CONNECTED"
        self.labjack = True  # Simulate successful connection

    @thread_execute
    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    @thread_execute
    def update_height(self):
        # call the labjack to get position
        # update self.height
        time.sleep(0.15)

    @thread_execute
    def set_height(self, value):
        # send command to lebjack
        time.sleep(1)
        self.height = value  # in real this line will not be present here

    def move_relative(self, value):
        self.update_height()
        new_height = self.height + value
        # send command to labjack
        self.set_height(new_height)

    def get_status(self):
        self.update_height()
        return {
            "connection": self.con_stat,
            "height": self.height,
        }
