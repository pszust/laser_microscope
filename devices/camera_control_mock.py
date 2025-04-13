import time
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np


class CameraController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.port = None
        self.rotation = 0.0

        self.num = 0
        self.image_list = []
        colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
        for color in colors:
            self.image_list.append(Image.new("RGB", (800, 600), color=color))

    def connect(self):
        self.con_stat = "CONNECTING"
        time.sleep(2)  # Simulate delay
        self.con_stat = "CONNECTED"
        self.rigol = True  # Simulate successful connection

    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    def get_image(self) -> Image:
        image = self.image_list[self.num]
        self.num += 1
        if self.num == len(self.image_list):
            self.num = 0
        time.sleep(0.75)
        return image

    def get_status(self) -> dict:
        """Possible values are
        'connection':
            'CONNECTED',
            'CONNECTING',
            'UNKNOWN',
            'NOT CONNECTED'
        """
        return {
            "connection": self.con_stat,
        }
