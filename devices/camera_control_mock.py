import time
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os


class CameraController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.port = None
        self.rotation = 0.0
        self.placeholder_image_words = 20
        self.wordlist = self._get_wordlist()

    def connect(self):
        self.con_stat = "CONNECTING"
        time.sleep(2)  # Simulate delay
        self.con_stat = "CONNECTED"
        self.rigol = True  # Simulate successful connection

    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    def _get_wordlist(self):
        with open("gui/main_window.py", "r") as f:
            cnt = f.read()
        chrs = "\n.,:\'\"=()[]{}=-+_*"
        for c in chrs:
            cnt = cnt.replace(c, " ")
        cnt = "".join(char for char in cnt if char.isalpha)
        return cnt.split(" ")

    def _generate_placeholder_image(self):
        # this is not for joke, it has a real purpose!
        img = np.zeros((600, 800, 3), dtype=np.uint8)  # cv2 image with 800 x 600 dims, black
        self.placeholder_image_words += 2-np.random.randint(4)
        if self.placeholder_image_words < 5: self.placeholder_image_words = 5
        if self.placeholder_image_words > 100: self.placeholder_image_words = 100
        clr = 55 + np.random.randint(200)
        clr_var = 20
        for i in range(self.placeholder_image_words):
            clr_r = clr - clr_var + np.random.randint(2*clr_var)
            clr_g = clr - clr_var + np.random.randint(2*clr_var)
            clr_b = clr - clr_var + np.random.randint(2*clr_var)
            txt = np.random.choice(self.wordlist)
            size = 0.2 + np.random.random()
            img = cv2.putText(
                img,
                txt,
                (np.random.randint(0, 800), np.random.randint(0, 600)),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (clr_r, clr_g, clr_b), # clr
                1, # thickness
            )
        return img

    def get_image(self) -> Image:
        image = self._generate_placeholder_image()
        time.sleep(0.75)
        return Image.fromarray(image)

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
