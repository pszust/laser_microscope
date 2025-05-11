import os
import time
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
import logging

logger = logging.getLogger(__name__)


class CameraController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.port = None
        self.placeholder_image_words = 150
        self.wordlist = self._get_wordlist()
        logger.debug(f"Initialization done.")

    def connect(self):
        self.con_stat = "CONNECTING"
        time.sleep(2)  # Simulate delay
        self.con_stat = "CONNECTED"

    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    def _get_wordlist(self):
        with open("gui/main_window.py", "r") as f:
            cnt = f.read()
        chrs = "\n.,:'\"=()[]{}=-+_*"
        for c in chrs:
            cnt = cnt.replace(c, " ")
        cnt = "".join(char for char in cnt if char.isalpha)
        cnt = cnt.split(" ")
        cnt.extend(list(range(1, 100)))
        return cnt

    def _generate_placeholder_image(self):
        # this is not for joke, it has a real purpose!
        img = np.zeros((600, 800, 3), dtype=np.uint8)  # cv2 image with 800 x 600 dims, black
        self.placeholder_image_words += 5 - np.random.randint(10)
        if self.placeholder_image_words < 50:
            self.placeholder_image_words = 50
        if self.placeholder_image_words > 250:
            self.placeholder_image_words = 250
        clr = 55 + np.random.randint(200)
        clr_var = 20
        for i in range(self.placeholder_image_words):
            clr_r = clr - clr_var + np.random.randint(2 * clr_var)
            clr_g = clr - clr_var + np.random.randint(2 * clr_var)
            clr_b = clr - clr_var + np.random.randint(2 * clr_var)
            txt = np.random.choice(self.wordlist)
            size = 0.2 + np.random.random()
            thc = 1 if np.random.random() < 0.85 else np.random.choice([2, 3])
            img = cv2.putText(
                img,
                txt,
                (np.random.randint(0, 800), np.random.randint(0, 600)),
                cv2.FONT_HERSHEY_SIMPLEX,
                size,
                (clr_r, clr_g, clr_b),  # clr
                thc,  # thickness
            )
            # this is also not a joke - extensive calcs help spot the perfomance issues
            # TODO: add multiporcessing for time-requiring functions?
            if i % ((self.placeholder_image_words) // 4 + 1) == 0:
                img = cv2.GaussianBlur(img, (5, 5), 1)
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
