import logging
import os
import time
from multiprocessing import Event, Process, RawArray, Value
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from utils.consts import CamConsts

logger = logging.getLogger(__name__)


class CameraController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.event = Event()
        # self.sharr = RawArray('c', 448*800*3)  # holds camera frame
        self.sharr = RawArray("c", CamConsts.SHAPE[1] * CamConsts.SHAPE[0] * 3)  # holds camera frame
        self.config_arr = RawArray("i", (6, 8, -1))  # hold the camera configuration (gain, expo)
        self.camparam_arr = RawArray(
            "i", (0, 0, 0)
        )  # hold the camera configuration (gain, expo, invert_colors)

        logger.debug(f"Initialization done.")

    def connect(self):
        self.con_stat = "CONNECTING"
        self.cam_reader = CamReader(self.event, self.sharr, self.config_arr, self.camparam_arr)
        self.cam_reader.start()
        self.con_stat = "CONNECTED"

    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    def get_image(self) -> Image:
        frame = np.frombuffer(self.sharr, dtype=np.uint8).reshape(CamConsts.SHAPE[1], CamConsts.SHAPE[0], 3)
        return Image.fromarray(frame)

    def save_image(self, path: str):
        frame = self._generate_placeholder_image()
        image = Image.fromarray(frame)
        image.save(path)
        logger.info(f"Saving image as {path}")

    def save_as_array(self, path: str):
        frame = self._generate_placeholder_image()
        np.save(path, frame)
        logger.info(f"Saving raw array as {path}")

    def exit_camera(self):
        self.event.set()  # to close CamReader
        logger.info("Initiated camera exit")

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


class CamReader(Process):
    #     override the constructor
    def __init__(self, event, sharr, config_arr, camparam_arr):
        # execute the base constructor
        Process.__init__(self)
        # initialize integer attribute
        self.event = event
        self.config_arr = config_arr
        self.camparam_arr = camparam_arr
        self.gain = self.config_arr[0]
        self.expo = self.config_arr[1]
        self.expoAbs = self.config_arr[2]
        self.sharr = sharr
        self.data = Value("i", 0)

        self.placeholder_image_words = 150
        self.wordlist = self._get_wordlist()

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

    # override the run function
    def run(self):
        final_img = np.zeros((960, 1280, 4), dtype=np.ubyte)
        while True:
            try:
                final_img = self._generate_placeholder_image()
            except:
                pass

            # copy frame to shared array (sharr)
            self.sharr_np = np.frombuffer(self.sharr, dtype=np.uint8).reshape(
                *[CamConsts.SHAPE[1], CamConsts.SHAPE[0], 3]
            )
            np.copyto(self.sharr_np, final_img[:, :, :3])

            if self.event.is_set():
                break

    def _generate_placeholder_image(self):
        # this is not for joke, it has a real purpose!
        img = np.zeros(
            (CamConsts.SHAPE[1], CamConsts.SHAPE[0], 3), dtype=np.uint8
        )  # cv2 image with 800 x 600 dims, black
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
                (np.random.randint(0, CamConsts.SHAPE[0]), np.random.randint(0, CamConsts.SHAPE[1])),
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
