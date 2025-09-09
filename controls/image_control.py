# controls/chiral_control.py

import logging
import os
import re
import time
from collections import defaultdict

# Only the file dialog lives here from tkinter; we do not show message boxes in control
from tkinter import filedialog
from typing import TYPE_CHECKING, Optional

import cv2
import imutils
import numpy as np
from PIL import Image, ImageChops

from utils.consts import CamConsts

if TYPE_CHECKING:
    from gui.main_window import MainWindow

logger = logging.getLogger(__name__)
IMG_SAVE_PATH = "saved_images/"
ACTIVE_MAP_SCR = "active_mapping.scrpt"
ACTIVE_ROT_SCR = "active_rotation.scrpt"


class ImageControl:
    def __init__(self, master: "MainWindow"):
        self.master = master
        self.img_mode = None

    def active_rotation(self, deg: float, delay: float) -> bool:
        self.master.automation_controller.cancel_execution(info_only=True)
        if self.img_mode == "ACTIVE_ROTATION":
            self.img_mode = None
        else:
            self.img_mode = "ACTIVE_ROTATION"
            self.master.automation_controller.execute_script_file(
                ACTIVE_ROT_SCR, args=[90 - deg, 90 + deg, delay]
            )

    def active_mapping(self, deg: float, delay: float) -> str:
        self.master.automation_controller.cancel_execution(info_only=True)
        if self.img_mode == "ACTIVE_MAPPING":
            self.img_mode = None
            time.sleep(0.4)  # there needs to be a delay before we pass next command to automation
            self.master.automation_controller.pass_command("reset_alt_image()")
        else:
            self.img_mode = "ACTIVE_MAPPING"
            self.master.automation_controller.execute_script_file(
                ACTIVE_MAP_SCR, args=[90 - deg, 90 + deg, delay]
            )

    def save_image(self, name: str):
        fpath = self._format_name(name)
        self.master.camera_panel.save_image(fpath)

    @staticmethod
    def get_saved_imgs_names_and_nums() -> dict[str : list[int]]:
        pattern = re.compile(r"^(.+)_(\d{3,8}).png$")
        result = defaultdict(list)
        img_files = [f for f in os.listdir(IMG_SAVE_PATH) if f.endswith(".png")]
        for f in img_files:
            m = pattern.match(f)
            if m:
                name, num = m.groups()
                result[name].append(int(num))
        return result

    def _format_name(self, name: str) -> str:
        name = name.split(".")[0]  # dots are not allowed
        name_nums = self.get_saved_imgs_names_and_nums()
        max_num = max(name_nums.get(name, [0]))
        formatted_name = f"{name}_{str(max_num+1).zfill(5)}.png"
        return os.path.join(IMG_SAVE_PATH, formatted_name)
