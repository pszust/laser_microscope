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

import utils.consts as cst
from utils.consts import CamConsts

if TYPE_CHECKING:
    from gui.main_window import MainWindow

logger = logging.getLogger(__name__)


class BasepanelControl:
    def __init__(self, master: "MainWindow"):
        self.master = master

    # TODO: move somewhere to utils
    @staticmethod
    def get_objective_mag(obj: str):
        mag = int(obj.replace("X", ""))
        return mag

    def scale_mm2px(self, dist_mm: float) -> int:
        obj = self.master.camera_panel.get_objective()
        px_in_mm = cst.SCALE_PX2MM_AT_1X * self.get_objective_mag(obj)
        return int(px_in_mm * dist_mm)

    def scale_px2mm(self, dist_px: int) -> float:
        obj = self.master.camera_panel.get_objective()
        px_in_mm = cst.SCALE_PX2MM_AT_1X * self.get_objective_mag(obj)
        return dist_px / px_in_mm
