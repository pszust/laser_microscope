# controls/chiral_control.py

import logging
import os
import time

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


class ChiralControl:
    """
    All business logic for the Chiral module.
    GUI calls into these methods and reads lightweight state (e.g., pattern_image).
    """

    def __init__(self, master: "MainWindow"):
        self.master = master
        self.pattern_array: np.ndarray | None = None
        self.pattern_image: Optional[Image.Image] = None
        self.pattern_path: Optional[str] = None
        self._last_open_dir: Optional[str] = None
        self.melting_script_path = ""
        self.melting_vars_txt = ""

    # -------------------------------------------------------------------------
    # Pattern loading
    # -------------------------------------------------------------------------
    def load_pattern_image(self, path: str) -> None:
        """
        Loads image, convers to pattern_array and also creates pattern_image
        """
        try:
            arr = self.png_to_binary_array(path, threshold=127)
            self.pattern_array = arr
            self.pattern_image = Image.fromarray(arr * 255, mode="L")
        except Exception as e:
            logger.error(f"Failed to load image: {e}")

    # Optional convenience getters (handy if used elsewhere)
    def has_pattern(self) -> bool:
        return self.pattern_image is not None

    def get_pattern_size(self) -> Optional[tuple[int, int]]:
        return self.pattern_image.size if self.pattern_image else None

    def start_melting(self, x, y, pa, pb, pc):
        # update automation variables with on-start variables
        on_start_variables = {
            "x_start": x,
            "y_start": y,
            "the_pattern": self.pattern_array,
            "work_size": 400,
        }
        msg = "Loading on-start variables for melting:"
        self.master.automation_controller.update_variables(on_start_variables, optional_msg=msg)
        self.master.automation_controller.execute_script_file(self.melting_script_path)

    @staticmethod
    def png_to_binary_array(path: str, threshold: int = 127) -> np.ndarray:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        binary = (arr > threshold).astype(np.uint8)
        return binary
