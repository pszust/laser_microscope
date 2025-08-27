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
        self.pattern_image: Optional[Image.Image] = None
        self.pattern_path: Optional[str] = None
        self._last_open_dir: Optional[str] = None
        self.melting_script_path = "custom_scripts/base_melt.scrpt"
        logger.debug("ChiralControl initialization done.")

    # -------------------------------------------------------------------------
    # Pattern loading
    # -------------------------------------------------------------------------
    def load_pattern_image(self, path: Optional[str] = None) -> None:
        """
        Load a pattern PNG into self.pattern_image (PIL.Image).
        If `path` is None, opens a file picker. Raises on error.

        This method is GUI-agnostic except for the file dialog — it does not
        show message boxes; errors are propagated so the caller can decide how
        to surface them.
        """
        if path is None:
            # Try to anchor the dialog in the last used directory
            initialdir = self._last_open_dir or os.getcwd()

            # If the MainWindow/root exists, pass it as parent for modality
            parent_widget = getattr(self.master, "root", None)

            path = filedialog.askopenfilename(
                title="Select pattern PNG",
                filetypes=[("PNG images", "*.png"), ("All files", "*.*")],
                initialdir=initialdir,
                parent=parent_widget if parent_widget else None,
            )

            if not path:
                logger.info("Pattern load canceled by user.")
                return  # treat cancel as a no-op

        # Normalize and remember directory
        path = os.path.abspath(path)
        self._last_open_dir = os.path.dirname(path)

        if not os.path.isfile(path):
            raise FileNotFoundError(f"File does not exist: {path}")

        # Basic extension check (not strictly necessary, PIL will validate anyway)
        _, ext = os.path.splitext(path)
        if ext.lower() not in {".png"}:
            logger.warning(f"Selected file is not .png: {path}")

        # Load with PIL (keeps small images exactly as-is)
        try:
            with Image.open(path) as im:
                # For pixel-accurate previews later, we avoid any resampling here.
                # Preserve mode if it's L/RGB/RGBA; otherwise coerce to RGB.
                if im.mode not in ("L", "RGB", "RGBA"):
                    im = im.convert("RGB")
                self.pattern_image = im.copy()
        except Exception as e:
            logger.exception("Failed to open image.")
            raise RuntimeError(f"Failed to open image: {e}") from e

        self.pattern_path = path

        # Validate size (not mandatory, but helpful)
        w, h = self.pattern_image.size
        if w <= 0 or h <= 0:
            raise ValueError("Loaded image has invalid dimensions (0 size).")

        # Informative warnings for expected operating range (2..~80 px)
        if (w < 2 or h < 2) or (w > 200 or h > 200):
            # Keep it flexible; GUI will downscale if needed,
            # but log a heads-up for unexpected sizes.
            logger.warning(
                "Pattern size is unusual for pixel-accurate display: %dx%d. "
                "Expected roughly 2..80 px per side.",
                w,
                h,
            )

        logger.info(
            "Pattern loaded: %s (%dx%d, mode=%s)", os.path.basename(path), w, h, self.pattern_image.mode
        )

    # Optional convenience getters (handy if used elsewhere)
    def has_pattern(self) -> bool:
        return self.pattern_image is not None

    def get_pattern_size(self) -> Optional[tuple[int, int]]:
        return self.pattern_image.size if self.pattern_image else None

    def start_melting(self, x, y, pa, pb, pc):
        loop_x, loop_y = self.pattern_image.size
        args = [str(loop_x), str(loop_y)]
        self.master.automation_controller.execute_script_file(self.melting_script_path, args)
