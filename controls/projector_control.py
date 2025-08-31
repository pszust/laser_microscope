import logging
import tkinter as tk
from tkinter import Canvas
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageTk

from utils.consts import CamConsts, ProjConsts
from utils.utils import get_homography_matrix, num_to_coords

if TYPE_CHECKING:
    from gui.main_window import MainWindow  # only used for type hints
logger = logging.getLogger(__name__)

calib_dots_dim = 4


class ProjectorControl:
    def __init__(self, master: "MainWindow"):
        self.master = master
        self.projector_window = None
        # self.projector_image = np.zeros(ProjConsts.PROJ_IMG_SHAPE, dtype=np.uint8)
        self.is_active = False
        self.homomatrix = None  # TODO: this needs to be taken from calibration

        # raw_ images are before transposing to proj_shape and have cam_shape dimensions (REAL_WIDTH etc)
        raw_img_shape = (CamConsts.SHAPE[1], CamConsts.SHAPE[0], CamConsts.SHAPE[2])
        self.raw_animation_img = None
        self.raw_loaded_img = None
        self.raw_drawn_image = np.zeros(raw_img_shape, np.uint8)

        # direct_ images that are in project dimensions and are not being transposed (they are displayed as-is)
        self.direct_calib_img = None

    def initiate_projector_window(self):
        if self.projector_window == None:
            # self.projector_window = ProjectorWindow(root)
            # self.app = ProjectorWindow(self.projector_window)
            self.projector_window = tk.Toplevel(self.master)
            self.projector_window.title("Projector window - move to projector screen")
            self.projector_window.geometry("400x400")
            # self.master.log("Opened projector window")

    def close_projector_window(self):
        if self.projector_window != None:
            self.projector_window.destroy()
            self.projector_window = None
            # self.master.log("Closed projector window")
            self.is_active = False

    def activate_projector_window(self):
        # initialize full screen mode
        self.projector_window.overrideredirect(True)
        self.projector_window.state("zoomed")
        # self.projector_window.activate()

        self.canvas_proj = Canvas(
            self.projector_window,
            width=ProjConsts.PROJ_IMG_SHAPE[0],
            height=ProjConsts.PROJ_IMG_SHAPE[1],
            bg="black",
            highlightthickness=0,
            relief="ridge",
        )
        self.canvas_proj.pack(side=tk.LEFT)
        self.is_active = True
        # self.master.log("Projector window activated")

    # def set_image(self, image: np.array):
    #     if image.shape == ProjConsts.PROJ_IMG_SHAPE:
    #         err_msg = f"Invalid shape of the projector image, should be {ProjConsts.PROJ_IMG_SHAPE}, is {image.shape}"
    #         print(err_msg)  # todo: logging
    #     else:
    #         self.projector_image = image

    def load_and_set_image(self, path: str) -> None:
        self.raw_loaded_img = cv2.imread(path)

    def unload_img(self) -> None:
        self.raw_loaded_img = None

    def update_animation_image(self, img: np.ndarray | None) -> None:
        self.raw_animation_img = img

    def set_calibration_img(self, num: int):
        logger.info(f"Calib num {num} and it is {type(num)}")
        if num == -1:
            self.direct_calib_img = None
        else:
            # draw dot on black canvas at specified postion
            proj_x, proj_y = num_to_coords(num, size=calib_dots_dim)
            self.direct_calib_img = np.zeros(
                (ProjConsts.PROJ_IMG_SHAPE[1], ProjConsts.PROJ_IMG_SHAPE[0], 3), np.uint8
            )
            self.direct_calib_img = cv2.circle(
                self.direct_calib_img, (proj_x, proj_y), 18, (255, 255, 255), -1
            )

    def refresh_projector_image(self) -> np.ndarray | None:
        # refresh the actual screen
        final_image = self.compose_projector_image()
        if final_image is None:
            final_image = np.zeros(ProjConsts.PROJ_IMG_SHAPE, dtype=np.uint8)

        img = Image.fromarray(final_image)
        self.proj_imgtk = ImageTk.PhotoImage(image=img)
        # self.canvas_proj.create_image(512, 384, image=self.proj_imgtk, anchor=tk.CENTER)
        self.canvas_proj.create_image(0, 0, image=self.proj_imgtk, anchor=tk.NW)
        return final_image  # this is to display the image in GUI

    @staticmethod
    def merge_images(images: list[np.ndarray]) -> np.ndarray | None:
        if not images:
            return None

        result = images[0].copy()
        if len(images) > 1:
            for img in images[1:]:
                result = cv2.add(result, img)
        return result

    def get_calibration_matrix(self):
        self.homomatrix = get_homography_matrix()

    def compose_projector_image(self) -> np.ndarray | None:
        transposed_image = None
        direct_image = None

        # first add together all the images that need to be composed nad transposed by homomatrix (in cam shape)
        if images_to_merge := [
            img
            for img in (self.raw_animation_img, self.raw_loaded_img, self.raw_drawn_image)
            if img is not None
        ]:
            if self.raw_animation_img is not None:
                aaa = 1
            merged = self.merge_images(images_to_merge)

            # transpose using the thing
            if self.homomatrix is not None:
                transposed_image = cv2.warpPerspective(
                    merged, self.homomatrix, ProjConsts.PROJ_IMG_SHAPE[:2]
                )
            else:
                transposed_image = cv2.resize(merged, ProjConsts.PROJ_IMG_SHAPE[:2])

        # then add together all the images that are to be displayed as-is (without transposing)
        if direct_images_to_merge := [img for img in (self.direct_calib_img,) if img is not None]:
            direct_image = self.merge_images(direct_images_to_merge)

        final_img = self.merge_images([img for img in (transposed_image, direct_image) if img is not None])
        return final_img

    # Helper methods
    def draw_using_brush(self, coords, radius, color):
        """
        called by camera panel when drawing, uses black/white color
        """
        self.raw_drawn_image = cv2.circle(self.raw_drawn_image, coords, radius, color, -1)
