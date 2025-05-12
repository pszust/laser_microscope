import threading
import tkinter as tk
from tkinter import (
    LEFT,
    Button,
    Canvas,
    Entry,
    Frame,
    Label,
    StringVar,
    X,
    Y,
)

from PIL import Image, ImageTk

import utils.consts as consts
from devices.camera_control_mock import CameraController
from utils.utils import thread_execute
import numpy as np
from utils.consts import CamConsts
from typing import TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from gui.main_window import MainWindow

logger = logging.getLogger(__name__)


def costly_overlay(img: Image, overlay: np.array, R: int, G: int, B: int, s=128):
    overlay_gray = Image.fromarray(overlay).convert("L")  # ensure it's in grayscale mode

    # Convert grayscale to blue-tinted image: (R=0, G=0, B=value, A=alpha)
    blue_overlay = Image.new("RGBA", overlay_gray.size)
    blue_overlay_data = [(R, G, B, s) for value in overlay_gray.getdata()]  # semi-transparent
    blue_overlay.putdata(blue_overlay_data)

    base_rgba = img.convert("RGBA")
    combined = Image.alpha_composite(base_rgba, blue_overlay)
    return combined.convert("RGB")


class CameraPanel:
    def __init__(self, parent: Frame, controller: CameraController, master: "MainWindow"):
        self.master = master
        self.controller = controller
        self.frame = Frame(parent)
        self.camera_image = Image.new(
            "RGB", (CamConsts.DISPLAY_WIDTH, CamConsts.DISPLAY_HEIGHT), color="grey"
        )
        self.interaction_mode = None

        self.canvas = Canvas(
            parent, width=CamConsts.DISPLAY_WIDTH, height=CamConsts.DISPLAY_HEIGHT, bg="black"
        )
        self.canvas.grid(row=0, column=0, sticky=tk.W + tk.E)

        cur_frame = Frame(parent)
        cur_frame.grid(row=1, column=0, sticky=tk.W + tk.E)
        self.lbl_test = Label(cur_frame, text="Init")
        self.lbl_test.pack(fill=tk.Y)
        self.last_b1_press_pos = (0, 0)
        logger.debug(f"Initialization done.")

    def display_image(self):
        # Display the selected image in the canvas
        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.camera_image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.bind("<ButtonPress-1>", self.canvas_button1_press)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_button1_release)
        # self.canvas.bind("<ButtonRelease-1>", self.cam_btn_release)
        self.canvas.bind("<B1-Motion>", self.canvas_button1_motion)
        self.canvas.bind("<Motion>", self.canvas_motion)

        # self.mouseWinX, self.mouseWinY = 0, 0
        # self.bind("<Motion>", self.press_in_window)

    @thread_execute
    def update_image(self):
        self.camera_image = self.controller.get_image()
        self.display_image()

    def canvas_button1_press(self, event):
        x = event.x
        y = event.y
        self.last_b1_press_pos = (x, y)
        logger.info(f"X: {x}; Y:{y}")

    def canvas_button1_motion(self, event):
        """This activates during B1 press-move"""
        x = event.x
        y = event.y
        logger.info(f"B1-Motion X: {x}; Y:{y}")
        # y = event.y - 76 + 50  # to account for bigger canvas than camera image

    def canvas_motion(self, event):
        """
        (only when the button is not pressed)
        """
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.lbl_test.config(text=f"X={self.mouse_x}, Y={self.mouse_y}")

    def canvas_button1_release(self, event):
        if self.interaction_mode == "ANIMATION":
            # get target
            # call animation control with given target
            pass
