import logging
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
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageTk

import utils.consts as consts
from devices.camera_control_mock import CameraController
from utils.consts import CamConsts
from utils.utils import thread_execute

if TYPE_CHECKING:
    from gui.main_window import MainWindow

logger = logging.getLogger(__name__)


def costly_overlay(img: Image, overlay: np.array, R: int, G: int, B: int, s=128):
    overlay_gray = Image.fromarray(overlay).convert("L")  # ensure its in grayscale mode

    # Convert grayscale to color-tinted image
    blue_overlay = Image.new("RGBA", overlay_gray.size)
    blue_overlay_data = [(R, G, B, s) for value in overlay_gray.getdata()]  # semi-transparent
    blue_overlay.putdata(blue_overlay_data)

    base_rgba = img.convert("RGBA")
    combined = Image.alpha_composite(base_rgba, blue_overlay)
    return combined.convert("RGB")


def display2real(display_cords: tuple[int, int]) -> tuple[int, int]:
    real_x = display_cords[0] * CamConsts.REAL_WIDTH / CamConsts.DISPLAY_WIDTH
    real_y = display_cords[1] * CamConsts.REAL_HEIGHT / CamConsts.DISPLAY_HEIGHT
    return (int(real_x), int(real_y))


def real2display(real_cords: tuple[int, int]) -> tuple[int, int]:
    disp_x = real_cords[0] * CamConsts.DISPLAY_WIDTH / CamConsts.REAL_WIDTH
    disp_y = real_cords[1] * CamConsts.DISPLAY_HEIGHT / CamConsts.REAL_HEIGHT
    return (int(disp_x), int(disp_y))


class CameraPanel:
    def __init__(self, parent: Frame, controller: CameraController, master: "MainWindow"):
        self.master = master
        self.controller = controller
        self.alt_image = None  # if present it is displayed as image instead of camera image
        self.frame = Frame(parent)
        self.full_size_img = Image.new("RGB", (CamConsts.REAL_WIDTH, CamConsts.REAL_HEIGHT), color="grey")
        self.disp_cam_img = Image.new(
            "RGB", (CamConsts.DISPLAY_WIDTH, CamConsts.DISPLAY_HEIGHT), color="grey"
        )
        self.last_b1_press_pos = (0, 0)  # this is in display coordinates
        self.brush_index = 4
        self.brush_size = CamConsts.BRUSH_SIZR_ARR[self.brush_index]  # this will be in real coords

        self.canvas = Canvas(
            parent, width=CamConsts.DISPLAY_WIDTH, height=CamConsts.DISPLAY_HEIGHT, bg="black"
        )
        self.canvas.grid(row=0, column=0, sticky=tk.W + tk.E)

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.grid(row=1, column=0, sticky=tk.W + tk.E)
        Label(cur_frame, text="CAMERA CONTROL", font=consts.subsystem_name_font).pack(side=tk.LEFT)

        # Connection controls
        cur_frame = Frame(parent)
        cur_frame.grid(row=2, column=0, sticky=tk.W + tk.E)
        Button(cur_frame, text="Connect to camera", command=self.controller.connect).pack(side=tk.LEFT)
        self.lbl_status = Label(cur_frame, text="CAMERA status: unknown", bg="gray")
        self.lbl_status.pack(side=tk.LEFT)

        cur_frame = Frame(parent)
        cur_frame.grid(row=3, column=0, sticky=tk.W + tk.E)
        self.lbl_test = Label(cur_frame, text="Init")
        self.lbl_test.pack(fill=tk.Y)

        cur_frame = Frame(parent)
        cur_frame.grid(row=4, column=0, sticky=tk.W + tk.E)
        self.setup_interaction_mode_selector(cur_frame)
        self.setup_bindings()

        logger.debug(f"Initialization done.")

    def setup_bindings(self):
        self.canvas.bind("<ButtonPress-1>", self.canvas_button1_press)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_button1_release)
        # self.canvas.bind("<ButtonRelease-1>", self.cam_btn_release)
        self.canvas.bind("<B1-Motion>", self.canvas_button1_motion)
        self.canvas.bind("<Motion>", self.canvas_motion)

    def setup_interaction_mode_selector(self, frame):
        self.interaction_var = tk.StringVar(value="NONE")
        modes = ["NONE", "ANMT", "DRAW"]
        Label(frame, text="Interaction mode:").pack(side=tk.LEFT)
        for mode in modes:
            tk.Radiobutton(frame, text=mode, variable=self.interaction_var, value=mode).pack(side=tk.LEFT)

    # def set_mode(self, mode):
    #     self.interaction_mode = mode
    #     logger.debug(f"Interaction mode {self.interaction_mode}")

    def display_image(self):
        # Display the selected image in the canvas
        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.disp_cam_img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # self.mouseWinX, self.mouseWinY = 0, 0
        # self.bind("<Motion>", self.press_in_window)

    # @thread_execute
    def update_image(self):
        if self.alt_image:
            self.full_size_img = self.alt_image
        else:
            self.full_size_img = self.controller.get_image()
        self.disp_cam_img = self.full_size_img.resize(CamConsts.DISPLAY_SHAPE)
        self.display_image()

    def update(self):
        # todo: check out if there is no async (get_status uses method with thread_execute)
        status = self.controller.get_status()

        # connection label
        con_state = status.get("connection", "UNKNOWN")
        con_color = consts.con_colors.get(con_state, "gray")
        self.lbl_status.config(text=f"CAMERA status: {con_state}", bg=con_color)

    def change_brush_size(self, direction):
        self.brush_index += direction
        if self.brush_index > len(CamConsts.BRUSH_SIZR_ARR):
            self.brush_index = len(CamConsts.BRUSH_SIZR_ARR)
        if self.brush_index < 0:
            self.brush_index = 0
        self.brush_size = CamConsts.BRUSH_SIZR_ARR[self.brush_index]
        logger.info(f"Brush sie is now {self.brush_size}")

    def canvas_button1_press(self, event):
        x = event.x
        y = event.y
        self.last_b1_press_pos = (x, y)
        logger.info(f"Button press at X: {x}; Y:{y}")

    def canvas_button1_motion(self, event):
        """This activates during B1 press-move"""
        x = event.x
        y = event.y

    def canvas_motion(self, event):
        """
        (only when the button is not pressed)
        """
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.lbl_test.config(text=f"X={self.mouse_x}, Y={self.mouse_y}")

    def canvas_button1_release(self, event):
        logger.info(f"Button release at X: {event.x}; Y:{event.y}")
        interaction_mode = self.interaction_var.get()
        if interaction_mode == "ANMT":
            # get target
            dx = event.x - self.last_b1_press_pos[0]
            dy = -(event.y - self.last_b1_press_pos[1])
            dx = 0.00001 if dx == 0 else dx
            angle = int(np.arctan(dy / dx) * 180 / np.pi)  # is this correct?
            if angle < 0:
                angle += 180
            if dy < 0:
                angle += 180

            # call animation tab with given values
            anim_start_real = display2real(self.last_b1_press_pos)
            self.master.animation_control.start_animation_gui_params(
                anim_start_real[1], anim_start_real[0], angle, self.brush_size
            )

    def display_alt_image(self, image, timeout=0):
        if type(image) is np.ndarray:
            image = Image.fromarray(image)
        self.alt_image = image

    def reset_alt_image(self):
        self.alt_image = None
