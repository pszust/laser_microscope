import tkinter as tk
from tkinter import Canvas
from typing import TYPE_CHECKING
import numpy as np
import cv2
from PIL import Image, ImageTk
from utils.consts import ProjConsts

if TYPE_CHECKING:
    from gui.main_window import MainWindow  # only used for type hints

class ProjectorControl:
    def __init__(self, master: "MainWindow"):
        self.master = master
        self.projector_window = None
        self.projector_image = np.zeros(ProjConsts.PROJ_IMG_SHAPE, dtype=np.uint8)
        self.is_active = False

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

    def set_image(self, image: np.array):
        if image.shape == ProjConsts.PROJ_IMG_SHAPE:
            err_msg = f"Invalid shape of the projector image, should be {ProjConsts.PROJ_IMG_SHAPE}, is {image.shape}"
            print(err_msg)  # todo: logging
        else:
            self.projector_image = image

    def load_and_set_image(self, path: str) -> None:
        image = cv2.imread(path)
        self.projector_image = image

    def refresh_projector_image(self):
        # refresh the actual screen
        img = Image.fromarray(self.projector_image)
        self.proj_imgtk = ImageTk.PhotoImage(image=img)
        # self.canvas_proj.create_image(512, 384, image=self.proj_imgtk, anchor=tk.CENTER)
        self.canvas_proj.create_image(0, 0, image=self.proj_imgtk, anchor=tk.NW)