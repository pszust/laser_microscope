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
import logging

logger = logging.getLogger(__name__)


class CameraPanel:
    def __init__(self, parent: Frame, controller: CameraController):
        self.controller = controller
        self.frame = Frame(parent)
        self.camera_image = Image.new("RGB", (800, 600), color="grey")

        self.canvas = Canvas(parent, width=800, height=600, bg="black")
        self.canvas.grid(row=0, column=0, sticky=tk.W + tk.E)

        logger.debug(f"Initialization done.")

    def display_image(self):
        # Display the selected image in the canvas
        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.camera_image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    @thread_execute
    def update_image(self):
        self.camera_image = self.controller.get_image()
        self.display_image()
