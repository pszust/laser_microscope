import threading
import tkinter as tk
from tkinter import LEFT, Button, Canvas, Frame, Label, StringVar, X, Y
from typing import TYPE_CHECKING
import cv2
from PIL import Image, ImageTk
from utils.consts import ProjConsts

if TYPE_CHECKING:
    from controls.projector_control import ProjectorControl

import utils.consts as consts


class ProjectorPanel:
    def __init__(self, parent: Frame, controller: "ProjectorControl"):
        self.controller = controller
        self.frame = Frame(parent)
        self.frame.pack(fill=Y)

        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)

        self.lab_title = Label(cur_frame, text="PROJECTOR CONTROL")
        self.lab_title.config(font=consts.subsystem_name_font)
        self.lab_title.pack(side=tk.LEFT)

        cur_frame = Frame(self.frame)
        # proj_frame1.grid(row=1, column=0, padx = padd)
        cur_frame.pack(fill=tk.Y)

        self.init_proj_win_btn = Button(
            cur_frame, text="Init window", command=self.controller.initiate_projector_window
        )
        self.init_proj_win_btn.pack(side=tk.LEFT)

        self.act_proj_win_btn = Button(
            cur_frame, text="Activate window", command=self.controller.activate_projector_window
        )
        self.act_proj_win_btn.pack(side=tk.LEFT)

        self.act_proj_win_btn = Button(cur_frame, text="Close window", command=self.controller.close_projector_window)
        self.act_proj_win_btn.pack(side=tk.LEFT)

        cur_frame = Frame(self.frame)
        # canvas_frame.grid(row=2, column=0, padx = padd)
        cur_frame.pack(fill=tk.Y)

        self.proj_mirror_canvas = Canvas(cur_frame, width=ProjConsts.SMALLER_SHAPE[0], height=ProjConsts.SMALLER_SHAPE[1], bg="black")
        self.proj_mirror_canvas.pack(side=tk.LEFT)

    def update(self):
        if self.controller.is_active:
            # refresh the actual image
            self.controller.refresh_projector_image()

            # refresh the mirror image (copy of actual image with 1/4 res)            
            img = cv2.resize(self.controller.projector_image, ProjConsts.SMALLER_SHAPE[:2], interpolation=cv2.INTER_AREA)
            img = Image.fromarray(img)
            self.proj_imgtk_mirror = ImageTk.PhotoImage(image=img)
            self.proj_mirror_canvas.create_image(0, 0, image=self.proj_imgtk_mirror, anchor=tk.NW)
