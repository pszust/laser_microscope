import logging
import tkinter as tk
from functools import partial
from tkinter import Button, Entry, Frame, Label, OptionMenu, Scrollbar, StringVar
from tkinter.ttk import Combobox

import utils.consts as consts
from controls.image_control import ImageControl

logger = logging.getLogger(__name__)


class BaseTab:
    def __init__(self, parent, control: ImageControl):
        self.control = control
        self.frame = Frame(parent)
        self.frame.pack(fill=tk.Y)
        self.img_mode = None

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="BASE CONTROLS TAB", font=consts.subsystem_name_font).pack(side=tk.LEFT)

        # next row: basic functions
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        # Label(cur_frame, text="Basic functions:").pack(side=tk.LEFT)

        Button(
            cur_frame, text="CON-ALL", command=lambda: self.call_script("connect_all"), bg="#7b9aff"
        ).pack(side=tk.LEFT)
        Button(
            cur_frame, text="CALIB-LOAD", command=lambda: self.call_script("load_calibration"), bg="#7b9aff"
        ).pack(side=tk.LEFT)

        # next row: basic functions
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)

        self.obj_buttons = {}
        for obj in ["5X", "10X", "20X", "50X", "100X"]:
            btn = Button(cur_frame, text=obj, command=partial(self.obj_btn_row, obj), bg="#f2ffd9")
            btn.pack(side=tk.LEFT)
            self.obj_buttons[obj] = btn

    def call_script(self, sname: str):
        self.control.master.automation_controller.execute_script_file(sname)

    def obj_btn_row(self, obj: str):
        self.control.master.camera_panel.set_objective(obj)
        for obj_name, obj_btn in self.obj_buttons.items():
            relief = tk.RAISED
            if obj_name == obj:
                relief = tk.SUNKEN
            obj_btn.config(relief=relief)
