import logging
import tkinter as tk
from tkinter import Button, Entry, Frame, Label, OptionMenu, Scrollbar, StringVar
from tkinter.ttk import Combobox

import utils.consts as consts
from controls.image_control import ImageControl

logger = logging.getLogger(__name__)


class ImageTab:
    def __init__(self, parent, control: ImageControl):
        self.control = control
        self.frame = Frame(parent)
        self.frame.pack(fill=tk.Y)
        self.img_mode = None

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="IMAGE TAB", font=consts.subsystem_name_font).pack(side=tk.LEFT)

        # next row: img name
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="Image name:").pack(side=tk.LEFT)

        self.var_image_name = StringVar(value="image")
        Entry(cur_frame, width=24, textvariable=self.var_image_name).pack(side=tk.LEFT)

        Button(
            cur_frame,
            text="Save image",
            command=lambda: self.control.save_image(self.var_image_name.get()),
        ).pack(side=tk.LEFT)

        # next row: params
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)

        Label(cur_frame, text="deltaDeg:").pack(side=tk.LEFT)
        self.var_ddeg_mapping = StringVar(value="1")
        self.cmb_ddeg_mapping = Combobox(
            cur_frame,
            width=5,
            state="readonly",
            values=[1, 2, 3, 5, 10, 15],
            textvariable=self.var_ddeg_mapping,
        )
        self.cmb_ddeg_mapping.pack(side=tk.LEFT)

        Label(cur_frame, text="delay [s]:").pack(side=tk.LEFT)
        self.var_delay_mapping = StringVar(value="3")
        self.cmb_delay_mapping = Combobox(
            cur_frame,
            width=5,
            state="readonly",
            values=[1, 2, 3, 5, 10, 15, 20, 30],
            textvariable=self.var_delay_mapping,
        )
        self.cmb_delay_mapping.pack(side=tk.LEFT)

        # next row: buttons
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)

        self.active_rotation_btn = Button(
            cur_frame,
            text="Active rotation",
            command=self.active_rotation_call,
        )
        self.active_rotation_btn.pack(side=tk.LEFT)

        self.active_mapping_btn = Button(
            cur_frame,
            text="Active mapping",
            command=self.active_mapping_call,
        )
        self.active_mapping_btn.pack(side=tk.LEFT)

    def _update_buttons(self):
        cur_mode = self.control.img_mode
        button_map: dict[str, Button] = {
            "ACTIVE_MAPPING": self.active_mapping_btn,
            "ACTIVE_ROTATION": self.active_rotation_btn,
        }
        for mode_name, btn in button_map.items():
            if cur_mode == mode_name:
                btn.config(relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED)

    def active_mapping_call(self):
        deg = float(self.var_ddeg_mapping.get())
        delay = float(self.var_delay_mapping.get())
        self.control.active_mapping(deg, delay)
        self._update_buttons()

    def active_rotation_call(self):
        deg = float(self.var_ddeg_mapping.get())
        delay = float(self.var_delay_mapping.get())
        self.control.active_rotation(deg, delay)
        self._update_buttons()
