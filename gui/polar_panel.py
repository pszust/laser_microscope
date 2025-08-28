import threading
import tkinter as tk
from tkinter import LEFT, Button, Entry, Frame, Label, StringVar, X, Y

import utils.consts as consts
from utils.utils import thread_execute

if consts.Device.USE_REAL_ROTATOR:
    from devices.polar_control import PolarController
else:
    from devices.polar_control_mock import PolarController


class PolarPanel:
    def __init__(self, parent, controller: PolarController, name="TOP POLARIZER CONTROL"):
        self.controller = controller
        self.frame = Frame(parent)
        self.frame.pack(fill=Y)

        # title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text=name, font=consts.subsystem_name_font).pack(side=LEFT)

        # main info label
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        self.lbl_info = Label(cur_frame, text="Rotation=%2.2f" % (0), fg=consts.info_label_color)
        self.lbl_info.config(font=consts.info_label_font)
        self.lbl_info.pack(side=LEFT)

        # Connection controls
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Button(cur_frame, text="Connect to Rotator", command=self.controller.connect).pack(side=tk.LEFT)

        self.lbl_status = Label(cur_frame, text="DEVICE status: unknown", bg="gray")
        self.lbl_status.pack(side=LEFT)

        # rotate relative
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text="Rotate relative: ").pack(side=LEFT)
        self.var_relative = StringVar(value="5")
        Entry(cur_frame, width=9, textvariable=self.var_relative).pack(side=LEFT, fill=X)
        Button(cur_frame, text="Rotate rel", command=self.rotate_relative).pack(side=LEFT)

        # rotate absolute
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text="Rotate absolute: ").pack(side=LEFT)
        self.var_absolute = StringVar(value="15")
        Entry(cur_frame, width=9, textvariable=self.var_absolute).pack(side=LEFT, fill=X)
        Button(cur_frame, text="Rotate rel", command=self.rotate_absolute).pack(side=LEFT)

        cur_frame = tk.Frame(parent)
        cur_frame.pack(fill=tk.Y)

        buttons = [
            "-10.0",
            "-5.00",
            "-1.00",
            "-0.50",
            "-0.10",
            "0.01",
            "0.50",
            "1.00",
            "5.00",
            "10.0",
        ]
        for btn_info in buttons:
            Button(
                cur_frame,
                text=btn_info,
                command=lambda val=float(btn_info): self.rotate_relative(rot=val),
            ).pack(side=tk.LEFT)

    # @thread_execute
    # def connect(self, port):
    #     print(f"Connecting to: {port}")
    #     self.controller.connect(port)

    @thread_execute
    def rotate_relative(self, rot=None):
        if rot is None:
            rot = float(self.var_relative.get())
        self.controller.rotate_rel(rot)

    @thread_execute
    def rotate_absolute(self):
        value = float(self.var_absolute.get())
        self.controller.rotate_abs(value)

    # GUI update hook
    def update(self):
        status = self.controller.get_status()

        # Update connection label
        con_state = status.get("connection", "UNKNOWN")
        con_color = {"CONNECTED": "lime", "CONNECTING": "yellow", "NOT CONNECTED": "gray"}.get(
            con_state, "gray"
        )
        self.lbl_status.config(text=f"DEVICE status: {con_state}", bg=con_color)

        # update info label
        self.lbl_info.config(text="Rotation=%2.2f" % (status["rotation"]))
