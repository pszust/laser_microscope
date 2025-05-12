import threading
import tkinter as tk
from tkinter import Button, Entry, Frame, Label, StringVar

import utils.consts as consts
from devices.labjack_controller_mock import LabjackController
from utils.utils import serial_ports, thread_execute


class LabjackPanel:
    def __init__(self, parent: Frame, control: LabjackController):
        self.control = control
        self.frame = Frame(parent)
        self.frame.pack(fill=tk.Y)

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="LABJACK CONTROL", font=consts.subsystem_name_font).pack(side=tk.LEFT)

        # Connection controls
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Button(cur_frame, text="Connect to Labjack", command=self.control.connect).pack(side=tk.LEFT)
        self.lbl_status = Label(cur_frame, text="LABJACK status: unknown", bg="gray")
        self.lbl_status.pack(side=tk.LEFT)
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)

        # main info label
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        self.lbl_info = Label(cur_frame, text="Z=%2.2f" % 0, fg=consts.info_label_color)
        self.lbl_info.config(font=consts.info_label_font)
        self.lbl_info.pack(side=tk.LEFT)
        self.lbl_info.pack(side=tk.LEFT)

        # Move controls absolute
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="Move absolute: ").pack(side=tk.LEFT)
        self.var_height = StringVar(value="0.0")
        Entry(cur_frame, width=7, textvariable=self.var_height).pack(side=tk.LEFT, fill=tk.X)
        btn = Button(
            cur_frame, text="Move", command=lambda: self.control.set_height(float(self.var_height.get()))
        )
        btn.pack(side=tk.LEFT)

        btn = Button(cur_frame, text="Home", command=lambda: self.control.home())
        btn.pack(side=tk.LEFT)

        # Move relative
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="Move relative: ").pack(side=tk.LEFT)

        cur_frame = tk.Frame(parent)
        cur_frame.pack(fill=tk.Y)

        buttons = [
            "-5.0",
            "-1.00",
            "-0.25",
            "-0.05",
            "-0.01",
            "0.01",
            "0.05",
            "0.25",
            "1.00",
            "5.00",
        ]
        for btn_info in buttons:
            Button(
                cur_frame,
                text=btn_info,
                command=lambda val=float(btn_info): self.control.move_relative(val),
            ).pack(side=tk.LEFT)

    def update(self):
        # todo: check out if there is no async (get_status uses method with thread_execute)
        status = self.control.get_status()

        # connection label
        con_state = status.get("connection", "UNKNOWN")
        con_color = consts.con_colors.get(con_state, "gray")
        self.lbl_status.config(text=f"LABJACK status: {con_state}", bg=con_color)

        # height label
        height = status.get("height")
        self.lbl_info.config(text=f"Z = {height:.3f} mm")
