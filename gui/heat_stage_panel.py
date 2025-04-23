import threading
import tkinter as tk
from tkinter import Button, Entry, Frame, Label, StringVar

from devices.heat_stage_control import HeatController
import utils.consts as consts
from utils.utils import serial_ports, thread_execute


class HeatPanel:
    def __init__(self, parent: Frame, control: HeatController):
        self.control = control
        self.frame = Frame(parent)
        self.frame.pack(fill=tk.Y)

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="HEAT STAGE CONTROL", font=consts.subsystem_name_font).pack(side=tk.LEFT)

        # Connection controls
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Button(cur_frame, text="Connect to heat stage", command=self.control.connect).pack(side=tk.LEFT)
        self.lbl_status = Label(cur_frame, text="HEAT STAGE status: unknown", bg="gray")
        self.lbl_status.pack(side=tk.LEFT)
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)

        # main info label
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        self.lbl_info = Label(cur_frame, text="T=%2.2f | R=%2.2f | S=%2.2f" % (0, 0, 0), fg=consts.info_label_color)
        self.lbl_info.config(font=consts.info_label_font)
        self.lbl_info.pack(side=tk.LEFT)
        self.lbl_info.pack(side=tk.LEFT)

        # Set Temp
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        self.var_set_temp = StringVar(value="0.0")
        Entry(cur_frame, width=7, textvariable=self.var_set_temp).pack(side=tk.LEFT, fill=tk.X)
        btn = Button(cur_frame, text="Set Temp", command=lambda: self.control.set_temperature(float(self.var_set_temp.get())))
        btn.pack(side=tk.LEFT)

        # Set Rate
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        self.var_set_rate = StringVar(value="0.0")
        Entry(cur_frame, width=7, textvariable=self.var_set_rate).pack(side=tk.LEFT, fill=tk.X)
        btn = Button(cur_frame, text="Set Rate", command=lambda: self.control.set_rate(float(self.var_set_rate.get())))
        btn.pack(side=tk.LEFT)

    def update(self):
        # todo: check out if there is no async (get_status uses method with thread_execute)
        status = self.control.get_status()

        # connection label
        con_state = status.get("connection", "UNKNOWN")
        con_color = consts.con_colors.get(
            con_state, "gray"
        )
        self.lbl_status.config(text=f"HEAT STAGE status: {con_state}", bg=con_color)

        # info label
        cur_temp = status.get("current_temp")
        rate = status.get("rate")
        set_temp = status.get("set_temp")
        self.lbl_info.config(text=f"T={cur_temp} | R={rate} | S={set_temp}")
