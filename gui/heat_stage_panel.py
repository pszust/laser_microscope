import threading
import tkinter as tk
from tkinter import Button, Entry, Frame, Label, OptionMenu, StringVar

import utils.consts as consts
from utils.utils import serial_ports, thread_execute

if consts.Device.USE_REAL_HEATSTAGE:
    from devices.heat_stage_control import HeatController
else:
    from devices.heat_stage_control_mock import HeatController


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
        self.port_var = StringVar(self.frame)
        self.port_var.set("1")  # default value
        self.port_sel_menu = OptionMenu(cur_frame, self.port_var, *serial_ports(), command=self.update_port)
        self.port_sel_menu.pack(side=tk.LEFT)
        Button(cur_frame, text="Connect to heat stage", command=self.control.connect).pack(side=tk.LEFT)
        self.lbl_status = Label(cur_frame, text="HEAT STAGE status: unknown", bg="gray")
        self.lbl_status.pack(side=tk.LEFT)
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)

        # main info label
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        self.lbl_info = Label(cur_frame, text="T=%2.2f | S=%2.2f" % (0, 0), fg=consts.info_label_color)
        self.lbl_info.config(font=consts.info_label_font)
        self.lbl_info.pack(side=tk.LEFT)

        # Set Temp
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        self.var_set_temp = StringVar(value="25.0")
        Entry(cur_frame, width=7, textvariable=self.var_set_temp).pack(side=tk.LEFT, fill=tk.X)
        btn = Button(
            cur_frame,
            text="Set Temp",
            command=lambda: self.control.set_temperature(float(self.var_set_temp.get())),
        )
        btn.pack(side=tk.LEFT)

        # secondary info label
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        self.lbl_info2 = Label(
            cur_frame,
            text="T-ramp=%2.2f | R-ramp=%2.2f" % (0, 0),
        )
        self.lbl_info2.pack(side=tk.LEFT)

        # Set Rate
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        self.var_set_rate = StringVar(value="3.0")
        Entry(cur_frame, width=7, textvariable=self.var_set_rate).pack(side=tk.LEFT, fill=tk.X)
        btn = Button(
            cur_frame,
            text="Set RATE [°C/min]",
            command=lambda: self.control.set_ramp_rate(float(self.var_set_rate.get())),
        )
        btn.pack(side=tk.LEFT)

        # Set ramp temp
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        self.var_set_tramp = StringVar(value="30.0")
        Entry(cur_frame, width=7, textvariable=self.var_set_tramp).pack(side=tk.LEFT, fill=tk.X)
        btn = Button(
            cur_frame,
            text="Set T-RAMP",
            command=lambda: self.control.set_ramp_temp(float(self.var_set_tramp.get())),
        )
        btn.pack(side=tk.LEFT)

        # ramp button
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        self.ramp_active_btn = Button(
            cur_frame,
            text="Activate T RAMP",
            command=lambda: self.control.activate_ramp(),
        )
        self.ramp_active_btn.pack(side=tk.LEFT)

    def update_tramp_button(self, is_ramp_active: bool):
        if is_ramp_active:
            self.ramp_active_btn.config(relief=tk.SUNKEN, bg="#F0C1A2")
        else:
            self.ramp_active_btn.config(relief=tk.RAISED, bg="#B3E6FA")

    def update_port(self, port):
        self.control.port = int(port)

    def update(self):
        # todo: check out if there is no async (get_status uses method with thread_execute)
        self.control.update_ramp()
        status = self.control.get_status()
        self.update_tramp_button(status.get("tramp_active"))

        # connection label
        con_state = status.get("connection", "UNKNOWN")
        con_color = consts.con_colors.get(con_state, "gray")
        self.lbl_status.config(text=f"HEAT STAGE status: {con_state}", bg=con_color)

        # info label 1
        cur_temp = status.get("current_temp")
        set_temp = status.get("set_temp")
        self.lbl_info.config(text=f"T-ACT={cur_temp:.3f} | SET={set_temp:.3f}")

        ramp_rate = status.get("ramp_rate")
        ramp_temp = status.get("ramp_temp")
        self.lbl_info2.config(text=f"RATE={ramp_rate:.3f} | T-RAMP={ramp_temp:.3f}")
