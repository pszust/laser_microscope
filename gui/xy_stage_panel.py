import threading
import tkinter as tk
from tkinter import LEFT, Button, Entry, Frame, Label, StringVar, X, Y

import utils.consts as consts
from utils.utils import serial_ports, thread_execute

if consts.Device.USE_REAL_M30:
    from devices.m30_control import StageController
else:
    from devices.m30_control_mock import StageController


class M30Panel:
    def __init__(self, parent: Frame, controller: StageController):
        self.controller = controller
        self.frame = Frame(parent)
        self.frame.pack(fill=Y)
        self.var_move_rel_dist = StringVar(value="1.0")

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text="XY-STAGE CONTROL", font=consts.subsystem_name_font).pack(side=LEFT)

        # Connection controls
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Button(cur_frame, text="Connect to XY-stage", command=self.controller.connect).pack(side=LEFT)
        self.lbl_status = Label(cur_frame, text="STAGE status: unknown", bg="gray")
        self.lbl_status.pack(side=LEFT)
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)

        # main info label
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        self.lbl_info = Label(
            cur_frame, text="X=%2.2f, Y=%2.2f, STATE=%s" % (0, 0, "NONE"), fg=consts.info_label_color
        )
        self.lbl_info.config(font=consts.info_label_font)
        self.lbl_info.pack(side=LEFT)

        # Move controls (X absolute)
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text="X-move: ").pack(side=LEFT)
        self.var_x_move = StringVar(value="0.0")
        Entry(cur_frame, width=7, textvariable=self.var_x_move).pack(side=LEFT, fill=X)
        Button(cur_frame, text="Move X", command=self.move_xy_rel(1, 1)).pack(side=LEFT)

        # Move controls (Y absolute)
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text="Y-move: ").pack(side=LEFT)
        self.var_y_move = StringVar(value="0.0")
        Entry(cur_frame, width=7, textvariable=self.var_y_move).pack(side=LEFT, fill=X)
        Button(cur_frame, text="Move Y", command=self.move_xy_rel(1, 1)).pack(side=LEFT)  # TODO: fix

        # Move controls (XY rel)
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Button(cur_frame, text="\\", command=lambda: self.move_xy_rel(-1, -1)).pack(side=LEFT)
        Button(cur_frame, text="T", command=lambda: self.move_xy_rel(0, -1)).pack(side=LEFT)
        Button(cur_frame, text="/", command=lambda: self.move_xy_rel(1, -1)).pack(side=LEFT)
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Button(cur_frame, text="L", command=lambda: self.move_xy_rel(-1, 0)).pack(side=LEFT)
        Button(cur_frame, text=".", command=lambda: self.move_xy_rel(0, 0)).pack(side=LEFT)
        Button(cur_frame, text="R", command=lambda: self.move_xy_rel(1, 0)).pack(side=LEFT)
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Button(cur_frame, text="/", command=lambda: self.move_xy_rel(-1, 1)).pack(side=LEFT)
        Button(cur_frame, text="D", command=lambda: self.move_xy_rel(0, 1)).pack(side=LEFT)
        Button(cur_frame, text="\\", command=lambda: self.move_xy_rel(1, 1)).pack(side=LEFT)

        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text="Distance: ").pack(side=LEFT)
        Entry(cur_frame, width=7, textvariable=self.var_move_rel_dist).pack(side=LEFT, fill=X)

    # @thread_execute
    # def move_x_abs(self):
    #     status = self.controller.get_status()
    #     cur_x, cur_y = status["x_pos"], status["y_pos"]
    #     x_value = float(self.var_x_move.get())
    #     self.controller.move_absolute_xy(x_value, cur_y)

    # @thread_execute
    # def move_y_abs(self):
    #     status = self.controller.get_status()
    #     cur_x, cur_y = status["x_pos"], status["y_pos"]
    #     y_value = float(self.var_y_move.get())
    #     self.controller.move_absolute_xy(cur_x, y_value)

    @thread_execute
    def move_xy_rel(self, x_multip, y_multip):
        status = self.controller.get_status()
        cur_x, cur_y = status["x_pos"], status["y_pos"]
        dst = float(self.var_move_rel_dist.get())
        self.controller.set_postion(cur_x + dst * x_multip, cur_y + dst * y_multip)

    # GUI update hook
    def update(self):
        status = self.controller.get_status()

        # connection label
        con_state = status.get("connection", "UNKNOWN")
        con_color = {"CONNECTED": "lime", "CONNECTING": "yellow", "NOT CONNECTED": "gray"}.get(
            con_state, "gray"
        )
        self.lbl_status.config(text=f"STAGE status: {con_state}", bg=con_color)

        # position label
        state = status.get("state", "err")
        state_map = {
            "IDLE": "I",
            "MOVING": "M",
        }
        self.lbl_info.config(
            text="X=%2.2f, Y=%2.2f, STATE=%s"
            % (status["x_pos"], status["y_pos"], state_map.get(state, "E"))
        )
