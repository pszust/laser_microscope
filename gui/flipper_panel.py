import threading
import tkinter as tk
from tkinter import Button, Entry, Frame, Label, Scrollbar, StringVar, Text

from devices.flipper_controller import FlipperController
import utils.consts as consts
from core.automation import Automation
from utils.command_handler import Command, parse_command
from utils.utils import thread_execute


class FlipperPanel:
    def __init__(self, parent, controllers: list[FlipperController, FlipperController]):
        self.control1 = controllers[0]
        self.control2 = controllers[1]
        self.frame = Frame(parent)
        self.frame.pack(fill=tk.Y)

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="FLIPPERS CONTROL", font=consts.subsystem_name_font).pack(side=tk.LEFT)

        # Connection controls
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Button(cur_frame, text="Connect to Flipper 1", command=self.control1.connect).pack(side=tk.LEFT)
        self.lbl_status1 = Label(cur_frame, text="FLIPPER 1 status: unknown", bg="gray")
        self.lbl_status1.pack(side=tk.LEFT)

        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Button(cur_frame, text="Connect to Flipper 2", command=self.control2.connect).pack(side=tk.LEFT)
        self.lbl_status2 = Label(cur_frame, text="FLIPPER 2 status: unknown", bg="gray")
        self.lbl_status2.pack(side=tk.LEFT)

        # main controls
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        l = Label(cur_frame, text="FLIPPER 1", fg=consts.info_label_color)
        l.config(font=consts.info_label_font)
        l.pack(side=tk.LEFT)
        Button(cur_frame, text="IN", command=self.flipper1_in).pack(side=tk.LEFT)
        Button(cur_frame, text="OUT", command=self.control1.flipper_out).pack(side=tk.LEFT)
        self.lbl_info1 = Label(cur_frame, text="State: UNKNOWN", fg=consts.info_label_color)
        self.lbl_info1.config(font=consts.info_label_font)
        self.lbl_info1.pack(side=tk.LEFT)

        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        l = Label(cur_frame, text="FLIPPER 2", fg=consts.info_label_color)
        l.config(font=consts.info_label_font)
        l.pack(side=tk.LEFT)
        Button(cur_frame, text="IN", command=self.flipper2_in).pack(side=tk.LEFT)
        Button(cur_frame, text="OUT", command=self.control2.flipper_out).pack(side=tk.LEFT)
        self.lbl_info2 = Label(cur_frame, text="State: UNKNOWN", fg=consts.info_label_color)
        self.lbl_info2.config(font=consts.info_label_font)
        self.lbl_info2.pack(side=tk.LEFT)

    def flipper1_in(self):
        if self.control2.get_status().get("state") == "OUT":
            self.control1.flipper_in()

    def flipper2_in(self):
        if self.control1.get_status().get("state") == "OUT":
            self.control2.flipper_in()

    def update(self):
        status1 = self.control1.get_status()
        status2 = self.control2.get_status()

        # connection label
        con_state1 = status1.get("connection", "UNKNOWN")
        con_color1 = consts.con_colors.get(con_state1, "gray")
        self.lbl_status1.config(text=f"STAGE status: {con_state1}", bg=con_color1)

        con_state2 = status2.get("connection", "UNKNOWN")
        con_color2 = consts.con_colors.get(con_state2, "gray")
        self.lbl_status2.config(text=f"STAGE status: {con_state2}", bg=con_color2)

        # position label
        self.lbl_info1.config(text=f"State: {status1.get("state")}")
        self.lbl_info2.config(text=f"State: {status2.get("state")}")
