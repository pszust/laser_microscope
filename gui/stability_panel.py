import threading
import tkinter as tk
from tkinter import LEFT, Button, Entry, Frame, Label, StringVar, X, Y

import utils.consts as consts
from utils.utils import serial_ports, thread_execute


class StabilityPanel:
    def __init__(self, parent, controller):
        self.controller = controller
        self.frame = Frame(parent)
        self.frame.pack(fill=Y)

        # title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text="Stability panel", font=consts.subsystem_name_font).pack(side=LEFT)

        # main info label
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        self.lbl_timings = Label(cur_frame, text="XXX")
        self.lbl_timings.pack(side=LEFT)

    # GUI update hook
    def update(self):
        status = self.controller.report()

        txt = " | ".join(f"{k}: {v:.3f} ms" for k, v in status.items())
        self.lbl_timings.config(text=txt)
