import threading
import tkinter as tk
from tkinter import Button, Entry, Frame, Label, Scrollbar, StringVar, OptionMenu

from core.animation import AnimationControl, AnimationInterpreter
import utils.consts as consts
from core.automation import Automation
from utils.command_handler import Command, parse_command
from utils.utils import thread_execute


class ArealTab:
    def __init__(self, parent, control: AnimationControl):
        self.control = control
        self.frame = Frame(parent)
        self.frame.pack(fill=tk.Y)

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="ANIMATION TAB", font=consts.subsystem_name_font).pack(side=tk.LEFT)

        # Select animation
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="Select animation: ").pack(side=tk.LEFT)

        self.var_anim_file = StringVar(value="simple-burn1")
        self.mnSDfreq = OptionMenu(cur_frame, self.var_anim_file, *control.get_anim_files())
        self.mnSDfreq.pack(side=tk.LEFT)
