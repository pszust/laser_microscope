import threading
import tkinter as tk
from tkinter import Button, Entry, Frame, Label, Scrollbar, StringVar, OptionMenu

from core.animation import AnimationControl, AnimationInterpreter
import utils.consts as consts
from core.automation import Automation
from utils.command_handler import Command, parse_command
from utils.utils import thread_execute
import logging

logger = logging.getLogger(__name__)


class AnimationTab:
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
        self.var_anim_path = StringVar(value="simple-burn1")
        self.opt_anim_path = OptionMenu(cur_frame, self.var_anim_path, *control.get_anim_files())
        self.opt_anim_path.pack(side=tk.LEFT)

        # Duration
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="Animation duration [s]: ").pack(side=tk.LEFT)
        self.var_duration = StringVar(value="0.0")
        Entry(cur_frame, width=7, textvariable=self.var_duration).pack(side=tk.LEFT, fill=tk.X)


