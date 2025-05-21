import logging
import threading
import tkinter as tk
from tkinter import Button, Entry, Frame, Label, Scrollbar, StringVar, Text

import utils.consts as consts
from core.automation import Automation
from utils.command_handler import Command, parse_command
from utils.utils import TextWidgetHandler


class ConsolePanel:
    def __init__(self, parent: Frame, controller: Automation):
        self.controller = controller
        self.frame = Frame(parent)
        self.frame.pack(fill=tk.Y)

        # console view
        frame = Frame(self.frame)
        frame.grid(row=0, column=0, sticky=tk.W + tk.E)
        self.console = Text(frame, wrap="word", height=15)
        self.console.pack(side=tk.LEFT, fill=tk.X, expand=True)

        scroll_bar = Scrollbar(frame, command=self.console.yview)
        scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)
        self.console.configure(yscrollcommand=scroll_bar.set)

        # Logger with selectable level logging thingy
        frame = Frame(self.frame)
        frame.grid(row=1, column=0, sticky=tk.W + tk.E)
        self.setup_logger(frame)

        # Input field for console
        frame = Frame(self.frame)
        frame.grid(row=2, column=0, sticky=tk.W + tk.E)
        self.console_input_label = Label(frame, text="Input Command:")
        self.console_input_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.console_input = Entry(frame, width=50)
        self.console_input.pack(side=tk.LEFT, fill=tk.BOTH)
        self.console_input.bind("<Return>", self.process_console_input)

    def process_console_input(self, event):
        # Process input from the console input field
        command = self.console_input.get()
        self.log(f"Command entered: {command}")
        self.controller.pass_command(command)
        self.console_input.delete(0, tk.END)

    def setup_logger(self, frame):
        self.log_level_var = tk.StringVar(value="INFO")
        self.level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        Label(frame, text="Log Level:").pack(side=tk.LEFT)
        for level in self.level_map:
            tk.Radiobutton(
                frame, text=level, variable=self.log_level_var, value=level, command=self.update_log_level
            ).pack(side=tk.LEFT)
        self.attach_logger()

    def attach_logger(self):
        self.text_handler = TextWidgetHandler(self.console)
        self.text_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(self.text_handler)

    def log(self, message):
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)

    def update_log_level(self):
        level = self.level_map[self.log_level_var.get()]
        self.text_handler.set_level(level)
