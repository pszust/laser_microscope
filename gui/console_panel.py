import logging
import threading
import tkinter as tk
from tkinter import Button, Entry, Frame, Label, Scrollbar, StringVar, Text

import utils.consts as consts
from core.automation import Automation
from utils.command_handler import Command, parse_command
from utils.utils import TextWidgetHandler


class TkTextHandler(logging.Handler):
    def __init__(self, text_widget: tk.Text):
        super().__init__()
        self.text = text_widget

    def emit(self, record):
        msg = self.format(record)
        tag = record.levelname  # DEBUG/INFO/WARNING/ERROR/CRITICAL

        def append():
            try:
                self.text.insert(tk.END, msg + "\n", tag)
                self.text.see(tk.END)
            except tk.TclError:
                pass

        try:
            self.text.after(0, append)
        except tk.TclError:
            pass


class ConsolePanel:
    def __init__(self, parent: Frame, controller: Automation):
        self.controller = controller
        self.frame = Frame(parent)
        self.frame.pack(fill=tk.Y)

        # console view
        frame = Frame(self.frame)
        frame.grid(row=0, column=0, sticky=tk.W + tk.E)
        self.console = Text(frame, wrap="word", height=25, width=120)
        self.console.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.console.configure(
            bg="#0b1e3a",  # dark blue
            fg="#e6eefc",  # light text
            insertbackground="#e6eefc",
            font=("Consolas", 8),
            padx=6,
            pady=6,
        )

        # Define per-level color tags
        self.console.tag_config("DEBUG", foreground="#7aa2f7")
        self.console.tag_config("INFO", foreground="#c6d8ff")
        self.console.tag_config("WARNING", foreground="#f6c177")
        self.console.tag_config("ERROR", foreground="#ff6b6b")
        self.console.tag_config("CRITICAL", foreground="#ff4d4d", font=("Consolas", 8, "bold"))

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
        self.console_input.configure(
            font=("Consolas", 10),
            bg="#0b1e3a",
            fg="#e6eefc",
            insertbackground="#e6eefc",
        )

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
        root = logging.getLogger()

        # remove any previous text handlers to avoid duplicate/plain lines
        for h in list(root.handlers):
            if isinstance(h, TkTextHandler):
                root.removeHandler(h)

        self.text_handler = TkTextHandler(self.console)
        self.text_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        # keep logger open; filter via handler level (your radio buttons)
        root.setLevel(logging.DEBUG)
        init_level = self.level_map[self.log_level_var.get()]
        self.text_handler.setLevel(init_level)

        root.addHandler(self.text_handler)

    def log(self, message):
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)

    def update_log_level(self):
        level = self.level_map[self.log_level_var.get()]
        self.text_handler.setLevel(level)
