import tkinter as tk
from tkinter import Frame, Label, Button, Entry, StringVar, Text, Scrollbar
import threading
from core.automation import Automation
import utils.consts as consts
from utils.command_handler import parse_command, Command


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

        # Input field for console
        frame = Frame(self.frame)
        frame.grid(row=1, column=0, sticky=tk.W + tk.E)
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

    def log(self, message):
        # Log messages to the console output
        self.console.insert(tk.END, f"{message}\n")
        self.console.see(tk.END)