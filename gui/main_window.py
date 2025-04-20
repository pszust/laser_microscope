import sys
import threading
import time
import tkinter as tk
from tkinter import (
    NW,
    Button,
    Canvas,
    E,
    Entry,
    Frame,
    Label,
    Menu,
    N,
    S,
    Scrollbar,
    StringVar,
    Text,
    W,
    filedialog,
    messagebox,
)

import cv2
import serial

from controls.projector_control import ProjectorControl
from gui.projector_panel import ProjectorPanel
import utils.consts as consts
from core.automation import Automation
from devices.camera_control_mock import CameraController
from devices.polar_control_mock import PolarController
from devices.rigol_control_mock import RigolController
from devices.xy_stage_control_mock import StageController
from gui.camera_panel import CameraPanel
from gui.console_panel import ConsolePanel
from gui.polar_panel import PolarPanel
from gui.rigol_panel import RigolPanel
from gui.xy_stage_panel import StagePanel

padd = 2


class MainWindow(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master

        self.camera_controller = CameraController()
        self.rigol_controller = RigolController()
        self.polar1_controller = PolarController()
        self.polar2_controller = PolarController()
        self.stage_controller = StageController()

        self.automation_controller = Automation(self)
        self.projector_control = ProjectorControl(self)

        self.elliptec_angle_var = StringVar()
        self.projector_window = None

        self.create_widgets()
        self.main_loop()
        self.automation_controller.start()  # start main loop of the automation thingy

    def create_widgets(self):
        # Create the menu bar
        self.create_menu()

        # COLUMN 0
        column_frame = Frame(self.master)
        column_frame.grid(row=0, column=0, padx=padd, sticky=N + S + E + W)

        # camera panel
        frame = Frame(column_frame)
        frame.grid(row=0, column=0, padx=padd, sticky=N + S + E + W)
        frame.grid_columnconfigure(0, weight=1)  # Make the frame expand horizontally
        self.camera_panel = CameraPanel(frame, self.camera_controller)

        # console panel
        frame = Frame(column_frame)
        frame.grid(row=1, column=0, padx=padd, sticky=N + S + E + W)
        frame.grid_columnconfigure(0, weight=1)  # Make the frame expand horizontally
        self.console_panel = ConsolePanel(frame, self.automation_controller)

        # COLUMN 1
        column_frame = Frame(self.master)
        column_frame.grid(row=0, column=1, padx=padd, sticky=N + S + E + W)

        # projector panel
        frame = Frame(column_frame)
        frame.pack(fill=tk.Y, padx=padd)
        # self.create_projector_frame(frame)
        self.projector_panel = ProjectorPanel(frame, self.projector_control)

        # rigol panel
        frame = Frame(column_frame)
        frame.pack(fill=tk.Y, padx=padd)
        self.rigol_panel = RigolPanel(frame, self.rigol_controller)

        # polar panel
        frame = Frame(column_frame)
        frame.pack(fill=tk.Y, padx=padd)
        self.polar1_panel = PolarPanel(frame, self.polar1_controller, name="TOP POLARIZER CONTROL")

        # polar panel 2
        frame = Frame(column_frame)
        frame.pack(fill=tk.Y, padx=padd)
        self.polar2_panel = PolarPanel(frame, self.polar2_controller, name="BOTTOM POLARIZER CONTROL")

        # xy-stage panel
        frame = Frame(column_frame)
        frame.pack(fill=tk.Y, padx=padd)
        self.stage_panel = StagePanel(frame, self.stage_controller)

    def create_menu(self):
        self.menu = Menu(self.master)
        self.master.config(menu=self.menu)

        self.file_menu = Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Select and execute script", command=self.sel_and_exe_scr)
        self.file_menu.add_command(label="Exit", command=self.master.quit)
        
        self.proj_menu = Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="Projector", menu=self.proj_menu)
        self.proj_menu.add_command(label="Load Image", command=self.load_image)

    def main_loop(self):
        self.update_labels()

        self.camera_panel.update_image()

        # self.automation_controller.execute()

        self.master.after(consts.main_loop_time, self.main_loop)

    def update_labels(self):
        self.rigol_panel.update()
        self.polar1_panel.update()
        self.polar2_panel.update()
        self.stage_panel.update()
        self.projector_panel.update()

    def load_image(self):
        # Placeholder function to load image
        filename = filedialog.askopenfilename(
            initialdir="/",
            title="Select Image",
            filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")),
        )
        if filename:
            self.projector_control.load_and_set_image(filename)

    def set_gain(self):
        # Placeholder function to set camera gain
        gain = self.gain_value.get()
        self.log(f"Camera gain set to {gain}.")

    def save_image(self):
        # Placeholder function to save image
        self.log("Save image clicked.")

    def sel_and_exe_scr(self):
        filename = filedialog.askopenfilename(
            initialdir="custom_scripts/",
            title="Select Script",
            filetypes=(("Script files", "*.scrpt"),),
        )
        if filename:
            self.automation_controller.execute_script_file(filename)
