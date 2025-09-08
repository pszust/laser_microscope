import logging
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
    ttk,
)

import cv2
import serial

import utils.consts as consts
from controls.chiral_control import ChiralControl
from controls.image_control import ImageControl
from controls.projector_control import ProjectorControl
from core.animation import AnimationControl
from core.automation import Automation
from gui.animation_tab import AnimationTab
from gui.camera_panel import CameraPanel
from gui.chiral_tab import ChiralTab
from gui.console_panel import ConsolePanel
from gui.flipper_panel import FlipperPanel
from gui.heat_stage_panel import HeatPanel
from gui.image_tab import ImageTab
from gui.labjack_panel import LabjackPanel
from gui.polar_panel import PolarPanel
from gui.projector_panel import ProjectorPanel
from gui.rigol_panel import RigolPanel
from gui.stability_panel import StabilityPanel
from gui.xy_stage_panel import M30Panel
from utils.timer import LoopTimer

if consts.Device.USE_REAL_CAMERA:
    from devices.camera_control import CameraController
else:
    from devices.camera_control_mock import CameraController

if consts.Device.USE_REAL_FLIPPERS:
    from devices.flipper_controller import FlipperController
else:
    from devices.flipper_controller_mock import FlipperController

if consts.Device.USE_REAL_LABJACK:
    from devices.labjack_controller import LabjackController
else:
    from devices.labjack_controller_mock import LabjackController

if consts.Device.USE_REAL_HEATSTAGE:
    from devices.heat_stage_control import HeatController
else:
    from devices.heat_stage_control_mock import HeatController

if consts.Device.USE_REAL_RIGOL:
    from devices.rigol_control import RigolController
else:
    from devices.rigol_control_mock import RigolController

if consts.Device.USE_REAL_M30:
    from devices.m30_control import StageController
else:
    from devices.m30_control_mock import StageController

if consts.Device.USE_REAL_ROTATOR:
    from devices.polar_control import PolarController
else:
    from devices.polar_control_mock import PolarController


logger = logging.getLogger(__name__)
padd = 2


class MainWindow(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        # self.loop_timer = LoopTimer()
        self.master = master

        self.camera_controller = CameraController()
        self.rigol_controller = RigolController()
        self.polar1_controller = PolarController()
        self.polar2_controller = PolarController()
        self.stage_controller = StageController()
        self.labjack_controller = LabjackController()
        self.flipper1_control = FlipperController(1)
        self.flipper2_control = FlipperController(2)
        self.heat_stage_control = HeatController()

        self.projector_control = ProjectorControl(self)
        self.animation_control = AnimationControl(self)
        self.image_control = ImageControl(self)
        self.chiral_control = ChiralControl(self)
        self.automation_controller = Automation(self)  # this has to be after all the controllers

        self.elliptec_angle_var = StringVar()
        self.projector_window = None

        # layout setup
        self.master.grid_columnconfigure(0, weight=2)
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_columnconfigure(2, weight=1)
        self.master.grid_columnconfigure(3, weight=2)

        self.create_widgets()
        self.create_menu()
        self.bind_keys()
        self.main_loop()
        self.camera_image_loop()
        self.automation_controller.start()  # start main loop of the automation thingy

    def create_widgets(self):
        # -- COLUMN 0 --
        column_frame = Frame(self.master)
        column_frame.grid(row=0, column=0, padx=padd, sticky=N + S + E + W)

        # camera panel
        frame = Frame(column_frame)
        frame.grid(row=0, column=0, padx=padd, sticky=N + S + E + W)
        frame.grid_columnconfigure(0, weight=1)  # Make the frame expand horizontally
        self.camera_panel = CameraPanel(frame, self.camera_controller, self)

        # console panel
        frame = Frame(column_frame)
        frame.grid(row=1, column=0, padx=padd, sticky=N + S + E + W)
        frame.grid_columnconfigure(0, weight=1)  # Make the frame expand horizontally
        self.console_panel = ConsolePanel(frame, self.automation_controller)

        # stability panel
        # frame = Frame(column_frame)
        # frame.grid(row=2, column=0, padx=padd, sticky=N + S + E + W)
        # frame.grid_columnconfigure(0, weight=1)  # Make the frame expand horizontally
        # self.stability_panel = StabilityPanel(frame, self.loop_timer)

        # -- COLUMN 1 --
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

        # -- COLUMN 2 --
        column_frame = Frame(self.master)
        column_frame.grid(row=0, column=2, padx=padd, sticky=N + S + E + W)

        # xy-stage panel
        frame = Frame(column_frame)
        frame.pack(fill=tk.Y, padx=padd)
        self.stage_panel = M30Panel(frame, self.stage_controller)

        # labjack panel
        frame = Frame(column_frame)
        frame.pack(fill=tk.Y, padx=padd)
        self.labjack_panel = LabjackPanel(frame, self.labjack_controller)

        # flippers panel
        frame = Frame(column_frame)
        frame.pack(fill=tk.Y, padx=padd)
        self.flipper_panel = FlipperPanel(frame, [self.flipper1_control, self.flipper2_control])

        # heat stage panel
        frame = Frame(column_frame)
        frame.pack(fill=tk.Y, padx=padd)
        self.heat_panel = HeatPanel(frame, self.heat_stage_control)

        # -- COLUMN 3 --
        column_frame = Frame(self.master)
        column_frame.grid(row=0, column=3, padx=padd, sticky=N + S + E + W)

        # Notebook widget creates tabs
        notebook = ttk.Notebook(column_frame)
        notebook.pack(fill="both", expand=True)

        # Tab 1 (ANMT)
        tab_anim = Frame(notebook)
        notebook.add(tab_anim, text="ANMT")

        # Add widgets to tab_gui1
        self.anim_tab = AnimationTab(tab_anim, self.animation_control)

        # Tab 2 (AERL)
        tab_gui2 = Frame(notebook)
        notebook.add(tab_gui2, text="AREL")

        # Tab 3 (CHRL)
        tab_gui3 = Frame(notebook)
        notebook.add(tab_gui3, text="CHRL")
        self.chiral_tab = ChiralTab(tab_gui3, self.chiral_control)

        # Tab 4 (IMAG)
        tab_gui4 = Frame(notebook)
        notebook.add(tab_gui4, text="IMAG")
        self.chiral_tab = ImageTab(tab_gui4, self.image_control)

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

    def bind_keys(self):
        self.master.bind("[", lambda _: self.camera_panel.change_brush_size(-1))
        self.master.bind("]", lambda _: self.camera_panel.change_brush_size(1))
        self.master.protocol("WM_DELETE_WINDOW", self.exit)
        self.master.bind("<Escape>", self.defocus_view_btn)

    def main_loop(self):
        # self.loop_timer.start_loop()
        self.update_labels()
        # self.loop_timer.event("LABELS_UPDATED")

        self.animation_control.loop_event()
        # self.loop_timer.event("ANIMATIONS")

        self.camera_panel.update_image()
        # self.loop_timer.event("CAMERA_PANEL")

        # self.automation_controller.execute()
        self.master.after(consts.MAIN_LOOP_TIME, self.main_loop)
        # self.loop_timer.end_loop()

    def camera_image_loop(self):
        self.camera_panel.update_image()
        self.master.after(consts.CAM_IMG_LOOP, self.camera_image_loop)

    def update_labels(self):
        # TODO: need to split some logic into  main loop kind of method because some of these are not just for labels but they execute internal
        # TODO: functionality of the modules/connectors
        self.rigol_panel.update()
        self.polar1_panel.update()
        self.polar2_panel.update()
        self.stage_panel.update()
        self.projector_panel.update()  # this is more than labels
        self.labjack_panel.update()
        self.flipper_panel.update()
        self.camera_panel.update()
        # self.stability_panel.update()

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

    def exit(self):
        logger.info("Exiting the software!")
        self.camera_controller.exit_camera()  # to close CamReader
        self.master.quit()

    def defocus_view_btn(self, value):
        self.camera_panel.connect_to_cam_btn.focus_set()
