import tkinter as tk
import cv2
from tkinter import (
    Frame,
    Label,
    Button,
    Entry,
    Text,
    StringVar,
    Canvas,
    Menu,
    filedialog,
    Scrollbar,
    END,
    LEFT,
    RIGHT,
    BOTH,
    Y,
    X,
    W,
    E,
    N,
    S,
    NW,
)
from devices.camera_control_mock import CameraController
from devices.polar_control_mock import PolarController
from devices.rigol_control_mock import RigolController
from gui.camera_panel import CameraPanel
from gui.polar_panel import PolarPanel
from gui.rigol_panel import RigolPanel
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import time
import serial
import sys
import threading
import utils.consts as consts


padd = 2


class MainWindow(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master

        self.camera_controller = CameraController()
        self.rigol_controller = RigolController()
        self.polar1_controller = PolarController()

        self.elliptec_angle_var = StringVar()
        self.projector_window = None

        self.create_widgets()
        self.main_loop()

    def create_widgets(self):
        # Create the menu bar
        self.create_menu()

        # COLUMN 0
        column_frame = Frame(self.master)
        column_frame.grid(row=0, column=0, padx=padd, sticky=N + S + E + W)

        frame = Frame(column_frame)
        frame.grid(row=0, column=0, padx=padd, sticky=N + S + E + W)
        frame.grid_columnconfigure(0, weight=1)  # Make the frame expand horizontally
        self.camera_panel = CameraPanel(frame, self.camera_controller)

        frame = Frame(column_frame)
        frame.grid(row=1, column=0, padx=padd, sticky=N + S + E + W)
        frame.grid_columnconfigure(0, weight=1)  # Make the frame expand horizontally
        self.create_console_frame(frame)

        # COLUMN 1
        column_frame = Frame(self.master)
        column_frame.grid(row=0, column=1, padx=padd, sticky=N + S + E + W)

        frame = Frame(column_frame)
        frame.pack(fill=Y, padx=padd)
        self.create_projector_frame(frame)

        frame = Frame(column_frame)
        frame.pack(fill=Y, padx=padd)
        self.rigol_panel = RigolPanel(frame, self.rigol_controller)

        frame = Frame(column_frame)
        frame.pack(fill=Y, padx=padd)
        self.polar1_panel = PolarPanel(frame, self.polar1_controller, name="TOP POLARIZER CONTROL")

    def create_menu(self):
        self.menu = Menu(self.master)
        self.master.config(menu=self.menu)
        self.file_menu = Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Load Image", command=self.load_image)
        self.file_menu.add_command(label="Exit", command=self.master.quit)

    def create_console_frame(self, parent):
        # console view
        frame = Frame(parent)
        frame.grid(row=0, column=0, sticky=W + E)
        self.console = Text(frame, wrap="word", height=15)
        self.console.pack(side=LEFT, fill=X, expand=True)

        scroll_bar = Scrollbar(frame, command=self.console.yview)
        scroll_bar.pack(side=RIGHT, fill=Y)
        self.console.configure(yscrollcommand=scroll_bar.set)

        # Input field for console
        frame = Frame(parent)
        frame.grid(row=1, column=0, sticky=W + E)
        self.console_input_label = Label(frame, text="Input Command:")
        self.console_input_label.pack(side=LEFT, fill=BOTH, expand=True)

        self.console_input = Entry(frame, width=50)
        self.console_input.pack(side=LEFT, fill=BOTH)
        self.console_input.bind("<Return>", self.process_console_input)

    def create_projector_frame(self, frame):
        cur_frame = Frame(frame)
        cur_frame.pack(fill=Y)

        self.labLaser = Label(cur_frame, text="PROJECTOR CONTROL")
        self.labLaser.config(font=consts.subsystem_name_font)
        self.labLaser.pack(side=LEFT)

        cur_frame = Frame(frame)
        # proj_frame1.grid(row=1, column=0, padx = padd)
        cur_frame.pack(fill=Y)

        self.init_proj_win_btn = Button(
            cur_frame, text="Init window", command=self.initiate_projector_window
        )
        self.init_proj_win_btn.pack(side=LEFT)

        self.act_proj_win_btn = Button(
            cur_frame, text="Activate window", command=self.activate_projector_window
        )
        self.act_proj_win_btn.pack(side=LEFT)

        self.act_proj_win_btn = Button(cur_frame, text="Close window", command=self.close_projector_window)
        self.act_proj_win_btn.pack(side=LEFT)

        cur_frame = Frame(frame)
        # canvas_frame.grid(row=2, column=0, padx = padd)
        cur_frame.pack(fill=Y)

        self.proj_mirror_canvas = Canvas(cur_frame, width=256, height=192, bg="black")
        self.proj_mirror_canvas.pack(side=LEFT)

    def main_loop(self):
        self.update_labels()

        self.camera_panel.update_image()

        self.master.after(100, self.main_loop)

    def update_labels(self):
        self.rigol_panel.update()
        self.polar1_panel.update()

    def load_image(self):
        # Placeholder function to load image
        filename = filedialog.askopenfilename(
            initialdir="/",
            title="Select Image",
            filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")),
        )
        if filename:
            self.display_image(filename)
            self.log(f"Loaded image: {filename}")

    def set_gain(self):
        # Placeholder function to set camera gain
        gain = self.gain_value.get()
        self.log(f"Camera gain set to {gain}.")

    def save_image(self):
        # Placeholder function to save image
        self.log("Save image clicked.")

    def process_console_input(self, event):
        # Process input from the console input field
        command = self.console_input.get()
        self.log(f"Command entered: {command}")
        self.console_input.delete(0, END)

    def log(self, message):
        # Log messages to the console output
        self.console.insert(END, f"{message}\n")
        self.console.see(END)

    def initiate_projector_window(self):
        if self.projector_window == None:
            # self.projector_window = ProjectorWindow(root)
            # self.app = ProjectorWindow(self.projector_window)
            self.projector_window = tk.Toplevel(self.master)
            self.projector_window.title("Projector window - move to projector screen")
            self.projector_window.geometry("400x400")
            self.log("Opened projector window")

    def close_projector_window(self):
        if self.projector_window != None:
            self.projector_window.destroy()
            self.projector_window = None
            self.log("Closed projector window")

    def activate_projector_window(self):
        print("Projector window activated!")

        # initialize full screen mode
        self.projector_window.overrideredirect(True)
        self.projector_window.state("zoomed")
        # self.projector_window.activate()

        self.canvas_proj = Canvas(
            self.projector_window,
            width=1024,
            height=768,
            bg="black",
            highlightthickness=0,
            relief="ridge",
        )
        self.canvas_proj.pack(side=LEFT)
        self.log("Projector window activated")

    def load_pattern_image(self, path):
        self.projector_arr = cv2.imread(path)
        self.refresh_projector_image()
        self.log("Image %s loaded" % path)

    def refresh_projector_image(self):
        # refresh image displayed in window (4x smaller res)
        img = cv2.resize(self.projector_arr, (256, 192), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(img)
        self.proj_imgtk_mirror = ImageTk.PhotoImage(image=img)
        # self.proj_mirror_canvas.create_image(128, 96, image=self.proj_imgtk, anchor=CENTER)
        self.proj_mirror_canvas.create_image(0, 0, image=self.proj_imgtk_mirror, anchor=NW)

        # refresh the actual screen
        img = Image.fromarray(self.projector_arr)
        self.proj_imgtk = ImageTk.PhotoImage(image=img)
        self.canvas_proj.create_image(512, 384, image=self.proj_imgtk, anchor=tk.CENTER)
