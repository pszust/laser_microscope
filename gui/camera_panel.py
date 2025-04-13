import tkinter as tk
from tkinter import Frame, Label, Button, Entry, StringVar, LEFT, X, Y
import threading
import utils.consts as consts
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
from PIL import Image, ImageTk
from utils.utils import thread_execute


class CameraPanel:
    def __init__(self, parent, controller):
        self.controller = controller
        self.frame = Frame(parent)
        self.camera_image = Image.new("RGB", (800, 600), color="grey")

        self.canvas = Canvas(parent, width=800, height=600, bg="black")
        self.canvas.grid(row=0, column=0, sticky=W + E)

    def display_image(self):
        # Display the selected image in the canvas
        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.camera_image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    @thread_execute
    def update_image(self):
        self.camera_image = self.controller.get_image()
        self.display_image()
