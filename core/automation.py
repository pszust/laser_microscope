import time
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

class Automation:
    def __init__(self, parent):
        self.master = parent
