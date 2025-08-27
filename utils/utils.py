import functools
import glob
import logging
import sys
import threading
import tkinter as tk

import cv2
import numpy as np
import serial
from matplotlib import pyplot as plt

from utils.command_handler import Command
from utils.consts import ProjConsts


def thread_execute(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True).start()

    return wrapper


def serial_ports():
    """Lists serial port names

    :raises EnvironmentError:
        On unsupported or unknown platforms
    :returns:
        A list of the serial ports available on the system
    """
    return ["COM3", "COM5", "COM6", "COM9", "COM10"]  # temporary

    if sys.platform.startswith("win"):
        ports = ["COM%s" % (i + 1) for i in range(256)]
    elif sys.platform.startswith("linux") or sys.platform.startswith("cygwin"):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob("/dev/tty[A-Za-z]*")
    elif sys.platform.startswith("darwin"):
        ports = glob.glob("/dev/tty.*")
    else:
        raise EnvironmentError("Unsupported platform")

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result


class TextWidgetHandler(logging.Handler):
    def __init__(self, text_widget: tk.Text, min_level=logging.INFO):
        super().__init__()
        self.text_widget = text_widget
        self.min_level = min_level

    def emit(self, record):
        if record.levelno < self.min_level:
            return
        msg = self.format(record)
        self.text_widget.after(0, self.text_widget.insert, tk.END, msg + "\n")
        self.text_widget.after(0, self.text_widget.see, tk.END)

    def set_level(self, level):
        self.min_level = level


calib_dots_dim = 4


def num_to_coords(num, size=4, dim=(ProjConsts.PROJ_IMG_SHAPE[0], ProjConsts.PROJ_IMG_SHAPE[1])):
    nx = int(num % size)
    ny = int((num - num % size) / size)

    inc_x = int(dim[0] / size)
    inc_y = int(dim[1] / size)

    x = int(inc_x / 2) + nx * inc_x
    y = int(inc_y / 2) + ny * inc_y

    #     print('x = %d, y = %d'%(x, y))
    return x, y


def get_homography_matrix():
    rgb_weights = [0.2989, 0.5870, 0.1140]
    baseline = np.dot(np.load("calibration/calibration_array_baseline.npy")[..., :3], rgb_weights)
    # images = []
    # for i in range(0, calib_dots_dim**2):
    # temp = np.load('calibration/num%d.npy'%i)
    # images.append(np.dot(temp[...,:3], rgb_weights))

    coords_prj = []
    coords_cam = []
    sigma = 11

    combine = baseline.copy() * 0
    for i in range(0, calib_dots_dim**2):
        img = np.load(f"calibration/calibration_array_{str(i).zfill(2)}.npy")
        img = np.dot(img[..., :3], rgb_weights)
        bimg = img - baseline
        combine += bimg / calib_dots_dim**2
        bimg = cv2.GaussianBlur(bimg, (sigma, sigma), cv2.BORDER_DEFAULT)
        x, y = np.where(bimg == bimg.max())
        c = [y[0], x[0]]
        coords_cam.append(c)
        crds = num_to_coords(i, size=calib_dots_dim)
        coords_prj.append([crds[0], crds[1]])

    coords_prj_arr = np.array(coords_prj)
    coords_cam_arr = np.array(coords_cam)

    #     return coords_prj_arr, coords_cam_arr
    h, status = cv2.findHomography(coords_cam_arr, coords_prj_arr)
    im_out = cv2.warpPerspective(baseline, h, (1024, 768))

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_title("camera image")
    ax1.imshow(combine, cmap="gray")
    for i, crd in enumerate(coords_cam):
        ax1.text(crd[0], crd[1], str(i), c="red")

    ax2.set_title("projector array")
    ax2.imshow(im_out, cmap="gray")
    for i, crd in enumerate(coords_prj):
        ax2.text(crd[0], crd[1], str(i), c="white")

    fig.savefig("calibration/calib_result.png", dpi=400)

    return h


def print_command(cmd, nest=0):
    if type(cmd) == Command:
        print(f"{"   "*nest}{cmd.get_format()}")
    elif type(cmd) == list:
        for c in cmd:
            print_command(c, nest=nest + 1)
