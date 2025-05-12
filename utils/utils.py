import functools
import glob
import sys
import threading

import logging
import tkinter as tk
import serial


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
