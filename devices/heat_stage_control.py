import logging
import time
from tkinter import messagebox

import pyvisa

from utils.utils import thread_execute

logger = logging.getLogger(__name__)


class HeatController:
    def __init__(self):
        self.heat_control = None
        self.con_stat = "UNKNOWN"
        self.cur_temp = -1.0
        self.set_temp = -1.0
        self.rate = -1.0
        logger.debug(f"Initialization done.")

    @thread_execute
    def connect(self):
        if self.heat_control is not None:
            messagebox.showinfo(title="Heat stage", message="Heat stage is already connected.")
            return True
        else:
            self.con_stat = "CONNECTING"
            time.sleep(1)
            self.heat_control = 1
            self.con_stat = "CONNECTED"

    @thread_execute
    def disconnect(self):
        if self.heat_control:
            self.heat_control = None
            self.con_stat = "NOT CONNECTED"

    def get_status(self):
        return {
            "connection": self.con_stat,
            "current_temp": self.cur_temp,
            "set_temp": self.set_temp,
            "rate": self.rate,
        }

    @thread_execute
    def set_temperature(self, value):
        self.set_temp = float(value)

    @thread_execute
    def set_rate(self, value):
        self.rate = float(value)
