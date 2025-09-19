import logging
import time
from tkinter import messagebox

import serial

from utils.utils import thread_execute

logger = logging.getLogger(__name__)


RAMP_UPDATE_COUNTER = 10


class HeatController:
    def __init__(self):
        self.device = None
        self.con_stat = "UNKNOWN"
        self.port = 0
        self.cur_temp = -1.0
        self.set_temp = -1.0
        self.ramp_rate = -1.0
        self.ramp_temp = -1.0  # set temp but for ramp
        self.ramp_counter = 0  # counter to update set_temp when ramping
        self.ramp_timer = 0
        self.ramp_active = False
        logger.debug(f"Initialization done.")

    @thread_execute
    def connect(self):
        if self.device is not None:
            messagebox.showinfo(title="Heat stage", message="Heat stage is already connected.")
            return True
        else:
            port = f"COM{self.port}"
            self.con_stat = "CONNECTING"
            ser = serial.Serial()
            ser.baudrate = 115200
            ser.port = port
            ser.timeout = 6

            ser.parity = serial.PARITY_NONE
            ser.bytesize = serial.EIGHTBITS
            ser.stopbits = serial.STOPBITS_ONE

            try:
                ser.open()
                ser.write(b"tact?\r")
                resp = ser.read_until(b">")
                if str(resp).find("tact?") >= 0:
                    logging.info(f"Connected to thortemp on {port}")
                    self.device = ser
                    self.con_stat = "CONNECTED"
                else:
                    logging.info(f"{port} is not thortemp")
                    logging.info("Response: %s" % resp)
                    return None
            except:
                logging.warning(f"Device on {port} is not available")
                return None

    @thread_execute
    def disconnect(self):
        if self.device:
            self.device = None
            self.con_stat = "NOT CONNECTED"

    def get_status(self):
        self.read_temp()
        return {
            "connection": self.con_stat,
            "current_temp": self.cur_temp,
            "set_temp": self.set_temp,
            "ramp_rate": self.ramp_rate,
            "ramp_temp": self.ramp_temp,
            "tramp_active": self.ramp_active,
        }

    @thread_execute
    def set_temperature(self, value: float):
        if self.device != None:
            self.set_temp = value
            self.device.write(b"tset=%2.1f" % value)

    def set_ramp_rate(self, value):
        self.ramp_rate = float(value)

    def set_ramp_temp(self, value):
        self.ramp_temp = float(value)

    def activate_ramp(self):
        if self.device != None:
            if self.ramp_active:
                self.ramp_active = False
            else:
                self.ramp_active = True
                self.set_temp = self.cur_temp

    def thortemp_switch_enable(self):
        if self.device != None:
            self.device.write(b"ens\r")

    @thread_execute
    def read_temp(self):
        if self.device != None:
            self.device.write(b"tact?\r")
            resp = self.device.read_until(b">")
            resp_str = str(resp)
            if resp_str.find("tact") >= 0:
                try:
                    self.cur_temp = float(resp_str[resp_str.find("\\r") + 2 : resp_str.find("C\\") - 1])
                except:
                    print("Some stupid error occured, but dont worry!")
                    print(resp_str)
        return -1

    def update_ramp(self):
        if self.device != None:
            self.ramp_counter += 1
            if self.ramp_counter > RAMP_UPDATE_COUNTER:
                self.ramp_counter = 0
                sec_passed = time.process_time() - self.ramp_timer
                temp_step = sec_passed * self.ramp_rate / 60
                self.ramp_timer = time.process_time()

                if self.ramp_active:
                    temp_diff = self.ramp_temp - self.set_temp
                    if abs(temp_diff) < temp_step:
                        self.set_temp = self.ramp_temp
                        self.ramp_active = False
                    elif temp_diff > 0:
                        self.set_temp += temp_step
                    elif temp_diff < 0:
                        self.set_temp -= temp_step
                    self.set_temperature(self.set_temp)
