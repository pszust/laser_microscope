import logging
import os
import time
from ctypes import *
from tkinter import messagebox

import clr
import cv2
import numpy as np
from PIL import Image, ImageTk

from devices.TC300_COMMAND_LIB import *
from utils.consts import M30Consts
from utils.utils import thread_execute

logger = logging.getLogger(__name__)

# Add References to .NET libraries
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.Benchtop.DCServoCLI.dll.")
clr.AddReference(
    "C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.IntegratedStepperMotorsCLI.dll."
)
clr.AddReference(
    "C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.IntegratedStepperMotorsUI.dll"
)

from System import Decimal  # Required for real units  # type: ignore
from Thorlabs.MotionControl.Benchtop.DCServoCLI import *  # type: ignore
from Thorlabs.MotionControl.DeviceManagerCLI import *  # type: ignore
from Thorlabs.MotionControl.GenericMotorCLI import *  # type: ignore
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *  # type: ignore
from Thorlabs.MotionControl.IntegratedStepperMotorsUI import *  # type: ignore


class StageController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.m30_device = None
        self.state = "IDLE"
        # self.m30_event = m30_event
        # self.m30_param = m30_param
        self.x_pos, self.y_pos = 0, 0
        self.velocity = 2
        logger.debug(f"Initialization done.")

    @thread_execute
    def connect(self):
        self.con_stat = "CONNECTING"

        # connecting (weird thorlabs stuff)
        DeviceManagerCLI.BuildDeviceList()  # type: ignore
        self.m30_device = BenchtopDCServo.CreateBenchtopDCServo(M30Consts.SERIAL_NO)  # type: ignore
        time.sleep(1)
        self.m30_device.Connect(M30Consts.SERIAL_NO)
        print(self.m30_device)

        time.sleep(2.5)  # wait statements are important to allow settings to be sent to the device

        device_info = self.m30_device.GetDeviceInfo()
        print(f"DEVICE_INFO: {device_info}")

        self.x_channel = self.m30_device.GetChannel(1)  # Returns a benchtop channel object
        self.y_channel = self.m30_device.GetChannel(2)

        # Start Polling and enable channel
        self.x_channel.StartPolling(250)
        self.y_channel.StartPolling(250)
        time.sleep(0.25)
        self.x_channel.EnableDevice()
        self.y_channel.EnableDevice()
        time.sleep(0.25)

        # Check that the settings are initialised, else error.
        if not self.x_channel.IsSettingsInitialized() or not self.y_channel.IsSettingsInitialized():
            self.x_channel.WaitForSettingsInitialized(10000)  # 10 second timeout
            self.y_channel.WaitForSettingsInitialized(10000)
            assert self.m30_device.IsSettingsInitialized() is True

        # Load the motor configuration on the channel
        x_config = self.x_channel.LoadMotorConfiguration(self.x_channel.DeviceID)
        y_config = self.y_channel.LoadMotorConfiguration(self.y_channel.DeviceID)

        # Read in the device settings
        dev_settings = self.x_channel.MotorDeviceSettings

        # Get the Homing Params
        x_home_params = self.x_channel.GetHomingParams()
        y_home_params = self.y_channel.GetHomingParams()

        x_home_params.Velocity = Decimal(self.velocity)
        y_home_params.Velocity = Decimal(self.velocity)

        self.x_channel.SetHomingParams(x_home_params)
        self.y_channel.SetHomingParams(y_home_params)

        self.con_stat = "CONNECTED"

    @thread_execute
    def disconnect(self):
        self.x_channel.StopPolling()
        self.y_channel.StopPolling()
        self.m30_device.Disconnect()
        self.m30_device = None
        self.con_stat = "NOT CONNECTED"

    @thread_execute
    def update_position(self):
        if self.m30_device and self.con_stat == "CONNECTED":
            self.x_pos = round(
                float(self.x_channel.DevicePosition.ToString().replace(",", ".")), 3
            )  # warning: this uses , as decimal separator
            self.y_pos = round(float(self.y_channel.DevicePosition.ToString().replace(",", ".")), 3)

    def set_velocity(self, vel: float):
        if M30Consts.MIN_VEL < vel and vel < M30Consts.MAX_VEL:
            self.velocity = vel
            x_home_params = self.x_channel.GetHomingParams()
            y_home_params = self.y_channel.GetHomingParams()
            x_home_params.Velocity = Decimal(self.velocity)
            y_home_params.Velocity = Decimal(self.velocity)
            self.x_channel.SetHomingParams(x_home_params)
            self.y_channel.SetHomingParams(y_home_params)
        else:
            logger.warning(f"Velocity {vel:.2f} outside range")

    @thread_execute
    def set_postion(self, new_x, new_y):
        if self.m30_device and self.con_stat == "CONNECTED":
            if not (M30Consts.MIN_POS < new_x and new_x < M30Consts.MAX_POS) or not (
                M30Consts.MIN_POS < new_y and new_y < M30Consts.MAX_POS
            ):
                logger.warning(f"Position {new_x:.2f}, {new_y:.2f} outside range")
                return

            self.state = "MOVING"
            try:
                self.x_channel.MoveTo(Decimal(new_x), 60000)
                self.y_channel.MoveTo(Decimal(new_y), 60000)
                self.state = "IDLE"
            except AssertionError as error:
                print(error)
                print("LabJack: error, disconecting!")
                self.disconnect()

    def move_rel(self, dx: float, dy: float):
        if self.m30_device and self.con_stat == "CONNECTED":
            new_x = self.x_pos + dx
            new_y = self.y_pos + dy
            self.set_postion(new_x, new_y)

    @thread_execute
    def home(self):
        if self.m30_device and self.con_stat == "CONNECTED":
            self.x_channel.Home(60000)  # 60 second timeout
            self.y_channel.Home(60000)

    def get_status(self):
        self.update_position()
        return {
            "connection": self.con_stat,
            "x_pos": self.x_pos,
            "y_pos": self.y_pos,
            "state": self.state,
            "vel": self.velocity,
        }
