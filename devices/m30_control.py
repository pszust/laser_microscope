import os
import time
from tkinter import messagebox

import cv2
import numpy as np
from utils.consts import LabJackConsts
from utils.utils import thread_execute
from PIL import Image, ImageTk

import clr
from ctypes import *
from devices.TC300_COMMAND_LIB import *
import logging

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

from Thorlabs.MotionControl.DeviceManagerCLI import *  # type: ignore
from Thorlabs.MotionControl.GenericMotorCLI import *  # type: ignore
from Thorlabs.MotionControl.Benchtop.DCServoCLI import *  # type: ignore
from Thorlabs.MotionControl.IntegratedStepperMotorsUI import *  # type: ignore
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *  # type: ignore
from System import Decimal  # Required for real units  # type: ignore


class M30Control:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.m30_device = None
        # self.m30_event = m30_event
        # self.m30_param = m30_param
        self.x_pos, self.y_pos = 0, 0
        self.curAcc = 5
        self.curVel = 2
        logger.debug(f"Initialization done.")

    @thread_execute
    def connect(self):
        self.con_stat = "CONNECTING"

        # connecting (weird thorlabs stuff)
        DeviceManagerCLI.BuildDeviceList()  # type: ignore
        serial_no = "101507134"  # Replace this line with your device's serial number
        self.m30_device = BenchtopDCServo.CreateBenchtopDCServo(serial_no)  # type: ignore
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

        x_home_params.Velocity = Decimal(2.0)
        y_home_params.Velocity = Decimal(2.0)

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

    @thread_execute
    def set_postion(self, new_x, new_y):
        if self.m30_device and self.con_stat == "CONNECTED":
            try:
                self.x_channel.MoveTo(Decimal(new_x), 60000)
                self.y_channel.MoveTo(Decimal(new_y), 60000)
            except AssertionError as error:
                print(error)
                print("LabJack: error, disconecting!")
                self.disconnect()

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
        }
