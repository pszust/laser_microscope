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
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.IntegratedStepperMotorsCLI.dll.")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.IntegratedStepperMotorsUI.dll")

# I dont know how it works, ask thorlabs
from Thorlabs.MotionControl.DeviceManagerCLI import *  # type: ignore
from Thorlabs.MotionControl.GenericMotorCLI import *  # type: ignore
from Thorlabs.MotionControl.Benchtop.DCServoCLI import *  # type: ignore
from Thorlabs.MotionControl.IntegratedStepperMotorsUI import *  # type: ignore
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *  # type: ignore
from System import Decimal  # Required for real units  # type: ignore


class LabjackController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.height = 0.0
        self.labjack = None
        logger.debug(f"Initialization done.")

    @thread_execute
    def connect(self):
        self.con_stat = "CONNECTING"
        
        # connecting (weird thorlabs stuff)
        DeviceManagerCLI.BuildDeviceList()  # type: ignore
        self.labjack = LabJack.CreateLabJack(LabJackConsts.SERIAL_NO)  # type: ignore

        # Connect, begin polling, and enable
        self.labjack.Connect(LabJackConsts.SERIAL_NO)
        time.sleep(0.25)  # wait statements are important to allow settings to be sent to the device
        
        device_info = self.labjack.GetDeviceInfo()
        print(f"DEVICE_INFO: {device_info}")
        
        self.labjack.StartPolling(250)
        self.labjack.EnableDevice()
        motor_configuration = self.labjack.LoadMotorConfiguration(LabJackConsts.SERIAL_NO)
        work_done = self.labjack.InitializeWaitHandler()
        time.sleep(0.1)

        self.con_stat = "CONNECTED"

    @thread_execute
    def disconnect(self):
        self.labjack.StopPolling()
        self.labjack.Disconnect()
        self.labjack = None
        self.con_stat = "NOT CONNECTED"

    @thread_execute
    def update_height(self):
        if self.labjack and self.con_stat == "CONNECTED":
            self.height = round(float(self.labjack.Position.ToString().replace(',', '.')), 3)

    @thread_execute
    def set_height(self, value):
        if value < LabJackConsts.MIN_POS or LabJackConsts.MAX_POS < value:
            # TODO: proper logging needs to implemented
            err_msg = f"Requested z-value = {value:.3f} for labjack is outside the range "
            err_msg += f"({LabJackConsts.MIN_POS} to {LabJackConsts.MAX_POS})"
            print(err_msg)
        elif self.labjack and self.con_stat == "CONNECTED" and self.labjack.Status.IsInMotion == False:
            try:
                time.sleep(0.1)
                work_done = self.labjack.InitializeWaitHandler()
                self.labjack.MoveTo(Decimal(value), work_done)
            except AssertionError as error:
                print(error)
                print('LabJack: error, disconecting!')
                self.disconnect()

    @thread_execute
    def home(self):
        work_done = self.labjack.InitializeWaitHandler()
        self.labjack.Home(work_done)

    @thread_execute
    def move_relative(self, value):
        self.update_height()
        time.sleep(0.1)
        new_height = self.height + value
        self.set_height(new_height)

    def get_status(self):
        self.update_height()
        return {
            "connection": self.con_stat,
            "height": self.height,
        }
