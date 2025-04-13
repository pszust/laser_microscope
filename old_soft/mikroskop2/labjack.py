import os
import time
import sys
import clr
import numpy as np

# Add References to .NET libraries
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.Benchtop.DCServoCLI.dll.")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.IntegratedStepperMotorsCLI.dll.")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.IntegratedStepperMotorsUI.dll")


from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.Benchtop.DCServoCLI import *
from Thorlabs.MotionControl.IntegratedStepperMotorsUI import *
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import *
from System import Decimal  # Required for real units


DeviceManagerCLI.BuildDeviceList()
serial_no = "49337314"
device = LabJack.CreateLabJack(serial_no)
device.Connect(serial_no)
time.sleep(0.25)  # wait statements are important to allow settings to be sent to the device

# Get Device Information and display description
device_info = device.GetDeviceInfo()
print(device_info.Description)
device.StartPolling(250)


time.sleep(1)
device.Disconnect()