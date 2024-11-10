import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw, ImageChops
import os
import pandas as pd
# import sys, ftd2xx as ftd
import numpy as np
import serial
import sys
import glob
import time
from threading import Thread
import datetime
# from scipy.ndimage.filters import gaussian_filter, maximum_filter
import pyvisa
import random
import imutils
from multiprocessing import Process
from multiprocessing import Value
from multiprocessing import RawArray
from multiprocessing import Event
import threading
# from inputs import get_gamepad
import math
import clr
import subprocess
from ctypes import *
from TC300_COMMAND_LIB import *

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


ljLimits = [0, 40]

group_name_font = ('Segoe UI', 16)

padd = 10
camRes = (448, 800, 3)
wd = 'c:/Users/LAB218/Documents/mikroskop2/'

laser_on_color = '#772eff'
laser_off_color = '#5d615c'

group_name_font = ('Segoe UI', 16)
subsystem_name_font = ('Segoe UI', 14, 'bold')


# DeviceManagerCLI.BuildDeviceList()
# serial_no = "49337314"
# device = LabJack.CreateLabJack(serial_no)
# device.Connect(serial_no)
# time.sleep(0.25)  # wait statements are important to allow settings to be sent to the device


# device_info = device.GetDeviceInfo()
# print(device_info.Description)
# device.StartPolling(250)


# time.sleep(1)
# device.Disconnect()
# print('Test connection finished')

def from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb  


def generate_random_image():
    fontpath = 'OpenSans-Regular.ttf'
    font11 = ImageFont.truetype(fontpath, 34)

    frame = np.zeros((448, 800, 3), np.uint8)

    words = 'Jamnik – jedna z ras psów pochodząca z Niemiec. Niemiecka nazwa jamnika Dachshund oznacza w dosłownym tłumaczeniu "borsuczy pies", etymologia nazwy związana jest z jego zbliżoną do borsuków budową oraz wykorzystywaniem tej rasy do polowania na zwierzęta ryjące nory.'.split(' ')
    # words = ['jamnik', 'chirality', 'nanoparticles', 'gold', 'liquid crystal', 'camera error', 'impact factor',
            # 'laser', 'samples', 'work', 'nematic', 'helical', 'danger', 'run', 'thorlabs', 'microscope', 'science',
            # 'strange', 'temperature']
    
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    for i in range(0, 40):
        clr = int(60 + 195*np.random.random())
        draw.text((int(700*np.random.random()), int(400*np.random.random())), np.random.choice(words), font = font11, fill = (clr, clr, clr))
    return np.array(img_pil)


def read_script(fname, arguments = []):
    def clean_cmd(cmd):
        if cmd.find('#') >= 0: cmd = cmd[:cmd.find('#')]
        clean = cmd[:cmd.find('(')]
        args = cmd[cmd.find('(')+1:cmd.find(')')]
#         args = args.replace(',', ' ')
#         while args.find(' ') != -1:
#             args = args.replace(' ', '')
        args = args.split(',')
        for i in range(0, len(args)):
            args[i] = args[i].replace(';', ',')
            if args[i].find('\'') == -1:
                args[i] = args[i].replace(' ', '')
            else:
                args[i] = args[i].replace('\'', '')
        return [clean, args]
    
    possibleLocations = [fname, wd + fname, wd + 'scripts/' + fname, wd + 'scripts/base/' + fname]
    path = ''
    for loc in possibleLocations:
        print(loc + '.txt')
        if os.path.isfile(loc + '.txt') == True:
            path = loc + '.txt'
            break
    
    if path == '':
        print('Script %s not found'%fname)
        return -1
    
    with open(path, 'r') as f:
        cnt = f.read()
    
    for i, arg in enumerate(arguments):
        cnt = cnt.replace('arg' + str(i).zfill(2), arg)
    
    if cnt.find('arg') >= 0:
        print('Not enough arguments for script %s'%fname)
        return -2
    
    cnt = cnt.split('\n')
    
    answer = []
    for cmd in cnt:
        if cmd.find('(') >= 0:
            ans = clean_cmd(cmd)
            if ans[0] != '':
                answer.append(ans)
    return answer


def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result
    

class CamReader(Process):
#     override the constructor
    def __init__(self, event, sharr, config_arr, camparam_arr):
        # execute the base constructor
        Process.__init__(self)
        # initialize integer attribute
        self.event = event
        self.config_arr = config_arr
        self.camparam_arr = camparam_arr
        self.sharr = sharr
        self.gain = self.config_arr[0]
        self.expo = self.config_arr[1]

        self.frame = np.zeros((448, 800, 3), np.uint8)
        
        # self.run_camera()
        
        # fontpath = 'OpenSans-Regular.ttf'
        # self.font11 = ImageFont.truetype(fontpath, 34)
        # self.frame_is_new = True
        
        # self.change_gain(0)
        # time.sleep(1)
        # self.change_expo(-8)        
        
    def run(self):
        self.cap = cv2.VideoCapture(0)
        while(True):
            ret, self.frame = self.cap.read()
            time.sleep(0.01)
            # final_img = np.zeros((448, 800, 4), dtype = np.ubyte)
            self.sharr_np = np.frombuffer(self.sharr, dtype=np.uint8).reshape(*camRes)
            np.copyto(self.sharr_np, self.frame)
            
            
            if self.gain != self.config_arr[0]:
                parameter = cv2.CAP_PROP_GAIN
                self.cap.set(parameter, self.gain)
                print('new camera gain %s'%str(self.gain))
                self.gain = self.config_arr[0]
                
                
            if self.expo != self.config_arr[1]:
                parameter = cv2.CAP_PROP_EXPOSURE
                self.cap.set(parameter, expo)
                print('new camera expo %s'%str(expo))
                
            
            # exit event
            if self.event.is_set():
                print('Cam reader: Call to exit camera received!')
                # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
                self.cap.release()
                break


class M30(Process):
    def __init__(self, m30_event, m30_param):
        Process.__init__(self)
        self.m30_event = m30_event
        self.m30_param = m30_param
        self.curAcc = 5
        self.curVel = 2
        
    def run(self):
        DeviceManagerCLI.BuildDeviceList()
        # create new device
        serial_no = "101334424"  # Replace this line with your device's serial number

        device = BenchtopDCServo.CreateBenchtopDCServo(serial_no)

        # Connect, begin polling, and enable
        device.Connect(serial_no)
        time.sleep(0.25)  # wait statements are important to allow settings to be sent to the device

        # Get Device Information and display description
        device_info = device.GetDeviceInfo()
        print(device_info.Description)

        # Get the channel for the device
        x_channel = device.GetChannel(1)  # Returns a benchtop channel object
        y_channel = device.GetChannel(2)

        # Start Polling and enable channel
        x_channel.StartPolling(250)
        y_channel.StartPolling(250)
        time.sleep(0.25)
        x_channel.EnableDevice()
        y_channel.EnableDevice()
        time.sleep(0.25)

        # Check that the settings are initialised, else error.
        if not x_channel.IsSettingsInitialized() or not y_channel.IsSettingsInitialized():
            x_channel.WaitForSettingsInitialized(10000)  # 10 second timeout
            y_channel.WaitForSettingsInitialized(10000)
            assert device.IsSettingsInitialized() is True

        # Load the motor configuration on the channel
        x_config = x_channel.LoadMotorConfiguration(x_channel.DeviceID)
        y_config = y_channel.LoadMotorConfiguration(y_channel.DeviceID)

        # Read in the device settings
        dev_settings = x_channel.MotorDeviceSettings

        # Get the Homing Params
        x_home_params = x_channel.GetHomingParams()
        y_home_params = y_channel.GetHomingParams()

        x_home_params.Velocity = Decimal(2.0)
        y_home_params.Velocity = Decimal(2.0)

        x_channel.SetHomingParams(x_home_params)
        y_channel.SetHomingParams(y_home_params)
        
        
        while(True):
            time.sleep(0.1)
            
            # write current position to m30_param array
            posX = round(float(x_channel.DevicePosition.ToString().replace(',', '.')), 3)  # warning: this uses , as decimal separator
            posY = round(float(y_channel.DevicePosition.ToString().replace(',', '.')), 3)
            self.m30_param[0] = posX
            self.m30_param[1] = posY
            
            
            # change vel and acc parameters      
            vel, acc = round(self.m30_param[4], 3), round(self.m30_param[5], 3)
            if vel != self.curVel or acc != self.curAcc:                
                x_vel_params = x_channel.GetVelocityParams()
                y_vel_params = y_channel.GetVelocityParams()
                x_vel_params.Acceleration = Decimal(acc)
                x_vel_params.MaxVelocity = Decimal(vel)
                y_vel_params.Acceleration = Decimal(acc)
                y_vel_params.MaxVelocity = Decimal(vel)
                x_channel.SetVelocityParams(x_vel_params)
                y_channel.SetVelocityParams(y_vel_params)
                self.curVel = vel
                self.curAcc = acc
            
            
            
            # move if necessary
            setX, setY = round(self.m30_param[2], 3), round(self.m30_param[3], 3)
            
            if setX != posX:
                if setX > 15: setX = 15
                x_channel.MoveTo(Decimal(setX), 60000)
            if setY != posY:
                if setY > 15: setY = 15
                y_channel.MoveTo(Decimal(setY), 60000)
                
                
            if self.m30_param[6] == 1:
                self.m30_param[6] = 0
                x_channel.Home(60000)  # 60 second timeout
                y_channel.Home(60000)
                
            # exit event
            if self.m30_event.is_set():
                print('M30: Call to exit m30 received!')
                y_channel.StopPolling()
                x_channel.StopPolling()
                device.Disconnect()
                break
            

class LabJackZ(Process):
    def __init__(self, lj_event, lj_param):
        Process.__init__(self)
        self.lj_event = lj_event
        self.lj_param = lj_param
    
    
    def run(self):
        DeviceManagerCLI.BuildDeviceList()
        serial_no = "49337314"
        device = LabJack.CreateLabJack(serial_no)

        # Connect, begin polling, and enable
        device.Connect(serial_no)
        time.sleep(0.25)  # wait statements are important to allow settings to be sent to the device

        # Get Device Information and display description
        device_info = device.GetDeviceInfo()
        print(device_info)
        
        device.StartPolling(250)
        device.EnableDevice()
        motorConfiguration = device.LoadMotorConfiguration(serial_no)
        
        workDone = device.InitializeWaitHandler();
        time.sleep(0.1)
        devPos = round(float(device.Position.ToString().replace(',', '.')), 3)
        self.lj_param[1] = devPos
        
        while(True):
                time.sleep(0.1)
                devPos = round(float(device.Position.ToString().replace(',', '.')), 3)
                setPos = round(self.lj_param[1], 3)
                self.lj_param[0] = devPos
                
                try:
                    if setPos != float(device.TargetPosition.ToString().replace(',', '.')):
                        if device.Status.IsInMotion == False:
                            time.sleep(0.1)
                            workDone = device.InitializeWaitHandler()
                            device.MoveTo(Decimal(setPos), workDone)                
                except AssertionError as error:
                    print(error)
                    print('LabJack: error, disconecting!')
                    device.StopPolling()
                    device.Disconnect()
                    break
                
                if self.lj_param[2] == 1:
                    self.lj_param[2] = 0
                    workDone = device.InitializeWaitHandler();
                    device.Home(workDone);
        
                
                # exit event
                if self.lj_event.is_set():
                    print('LabJack: Call to exit LJ received!')
                    device.StopPolling()
                    device.Disconnect()
                    break


class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master
        self.master.protocol("WM_DELETE_WINDOW", self.exit)
        
        
        self.camera_gain = 0
        
        # camera reader
        self.event = Event()
        self.sharr = RawArray('c', 448*800*3)  # holds camera frame
        self.config_arr = RawArray('i', (6, 8, -1))  # hold the camera configuration (gain, expo)
        self.camparam_arr = RawArray('i', (0, 0, 0))  # hold the camera configuration (gain, expo, invert_colors)
        self.cam_reader = CamReader(self.event, self.sharr, self.config_arr, self.camparam_arr)
        self.cam_reader.start()
        self.video_writer = None
        self.recorder_counter = 0
        self.recorder_lapse = 1
        # self.cap = cv2.VideoCapture(0)
        
        
        # XY stage control
        self.m30_event = Event()
        # param array: curX (read), curY (read), setX, setY (write), vel, acc, home
        self.m30_param = RawArray(c_float, (0, 0, 0, 0, 2, 5, 0))
        self.m30 = None
        
        # LabJackZ
        self.lj_event = Event()
        self.lj_param = RawArray(c_float, (0, 0, 0))  # actual position, set position, should_go_home
        self.lj = None
        
        
        # projektor        
        self.projector_window = None
        self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
        
        # devices
        self.rigol = None
        self.laserduty = 1.5
        self.laserstate = 'OFF'
        self.laserstate2 = 'OFF'
        self.tc300_temp = 0
        self.tc300_target = 0
        self.tc300_hdl = -1
        self.aux = None
        self.aux_state = {}
        for i in range(2, 53):
            self.aux_state[i] = 0
        
        # script exec
        self.scripts = []  # holds lists of commands to be executed
        self.lines = []  # keeps track of commands execution        
        self.ctime = 0
        
        self.console_box = None
        self.init_window()
        self.main_loop()
        
        
    def init_window(self):
        col = 0
        row = 0

        # changing the title of our master widget      
        self.master.title("Mikroskop 2.0")
        self.master.bind('<Escape>', self.esc_key_btn)
        
        # menu
        menu = Menu(root)
        self.master.config(menu=menu)
        
        
        fileMenu = Menu(menu, tearoff=False)
        menu.add_cascade(label="Control", menu=fileMenu)
        fileMenu.add_command(label="Test")
        fileMenu.add_command(label = 'Load image', command = self.fmenu_load_img)
        # fileMenu.add_command(label = 'Load script', command = self.fmenu_load_script)
        fileMenu.add_command(label = 'Execute script', command = self.fmenu_exec_script)
       
        
        
        ## FIRST COLUMN
        frame = Frame(root)
        frame.grid(row=0, column=0, padx = padd)
        
        # MAIN CAMERA FRAME
        cam_frame = Frame(frame)
        cam_frame.pack(fill = Y, padx = padd)
        self.create_camera_frame(cam_frame)
        
        ## SECOND COLUMN        
        frame = Frame(root)
        frame.grid(row=0, column=1, padx = padd)
                
        # RIGOL FRAME
        rigol_frame = Frame(frame)
        rigol_frame.pack(fill = Y, padx = padd)
        self.create_rigol_frame(rigol_frame)        
        
        # M30 FRAME
        stage_frame = Frame(frame)
        stage_frame.pack(fill = Y, padx = padd)
        self.create_m30_frame(stage_frame)   
        
        # LabJackZ FRAME
        lj_frame = Frame(frame)
        lj_frame.pack(fill = Y, padx = padd)
        self.create_lj_frame(lj_frame)     
        
        # THERMAL FRAME
        thermal_frame = Frame(frame)
        thermal_frame.pack(fill = Y, padx = padd)
        self.create_thermal_frame(thermal_frame)
        
        # AUX FRAME
        aux_frame = Frame(frame)
        aux_frame.pack(fill = Y, padx = padd)
        self.create_aux_frame(aux_frame)
        
        
        ## THIRD COLUMN  
        frame = Frame(root)
        frame.grid(row=0, column=2, padx = padd)
        
        # console
        console_frame = Frame(frame)
        console_frame.pack(fill = Y, padx = padd)
        self.create_console_frame(console_frame)
        
        # PROJECTOR FRAME
        proj_frame = Frame(frame)
        proj_frame.pack(fill = Y, padx = padd)
        self.create_projector_frame(proj_frame)
    
    
    def add_padding(self, frame):
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.labSpace = Label(cur_frame, text = '')
        self.labSpace.pack(side = LEFT, pady = 3)
    
    
    def create_lj_frame(self, frame):
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.lab = Label(cur_frame, text = 'LabJack CONTROL')
        self.lab.config(font=subsystem_name_font)
        self.lab.pack(side =  LEFT)
        
        
        # position label       
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)         
        self.labLjPos = Label(cur_frame, text = 'Z = %2.2f'%(0.0), fg = laser_off_color)
        self.labLjPos.config(font=('Segoe UI', 13))
        self.labLjPos.pack(side =  LEFT)
        
        
        # connection frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.btn_lj_connect = Button(cur_frame, text = 'Connect to LabJack', command = self.connect_lj)
        self.btn_lj_connect.pack(side = LEFT)
        
        self.label_lj_status = Label(cur_frame, text = 'LabJack status: ')
        self.label_lj_status.pack(side =  LEFT)
        
        self.label_lj_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_lj_status2.pack(side =  LEFT)
        
        
        # Controls frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
           
        # move controls
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        # set absolute val     
        self.lj_var = StringVar()
        self.lj_var.set('0')
        self.lj_entry = Entry(cur_frame, width = 15, textvariable = self.lj_var)
        self.lj_entry.pack(side = LEFT, fill = X)
        
        self.set_lj_btn = Button(cur_frame, text = 'Set', command = self.btn_lj_set)
        self.set_lj_btn.pack(side = LEFT)
        self.set_lj_btn = Button(cur_frame, text = 'Home', command = self.btn_lj_home)
        self.set_lj_btn.pack(side = LEFT)
        
        # control buttons frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        # position label
        self.labLJPos = Label(cur_frame, text = 'Move relative:')
        self.labLJPos.pack(side = LEFT)
        
        self.btn = Button(cur_frame, text = '-1', command = lambda: self.btn_lj_moverel(-1))
        self.btn.pack(side = LEFT)
        self.btn = Button(cur_frame, text = '-.25', command = lambda: self.btn_lj_moverel(-0.25))
        self.btn.pack(side = LEFT)
        self.btn = Button(cur_frame, text = '-.05', command = lambda: self.btn_lj_moverel(-0.05))
        self.btn.pack(side = LEFT)
        self.btn = Button(cur_frame, text = '-.005', command = lambda: self.btn_lj_moverel(-0.005))
        self.btn.pack(side = LEFT)
        self.btn = Button(cur_frame, text = '+.005', command = lambda: self.btn_lj_moverel(0.005))
        self.btn.pack(side = LEFT)
        self.btn = Button(cur_frame, text = '+.05', command = lambda: self.btn_lj_moverel(0.05))
        self.btn.pack(side = LEFT)
        self.btn = Button(cur_frame, text = '+0.25', command = lambda: self.btn_lj_moverel(0.25))
        self.btn.pack(side = LEFT)
        self.btn = Button(cur_frame, text = '+1', command = lambda: self.btn_lj_moverel(1))
        self.btn.pack(side = LEFT)
        
        self.add_padding(frame)
    
    
    def create_aux_frame(self, frame):
        # grbl frame name
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.lab = Label(cur_frame, text = 'AUX ARDUINO')
        self.lab.config(font=subsystem_name_font)
        self.lab.pack(side =  LEFT)
        
        # serial connection frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.aux_con_var = StringVar(self.master)
        self.aux_con_var.set('COM6') # default value
        self.aux_com_menu = OptionMenu(cur_frame, self.aux_con_var, *serial_ports(), command = self.aux_connect)
        self.aux_com_menu.pack(side = LEFT)
        
        self.label_ell_status = Label(cur_frame, text = 'AUX status: ')
        self.label_ell_status.pack(side =  LEFT)
        
        self.lab_aux_con = Label(cur_frame, text = 'unknown', bg='gray')
        self.lab_aux_con.pack(side =  LEFT)   
        
        # current temp
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.auxButtons = {}
        for i in range(2, 15):
            # btn = Button(cur_frame, text = str(i), command = lambda: self.aux_btn(i))
            # btn.pack(side = LEFT)
            def aux_action(x = i):
                return self.aux_btn(x)
            btn = Button(cur_frame, text = str(i), command = aux_action)
            self.auxButtons[i] = btn
            self.auxButtons[i].pack(side = LEFT)
            
        self.add_padding(frame)
    
    
    def create_thermal_frame(self, frame):
        # grbl frame name
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.lab = Label(cur_frame, text = 'THERMAL CONTROL')
        self.lab.config(font=subsystem_name_font)
        self.lab.pack(side =  LEFT)
        
        # current temp
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.lab_tempcur = Label(cur_frame, text = 'T = %3.2f °C'%self.tc300_temp)
        self.lab_tempcur.config(font=('Segoe UI', 13))
        self.lab_tempcur.pack(side =  LEFT)
        
        
        # set temp frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
    
        lab = Label(cur_frame, text = 'Set temperature:')
        lab.config(font=('Segoe UI', 9))
        lab.pack(side =  LEFT)   
        self.set_temp_var = StringVar()
        self.set_temp_var.set('40')
        self.set_temp_entry = Entry(cur_frame, width = 15, textvariable = self.set_temp_var)
        self.set_temp_entry.pack(side = LEFT, fill = X)
        
        self.set_temp_btn = Button(cur_frame, text = 'Set', command = self.set_temperature)
        self.set_temp_btn.pack(side = LEFT)
        
        # ramp frame
        # cur_frame = Frame(frame)
        # cur_frame.pack(fill = Y)
        
        # lab = Label(cur_frame, text = 'Ramp rate:')
        # lab.config(font=('Segoe UI', 9))
        # lab.pack(side =  LEFT)   
        # self.set_ramp_var = StringVar()
        # self.set_ramp_var.set('5')
        # self.set_ramp_entry = Entry(cur_frame, width = 5, textvariable = self.set_ramp_var)
        # self.set_ramp_entry.pack(side = LEFT, fill = X)
        
        # lab = Label(cur_frame, text = 'target:')
        # lab.config(font=('Segoe UI', 9))
        # lab.pack(side =  LEFT)   
        # self.set_ramp_target_var = StringVar()
        # self.set_ramp_target_var.set('40')
        # self.set_ramp_target_entry = Entry(cur_frame, width = 5, textvariable = self.set_ramp_target_var)
        # self.set_ramp_target_entry.pack(side = LEFT, fill = X)
        
        # self.set_ramp_btn = Button(cur_frame, text = 'Ramp!', command = self.ramp_clicked)
        # self.set_ramp_btn.pack(side = LEFT)
        
        # connection frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        # grbl_com_var = StringVar(self.master)
        # grbl_com_var.set('COM5') # default value
        # self.grbl_com_menu = OptionMenu(cur_frame, grbl_com_var, *serial_ports(), command = self.thrm_refresh)
        # self.grbl_com_menu.pack(side = LEFT)
        
        self.btn_tc300_connect = Button(cur_frame, text = 'Connect', command = self.tc300_connect)
        self.btn_tc300_connect.pack(side = LEFT)
        
        self.label_thrm_status = Label(cur_frame, text = 'THERMAL status: ')
        self.label_thrm_status.pack(side =  LEFT)
        
        self.label_thrm_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_thrm_status2.pack(side =  LEFT) 
        
        self.add_padding(frame)
        
        
    def create_console_frame(self, frame):
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.lab = Label(cur_frame, text = 'CONSOLE')
        self.lab.config(font = subsystem_name_font)
        self.lab.pack(side = LEFT)
        
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.console_box = Text(cur_frame, height = 35, width = 80, background = '#00112b', foreground = '#c5d3e8', font=("Consolas", 8))
        self.console_box.pack(fill = Y, padx = 0, side = LEFT)

        scroll_bar = Scrollbar(cur_frame, command=self.console_box.yview)
        scroll_bar.pack(side=RIGHT, expand=True, fill = Y)
        self.console_box.configure(yscrollcommand=scroll_bar.set)

        long_text = """Welcome to mikroskop control!
        """        
        self.console_box.insert(END, long_text)
        
        # input field
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.lab = Label(cur_frame, text = 'Input command:')
        self.lab.pack(side =  LEFT)
        
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)   
        self.conInpVar = StringVar()
        self.conInpVar.set('')
        self.console_input = Entry(cur_frame, textvariable = self.conInpVar, width = 80, background = '#00112b', foreground = '#c5d3e8', font=("Consolas", 8))
        self.console_input.pack(fill = Y, padx = 0, side = LEFT)
        self.console_input.bind('<Return>', self.con_input)
        
        self.add_padding(frame)
        
        
    def create_projector_frame(self, frame):
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.labLaser = Label(cur_frame, text = 'PROJECTOR CONTROL')
        self.labLaser.config(font = subsystem_name_font)
        self.labLaser.pack(side =  LEFT)
        
        cur_frame = Frame(frame)
        # proj_frame1.grid(row=1, column=0, padx = padd)
        cur_frame.pack(fill = Y)
        
        self.init_proj_win_btn = Button(cur_frame, text = 'Init window', command = self.initiate_projector_window)
        self.init_proj_win_btn.pack(side = LEFT)
        
        self.act_proj_win_btn = Button(cur_frame, text = 'Activate window', command = self.activate_projector_window)
        self.act_proj_win_btn.pack(side = LEFT)
        
        self.act_proj_win_btn = Button(cur_frame, text = 'Close window', command = self.close_projector_window)
        self.act_proj_win_btn.pack(side = LEFT)
        
        cur_frame = Frame(frame)
        # canvas_frame.grid(row=2, column=0, padx = padd)
        cur_frame.pack(fill = Y)
        
        self.proj_mirror_canvas = Canvas(cur_frame, width=256, height=192, bg='black')
        self.proj_mirror_canvas.pack(side = LEFT)
        
        self.add_padding(frame)
        
        
    def create_rigol_frame(self, frame):
        # rigol frame name
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        self.labLaser = Label(cur_frame, text = 'LASER CONTROL')
        self.labLaser.config(font = subsystem_name_font)
        self.labLaser.pack(side =  LEFT)
        
        # laser status
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab_laser = Label(cur_frame, text = 'DUTY = %2.2f %%, CH1:LASER IS %s'%(0.0, 'OFF'), fg = laser_off_color)
        self.lab_laser.config(font=('Segoe UI', 13))
        self.lab_laser.pack(side =  LEFT)        
        
        # laser duty control
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.label = Label(cur_frame, text = 'Set duty cycle CH1:')
        self.label.pack(side =  LEFT)
        
        self.laserduty_var = StringVar()
        self.laserduty_var.set('1')
        self.laserduty_entry = Entry(cur_frame, width = 15, textvariable = self.laserduty_var)
        self.laserduty_entry.pack(side = LEFT, fill = X)
        
        self.set_temp_btn = Button(cur_frame, text = 'Set', command = self.btn_laserduty)
        self.set_temp_btn.pack(side = LEFT)
        
        # cur_frame = Frame(frame)
        # cur_frame.pack(fill = Y)
        
        self.btnLaserState = Button(cur_frame, text = 'CH1:laser', command = self.btn_laser_switch, width = 10)
        self.btnLaserState.pack(side =  LEFT)
        
        # laser2 status
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab_laser2 = Label(cur_frame, text = 'DUTY = %2.2f %%, CH2:AUX IS %s'%(0.0, 'OFF'), fg = laser_off_color)
        self.lab_laser2.config(font=('Segoe UI', 13))
        self.lab_laser2.pack(side =  LEFT)  
        
        # ch2 duty control
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.label = Label(cur_frame, text = 'Set duty cycle CH2:')
        self.label.pack(side =  LEFT)
        
        self.laserduty_var2 = StringVar()
        self.laserduty_var2.set('1')
        self.laserduty_entry = Entry(cur_frame, width = 15, textvariable = self.laserduty_var2)
        self.laserduty_entry.pack(side = LEFT, fill = X)
        
        self.set_temp_btn = Button(cur_frame, text = 'Set', command = self.btn_laserduty2)
        self.set_temp_btn.pack(side = LEFT)
        
        # cur_frame = Frame(frame)
        # cur_frame.pack(fill = Y)
        
        self.btnLaserState2 = Button(cur_frame, text = 'CH2:aux', command = self.btn_laser_switch2, width = 10)
        self.btnLaserState2.pack(side =  LEFT)
        
        # connection to rigol
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.btn_rigol_connect = Button(cur_frame, text = 'Connect to Rigol', command = self.connect_rigol)
        self.btn_rigol_connect.pack(side = LEFT)
        
        self.label_rigol_status = Label(cur_frame, text = 'RIGOL status: ')
        self.label_rigol_status.pack(side =  LEFT)
        
        self.label_rigol_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_rigol_status2.pack(side =  LEFT) 
        
        self.add_padding(frame)
        
        
    def create_camera_frame(self, frame):
        cframe = Frame(frame)
        cframe.grid(row=0, column=0, padx = padd)
        self.canvas = Canvas(cframe, width=800, height=600, bg='black')
        self.canvas.pack(fill = Y, padx = padd)
        
        # camera controls frame
        cframe = Frame(frame)
        cframe.grid(row=1, column=0, padx = padd)
        self.label = Label(cframe, text = 'Gain: ')
        self.label.pack(side =  LEFT)
        
        self.btnGdg = Button(cframe, text = '-5', command = lambda: self.camera_deltagain(-5))
        self.btnGdg.pack(side = LEFT)
        self.btnGdg = Button(cframe, text = '-1', command = lambda: self.camera_deltagain(-1))
        self.btnGdg.pack(side = LEFT)
        
        self.labelGain = Label(cframe, text = '%d'%self.camera_gain)
        self.labelGain.pack(side =  LEFT)
        
        self.btnGdg = Button(cframe, text = '+1', command = lambda: self.camera_deltagain(1))
        self.btnGdg.pack(side = LEFT)
        self.btnGdg = Button(cframe, text = '+5', command = lambda: self.camera_deltagain(5))
        self.btnGdg.pack(side = LEFT)
        
        self.label_save = Label(cframe, text = 'Sample name: ')
        self.label_save.pack(side =  LEFT)
        
        self.sname_var = StringVar()
        self.nameEntered = Entry(cframe, width = 15, textvariable = self.sname_var)
        self.nameEntered.pack(side = LEFT)
        
        # self.buttonRecord = Button(cframe, text = 'Record video', command = self.record_video)
        # self.buttonRecord.pack(side = LEFT)
        
        self.buttonSave = Button(cframe, text = 'Save', command = self.save_img)
        self.buttonSave.pack(side = LEFT)
        
        
    def create_m30_frame(self, frame):
        cframe = Frame(frame)
        cframe.pack(fill = Y)
        self.labLaser = Label(cframe, text = 'M30 STAGE CONTROL')
        self.labLaser.config(font = subsystem_name_font)
        self.labLaser.pack(side =  LEFT)
        
        
        # position label        
        cframe = Frame(frame)
        cframe.pack(fill = Y)        
        self.labM30PosX = Label(cframe, text = 'X = %2.2f'%(0.0), fg = laser_off_color)
        self.labM30PosX.config(font=('Segoe UI', 13))
        self.labM30PosX.pack(side =  LEFT)
        
        cframe = Frame(frame)
        cframe.pack(fill = Y)        
        self.labM30PosY = Label(cframe, text = 'Y = %2.2f'%(0.0), fg = laser_off_color)
        self.labM30PosY.config(font=('Segoe UI', 13))
        self.labM30PosY.pack(side =  LEFT)
        
        
        # connection frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        self.btn_m30_connect = Button(cur_frame, text = 'Connect to M30', command = self.connect_m30)
        self.btn_m30_connect.pack(side = LEFT)
        
        self.label_m30_status = Label(cur_frame, text = 'M30 status: ')
        self.label_m30_status.pack(side =  LEFT)
        
        self.label_m30_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_m30_status2.pack(side =  LEFT)
        
        
        self.m30_stp = StringVar()
        self.m30_stp.set('1.0')
        self.m30_vel = StringVar()
        self.m30_vel.set('2.0')
        self.m30_acc = StringVar()
        self.m30_acc.set('5.0')
                
        
        cframe = Frame(frame)
        cframe.pack(fill = Y)
        
        bframe = Frame(cframe)
        bframe.grid(row=0, column=0)
        btn = Button(bframe, text = '\\')
        btn.pack(side = LEFT)
        btn = Button(bframe, text = 'U', command = lambda: self.m30_move(0, 1))
        btn.pack(side = LEFT)
        btn = Button(bframe, text = '/')
        btn.pack(side = LEFT)
        
        bframe = Frame(cframe)
        bframe.grid(row=1, column=0)
        btn = Button(bframe, text = 'L', command = lambda: self.m30_move(-1, 0))
        btn.pack(side = LEFT)
        btn = Button(bframe, text = 'H', command = self.m30_home)
        btn.pack(side = LEFT)
        btn = Button(bframe, text = 'R', command = lambda: self.m30_move(1, 0))
        btn.pack(side = LEFT)
        
        bframe = Frame(cframe)
        bframe.grid(row=2, column=0)
        btn = Button(bframe, text = '/')
        btn.pack(side = LEFT)
        btn = Button(bframe, text = 'D', command = lambda: self.m30_move(0, -1))
        btn.pack(side = LEFT)
        btn = Button(bframe, text = '\\')
        btn.pack(side = LEFT)
        
        
        # cframe = Frame(frame)
        # cframe.pack(side = LEFT, fill = Y)
        
        l = Label(cframe, text = 'Stp:')
        l.grid(row = 0, column = 1)
        self.m30_stp_entry = Entry(cframe, width = 6, textvariable = self.m30_stp)
        self.m30_stp_entry.grid(row = 0, column = 2)
        
        l = Label(cframe, text = 'Vel:')
        l.grid(row = 1, column = 1)
        self.m30_vel_entry = Entry(cframe, width = 6, textvariable = self.m30_vel)
        self.m30_vel_entry.grid(row = 1, column = 2)
        
        l = Label(cframe, text = 'Acc:')
        l.grid(row = 2, column = 1)
        self.m30_acc_entry = Entry(cframe, width = 6, textvariable = self.m30_acc)
        self.m30_acc_entry.grid(row = 2, column = 2)
        
        self.add_padding(frame)
    
    
    def log(self, text, should_print = True, color = None, bold = None):
        numlines = int(self.console_box.index('end - 1 line').split('.')[0])
        
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M:%S")
        timetext = current_time + ' ' + text
        
        self.console_box['state'] = 'normal'
        if numlines >= 256:
            self.console_box.delete(1.0, 2.0)
        if self.console_box.index('end-1c')!='1.0':
            self.console_box.insert('end', '\n')
        self.console_box.insert('end', timetext)
        self.console_box.see('%3.1f'%(numlines+1))
        self.console_box['state'] = 'disabled'
        
        if should_print == True:
            print(text)
    
    
    def fmenu_exec_script(self):
        filename = filedialog.askopenfilename(initialdir="C:/Users/LAB218/Documents/mikroskop2/scripts")
        if filename == '': return 1
        fname = filename.replace('.txt', '')
        fname = fname[fname.rfind('/'):]
        script = read_script(fname)
        if type(script) == int:
            if script == -1: self.log('Script %s not found!'%filename)
            if script == -2: self.log('Not enought arguments for script %s'%filename)
            return 1
        self.scripts.append(script)
        self.lines.append(0)
 
 
    def m30_move(self, xMul, yMul):
        if self.m30 != None:   
            '''xMul, yMul are multipliers: actual move is xMul*step'''
            stp = round(float(self.m30_stp.get()), 3)
            vel = round(float(self.m30_vel.get()), 3)
            acc = round(float(self.m30_acc.get()), 3)
            
            self.m30_param[4] = vel
            self.m30_param[5] = acc
            
            if xMul != 0:
                self.m30_param[2] += xMul*stp
            if yMul != 0:
                self.m30_param[3] += yMul*stp
                
            if self.console_box != None: self.log('M30 moving to X=%2.3f Y=%2.3f'%(self.m30_param[2], self.m30_param[3]))
        else:
            self.log('M30 not connected')
    
    
    def m30_home(self):
        if self.m30 != None:            
            self.m30_param[6] = 1
            self.log('Moving M30 to home position')
        else:
            self.log('M30 not connected')
    
    
    def refresh_m30_label(self):
        posX = self.m30_param[0]
        posY = self.m30_param[1]
        self.labM30PosX.config(text = 'X = %2.3f'%(posX))
        self.labM30PosY.config(text = 'Y = %2.3f'%(posY))
    

    def con_input(self, event):
        fname = 'console_input'
        inputCmd = self.conInpVar.get()
        with open('scripts/base/console_input.txt', 'w') as f:
            f.write(inputCmd)
            
        script = read_script(fname)
        if type(script) == int:
            if script == -1: self.log('Script %s not found!'%fname)
            if script == -2: self.log('Not enought arguments for script %s'%fname)
            return 1
        
        self.scripts.append(script)
        self.lines.append(0)
        self.conInpVar.set('')
    
        
    def refresh_lj_label(self):
        posZ = self.lj_param[0]
        self.labLjPos.config(text = 'Z = %2.3f'%(posZ))
    

    def btn_laser_switch(self):
        if self.rigol != None:
            # switch laser state
            if self.laserstate == 'OFF': to_set = 'ON'
            if self.laserstate == 'ON': to_set = 'OFF'
            self.laserstate = to_set
            
            # update button state
            if self.laserstate == 'ON':
                self.btnLaserState.config(relief=SUNKEN, bg = laser_on_color)
                self.rigol.write(':OUTP1 ON')
                clr = laser_on_color
                self.log('CH1:laser ON!')
            if self.laserstate == 'OFF':
                self.btnLaserState.config(relief=RAISED, bg = '#f0f0f0')
                self.rigol.write(':OUTP1 OFF')
                clr = laser_off_color
                self.log('CH1:laser OFF!')
                
            # update label
            self.lab_laser.config(text = 'DUTY = %2.2f %%, CH1:LASER IS %s'%(self.laserduty, self.laserstate), fg=clr)
        else:
            messagebox.showwarning(title='Laser', message='Connection to Rigol is not established. To power up laser, connect to Rigol first')
            
          
    def btn_laser_switch2(self):
        if self.rigol != None:
            # switch laser state
            if self.laserstate2 == 'OFF': to_set = 'ON'
            if self.laserstate2 == 'ON': to_set = 'OFF'
            self.laserstate2 = to_set
            
            # update button state
            if self.laserstate2 == 'ON':
                self.btnLaserState2.config(relief=SUNKEN, bg = laser_on_color)
                self.rigol.write(':OUTP2 ON')
                clr = laser_on_color
                self.log('CH2:aux ON!')
            if self.laserstate2 == 'OFF':   
                self.btnLaserState2.config(relief=RAISED, bg = '#f0f0f0')
                self.rigol.write(':OUTP2 OFF')
                clr = laser_off_color
                self.log('CH2:aux OFF!')
                
            # update label
            self.lab_laser2.config(text = 'DUTY = %2.2f %%, CH2:AUX IS %s'%(self.laserduty2, self.laserstate2), fg=clr)
        else:
            messagebox.showwarning(title='Laser', message='Connection to Rigol is not established. To power up laser, connect to Rigol first')


    def btn_laserduty(self):
        self.laserduty = float(self.laserduty_var.get())
        self.rigol_set_laserduty(self.laserduty)
        
        
    def btn_laserduty2(self):
        self.laserduty2 = float(self.laserduty_var2.get())
        self.rigol_set_laserduty2(self.laserduty2)
    
    
    def save_img(self):
        fname = self.sname_var.get()
        if fname == '': fname = 'unnamed'
        all_files = os.listdir(wd + '/saved_images/')
        saved = 0
        num = 0
        while saved == 0:
            if fname + '_%s.png'%str(num).zfill(2) in all_files:
                num += 1
            else:
                img = Image.fromarray(self.cv2image)
                img.save(wd + 'saved_images/' + fname + '_%s.png'%str(num).zfill(2))
                self.log(fname + '_%s.png saved!'%str(num).zfill(2))
                saved =1
        
        
    def record_video(self):
        if self.video_writer == None:
            # get the correct filename
            fname = self.sname_var.get()
            if fname == '': fname = 'unnamed'
            all_files = os.listdir(wd + '/saved_images/')
            saved = 0
            num = 0
            while saved == 0:
                if fname + '_%s.avi'%str(num).zfill(2) in all_files:
                    num += 1
                else:
                    out_path = wd + 'saved_images/' + fname + '_%s.avi'%str(num).zfill(2)
                    saved =1
                    
            # frame_y, frame_x = self.cam_reader.get_frame_shape()[:2]
            frame_y, frame_x = self.cv2image.shape[:2]
            self.video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_x, frame_y))
            self.buttonRecord.configure(relief = SUNKEN, bg=from_rgb((245, 150, 150)))
            
        # if it is already recording, stop it
        else:
            self.video_writer.release()
            self.video_writer = None
            # self.buttonRecord.configure(text = 'Record video')
            self.buttonRecord.configure(relief = RAISED, bg=from_rgb((240, 240, 237)))
    
    
    def set_m30_params(self, vel, acc):
        self.log('Setting M30 vel = %2.2f and acc = %2.2f'%(vel, acc))
        self.m30_vel.set(vel)
        self.m30_acc.set(acc)
        self.m30_param[4] = vel
        self.m30_param[5] = acc
    
    
    def main_loop(self):
        self.frame = np.frombuffer(self.sharr, dtype=np.uint8).reshape(*camRes)
        
        # self.frame = generate_random_image()
        # ret, self.frame = self.cap.read()
        
        self.cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) #A
        
        img = Image.fromarray(self.cv2image)
        self.imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(400, 300, image=self.imgtk, anchor=CENTER)
        
        
        # m30 label
        self.refresh_m30_label()
        self.refresh_lj_label()
        
        # script execution (starts from the rightmost position in self.scripts list
        if len(self.scripts) > 0:
            comNum = self.lines[-1]
            if comNum == len(self.scripts[-1]):
                if len(self.scripts) == 1:
                    self.log('---- main script finished executing!')
                else:
                    scName = self.scripts[-2][self.lines[-2]-1]
                    self.log('-- script %d (%s) finished'%(len(self.scripts), scName))
                del self.lines[-1]
                del self.scripts[-1]
            else:
                cmd = self.scripts[-1][comNum]
                self.lines[-1] += 1
                self.execute_command(cmd)
        else:
            pass
            # self.log('No script to execute')
            
        # record video
        if self.video_writer != None:
            if self.recorder_counter >= self.recorder_lapse:
                self.video_writer.write(cv2.cvtColor(self.cv2image, cv2.COLOR_BGR2RGB))
                # print("show frame ", cv2.cvtColor(self.cv2image, cv2.COLOR_BGR2RGB).shape)
                self.recorder_counter = 0
            else:
                self.recorder_counter += 1
        
        self.master.after(50, self.main_loop)


    def execute_command(self, command):
        # self.log(str(command))
        cmd, args = command[0], command[1]
        
        if cmd[:6] != 'd_wait' and self.ctime == 0:  # printing command before execution is not necessary for some commands
            self.log('---- executing ' + str(cmd) + ', args:' + str(args))
        
        # execution of indirect commands (commands that spawn new commands)
        if cmd[:2] != 'd_':
            script = read_script(command[0], arguments = args)
            if type(script) == int:
                if script == -1: self.log('Script %s not found!'%command[0])
                if script == -2: self.log('Not enought arguments for script %s'%command[0])
                return 1
            self.scripts.append(script)
            self.lines.append(0)
            # _ = [self.log(str(sc)) for sc in self.scripts]
            # self.log(str(self.lines))        
        
        # execution of direct commands (d_)
        if cmd == 'd_move':
            self.m30_param[2] = float(args[0])
            self.m30_param[3] = float(args[1])
            
            # self.m30_param[4] = float(args[2])  # vel
            # self.m30_param[5] = float(args[3])  # acc
            
            self.log('-- M30 moving to X=%2.3f Y=%2.3f'%(self.m30_param[2], self.m30_param[3]))
            
        if cmd == 'd_wait_move':
            if abs(self.m30_param[0] - self.m30_param[2]) > 0.1 or abs(self.m30_param[1] - self.m30_param[3]) > 0.1:
                self.lines[-1] -= 1    
            else:          
                self.log('-- wait for move finished')
            
        if cmd == 'd_load_img':
            self.load_pattern_image(args[0].replace('\'', ''))
            
        if cmd == 'd_laser_duty':
            self.rigol_set_laserduty(float(args[0]))
            
        if cmd == 'd_laser_duty2':
            self.rigol_set_laserduty2(float(args[0]))
            
        if cmd == 'd_laser_switch':
            if args[0] == '1':  # means we want to turn the laser on
                if self.laserstate == 'OFF':
                    self.btn_laser_switch()
                    self.log('-- swithing the laser ON')
                else:
                    self.log('-- laser is already on!')
            if args[0] == '0':  # means we want to turn the laser off
                if self.laserstate == 'ON':
                    self.btn_laser_switch()
                    self.log('-- swithing the laser OFF')
                else:
                    self.log('-- laser is already off!')
                    
        if cmd == 'd_laser_switch2':
            if args[0] == '1':  # means we want to turn the laser on
                if self.laserstate2 == 'OFF':
                    self.btn_laser_switch2()
                    self.log('-- swithing the aux ON')
                else:
                    self.log('-- aux is already on!')
            if args[0] == '0':  # means we want to turn the laser off
                if self.laserstate2 == 'ON':
                    self.btn_laser_switch2()
                    self.log('-- swithing the aux OFF')
                else:
                    self.log('-- aux is already off!')
            
        if cmd == 'd_wait':
            if time.time() < self.ctime:  # move back by one in execution line number for current script if wait time did passed
                self.lines[-1] -= 1             
            if self.ctime == 0:  # initiate wait
                self.ctime = time.time() + float(args[0])
                self.lines[-1] -= 1        
                self.log('-- wait started for %2.1f'%float(args[0]))
            else:
                if time.time() > self.ctime:  # if time has passed, finish waiting
                    self.ctime = 0
                    self.log('-- wait finished')
                    
        if cmd == 'd_save_img':
            self.save_img()
            
        if cmd == 'd_set_sample_name':
            self.log('-- switching sample name to %s'%args[0])
            self.sname_var.set(args[0])
        
        if cmd == 'd_set_m30_params':
            self.set_m30_params(float(args[0]), float(args[1]))
            
        if cmd == 'd_set_temp':
            self.log('-- setting temperature to %2.2f'%float(args[0]))
            self.set_temp_var.set(float(args[0]))
            self.set_temperature()
            
        if cmd == 'd_wait_temp':
            if abs(self.tc300_temp - self.tc300_target) > float(args[0]):
                self.lines[-1] -= 1
            else:          
                self.log('-- wait for temperature finished')
        
        if cmd == 'd_move_relz':
            if self.lj != None:
                self.btn_lj_moverel(float(args[0]))
            else:
                self.log('-- LabJack not connected')
                
        if cmd == 'd_move_absz':
            if self.lj != None:
                self.lj_var.set(args[0])
                self.btn_lj_set()
            else:
                self.log('-- LabJack not connected')
                
        if cmd == 'd_wait_move_z':
            if abs(self.lj_param[0] - self.lj_param[1]) > 0.1:
                self.lines[-1] -= 1    
            else:          
                self.log('-- wait for move finished')
        
        if cmd == 'd_aux':
            if self.aux_state[int(args[0])] == int(args[1]):
                self.log('-- AUX pin %d is already set to %d'%(int(args[0]), int(args[1])))
            else:
                self.log('-- setting AUX pin %d to %d'%(int(args[0]), int(args[1])))
                self.aux_btn(int(args[0]))
                
        if cmd == 'd_rigol_dm':
            if self.rigol != None:
                self.log('-- sending %s to rigol'%args[0])
                self.rigol.write(args[0])
            else:
                self.log('-- rigol not connected')
        
        if cmd == 'd_move_rel':         
            self.m30_param[2] += float(args[0])
            self.m30_param[3] += float(args[1])
        
            self.log('-- M30 moving to X=%2.3f Y=%2.3f'%(self.m30_param[2], self.m30_param[3]))
            
        if cmd == 'd_ahk':
            program = '"c:/Program Files/AutoHotkey/AutohotkeyU64.exe"'
            path = '"c:/Users/LAB218/Documents/mikroskop2'
            command_to_send = f'{program} {path}/{args[0]}" {args[1]}'
            subprocess.run(command_to_send, check=True, shell=True)
            self.log(f"Executing command {command_to_send}")

    
    def aux_btn(self, val):
        if self.aux != None:    
            if self.aux_state[val] == 1:
                to_set = 0
            else:
                to_set = 1
            
            self.aux_state[val] = to_set
            if self.aux_state[val] == 1:
                self.aux.write(bytes('OUTD_%d_%d\r\n'%(val, 1), 'utf-8'))
                self.log('Toggle 1 on aux pin %d'%val)
                self.auxButtons[val].configure(relief = SUNKEN)
            else:
                self.aux.write(bytes('OUTD_%d_%d\r\n'%(val, 0), 'utf-8'))
                self.log('Toggle 0 on aux pin %d'%val)
                self.auxButtons[val].configure(relief = RAISED)
        else:
            self.log('AUX not connected!')

    
    def aux_connect(self, port):
        # port = self.aux_con_var.get()
        try:
            self.aux = serial.Serial(port, timeout = 3)
            if str(self.aux).find('open=True') > 0:
                self.log('Connected to AUX on %s'%port)
            else:
                self.log('Connection to AUX on %s failed!'%port)
                self.aux = None
                
            if self.aux == None:        
                self.lab_aux_con.config(text = 'not connected', bg='red')
            else:
                self.lab_aux_con.config(text = 'connected', bg='lime')
        except:
            self.log('Failed to connect to AUX')
            self.lab_aux_con.config(text = 'not connected', bg='red')
    

    def print_script(self):
        for i, line in enumerate(self.current_script):
            self.log(str(i) + '. ' + str(line))
    
    
    def rigol_set_laserduty(self, value):
        '''this changes self.laserduty variable, also updates the label'''
        self.laserduty = float(value)
        self.lab_laser.config(text = 'DUTY = %2.2f %%, CH1:LASER IS %s'%(self.laserduty, self.laserstate))
        
        if self.rigol != None:
            self.rigol.write(':SOUR1:FUNC:SQU:DCYC %2.2f'%self.laserduty)
            time.sleep(0.25)
            resp = self.rigol.query(':SOUR1:FUNC:SQU:DCYC?')
            self.log('CH1:Laser duty cycle set to %s.'%resp.replace('\n', ''))
            
            
    def rigol_set_laserduty2(self, value):
        '''this changes self.laserduty variable, also updates the label'''
        self.laserduty2 = float(value)
        self.lab_laser2.config(text = 'DUTY = %2.2f %%, CH2:AUX IS %s'%(self.laserduty2, self.laserstate2))
        
        if self.rigol != None:
            self.rigol.write(':SOUR2:FUNC:SQU:DCYC %2.2f'%self.laserduty2)
            time.sleep(0.25)
            resp = self.rigol.query(':SOUR2:FUNC:SQU:DCYC?')
            self.log('CH2:aux duty cycle set to %s.'%resp.replace('\n', ''))
    
    
    def connect_rigol(self):
        if self.rigol != None:
            messagebox.showinfo(title='Rigol', message='Rigol is already connected?')
        else:
            rm = pyvisa.ResourceManager()
            try:
                inst = rm.open_resource('USB0::0x1AB1::0x0643::DG8A224704187::INSTR')
                if inst.query("*IDN?")[:18] == 'Rigol Technologies':
                    self.rigol = inst
                    inst.write(':SOUR1:APPL:SQU 1000,5,2.5,0')
            except:
                messagebox.showerror(title='Rigol', message='Connection to Rigol failed!')
                self.rigol = None
                
        if self.rigol == None:        
            self.label_rigol_status2.config(text = 'not connected', bg='red')
            self.log('Rigol connection failed!')
        else:
            self.label_rigol_status2.config(text = 'connected', bg='lime')
            self.log('Rigol connected!')

    
    def connect_lj(self):
        if self.lj == None:
            self.lj = LabJackZ(self.lj_event, self.lj_param)
            self.lj.start()
        else:
            messagebox.showinfo(title='LabJack', message='LabJack is already connected?')
            
        if self.lj == None:
            self.label_lj_status2.config(text = 'not connected', bg='red')
            self.log('LabJack connection failed!')
        else:
            self.label_lj_status2.config(text = 'connected', bg='lime')
            self.log('LabJack connected!')

   
    def connect_m30(self):
        if self.m30 == None:
            self.m30 = M30(self.m30_event, self.m30_param)
            self.m30.start()
        else:
            messagebox.showinfo(title='M30', message='M30 is already connected?')
            
        if self.m30 == None:
            self.label_m30_status2.config(text = 'not connected', bg='red')
            self.log('M30 connection failed!')
        else:
            self.label_m30_status2.config(text = 'connected', bg='lime')
            self.log('M30 connected!')
    
    
    def btn_lj_set(self):
        if self.lj != None:
            val = float(self.lj_var.get())
            self.lj_param[1] = val
            if self.lj_param[1] > ljLimits[1]:
                self.log('LabJack position limit exceeded!')
                self.lj_param[1] = ljLimits[1]
            if self.lj_param[1] < ljLimits[0]:
                self.log('LabJack position limit exceeded!')
                self.lj_param[1] = ljLimits[0]
            self.log('Moving LabJack to %2.3f absolute position'%val)
        else:
            self.log('LabJack not connected')
            
        
    def btn_lj_home(self):
        if self.lj != None:            
            self.lj_param[2] = 1
            self.log('Moving LabJack to home position')
        else:
            self.log('LabJack not connected')
            
    
    def btn_lj_moverel(self, val):
        if self.lj != None:            
            self.lj_param[1] += val
            if self.lj_param[1] > ljLimits[1]:
                self.log('LabJack position limit exceeded!')
                self.lj_param[1] = ljLimits[1]
            if self.lj_param[1] < ljLimits[0]:
                self.log('LabJack position limit exceeded!')
                self.lj_param[1] = ljLimits[0]
            self.log('Moving LabJack %2.2f relative (%2.2f absolute)'%(val, self.lj_param[1]))
        else:
            self.log('LabJack not connected')
        
    
    def fmenu_load_script(self):
        # obsolete
        filename = filedialog.askopenfilename()
        fname = filename.replace('.txt', '')
        fname = fname[fname.rfind('/')]
        print(filename)
        print(fname)
        self.current_script = read_script(fname)
        self.print_script()
        
    
    def camera_deltagain(self, deltagain):
        self.camera_gain += deltagain
        if self.camera_gain > 63 : self.camera_gain = 63
        if self.camera_gain < 0 : self.camera_gain = 0
        self.config_arr[0] = self.camera_gain
        self.labelGain.config(text ='%d'%self.camera_gain)
        
        
    def initiate_projector_window(self):
        if self.projector_window == None:                
            # self.projector_window = ProjectorWindow(root)
            # self.app = ProjectorWindow(self.projector_window)
            self.projector_window = Toplevel(root)
            self.projector_window.title("Projector window - move to projector screen")
            self.projector_window.geometry("400x400")
            self.log('Opened projector window')
            
    
    def close_projector_window(self):
        if self.projector_window != None:
            self.projector_window.destroy()
            self.projector_window = None
            self.log('Closed projector window')
    
    
    def fmenu_load_img(self):        
        filename = filedialog.askopenfilename(initialdir="C:/Users/LAB218/Documents/mikroskop2/patterns")
        if filename == '': return 1
        self.load_pattern_image(filename)

        
    def activate_projector_window(self):
        print('Projector window activated!')
        
        # initialize full screen mode
        self.projector_window.overrideredirect(True)
        self.projector_window.state("zoomed")
        # self.projector_window.activate()
        
        self.canvas_proj = Canvas(self.projector_window, width=1024, height=768, bg='black', highlightthickness=0, relief='ridge')
        self.canvas_proj.pack(side = LEFT)
        self.log('Projector window activated')
    

    def load_pattern_image(self, path):
        self.projector_arr = cv2.imread(path)
        self.refresh_projector_image()
        self.log('Image %s loaded'%path)
    
    
    def refresh_projector_image(self):
        # refresh image displayed in window (4x smaller res)
        img = cv2.resize(self.projector_arr, (256, 192), interpolation = cv2.INTER_AREA)
        img = Image.fromarray(img)
        self.proj_imgtk_mirror = ImageTk.PhotoImage(image=img)
        # self.proj_mirror_canvas.create_image(128, 96, image=self.proj_imgtk, anchor=CENTER)
        self.proj_mirror_canvas.create_image(0, 0, image=self.proj_imgtk_mirror, anchor=NW)
        
        # refresh the actual screen
        img = Image.fromarray(self.projector_arr)
        self.proj_imgtk = ImageTk.PhotoImage(image=img)
        self.canvas_proj.create_image(512, 384, image=self.proj_imgtk, anchor=CENTER)


    def set_temperature(self):
        if self.tc300_hdl != -1:
            self.tc300_target = float(self.set_temp_var.get())
            res = TC300SetTargetTemperature(self.tc300_hdl, 1, self.tc300_target)
            if res == 0:
                self.log('TC300 target temperature set to %2.2f'%self.tc300_target)
                return 0
        self.log('TC300 set target temperature failed!')
        return 1


    def tc300_connect(self):
        if self.tc300_hdl < 0:
            devs = TC300ListDevices()
            TC300= devs[0]
            serialNumber= TC300[0]
            self.tc300_hdl = TC300Open(serialNumber,115200,3)
            if self.tc300_hdl > 0:
                self.log('Successfully connected to TC300')
            else:
                self.log('Connection to TC300 failed')
            print('hdl', self.tc300_hdl)
            # print current mode (it should be heater
            mode=[0]
            modeList={0: "Heater", 1: "Tec", 2: "Constant current", 3: "Synchronize with Ch1(only for channel 2)"}
            result=TC300GetMode(self.tc300_hdl,1,mode)
            if(result<0):
                self.log("Get mode fail",result)
            else:
                self.log("Get mode:",modeList.get(mode[0]))
                
            # print current target temperature
            TargetTemperature=[0]
            result=TC300GetTargetTemperature(self.tc300_hdl,1,TargetTemperature)
            if(result<0):
                self.log("Get Target Temperature fail",result)
            else:
                self.log("Target Temperature is:",TargetTemperature[0])
            
            
            if self.tc300_hdl == -1:        
                self.label_thrm_status2.config(text = 'not connected', bg='red')
            else:
                self.label_thrm_status2.config(text = 'connected', bg='lime')
                self.tc300_read_temp()
    
    
    def tc300_read_temp(self):        
        if self.tc300_hdl != -1:
            ActualTemperature=[0]
            result=TC300GetActualTemperature(self.tc300_hdl, 1, ActualTemperature)
            self.tc300_temp = ActualTemperature[0]
            self.lab_tempcur.config(text = 'T = %3.2f °C'%self.tc300_temp)
            self.master.after(500, self.tc300_read_temp)
        else:
            self.log('Cannot read from TC300!')
            
    
    def esc_key_btn(self, value):        
        self.buttonSave.focus_set()
        
        
    def exit(self):
        print('exit!')
        
        self.event.set()
        self.m30_event.set()
        self.lj_event.set()
        root.quit()
        
        
    
        
        
if __name__ == '__main__':
    print('Mikroskop 2.0')

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

    root = Tk()
    # root.geometry("1920x1080")

    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (w, h))
    root.state("zoomed")
    # root.iconbitmap('bicon02.ico')

    #creation of an instance
    app = Window(root)
    # app.show_frame()

    #mainloop
    root.iconbitmap('microscope.ico')
    root.mainloop()