import os
import time
from tkinter import messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from multiprocessing import Process
from multiprocessing import Value
from multiprocessing import RawArray
from multiprocessing import Event
from pyueye import ueye


class CameraController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.event = Event()
        # self.sharr = RawArray('c', 448*800*3)  # holds camera frame
        self.sharr = RawArray('c', 960*1280*3)  # holds camera frame
        self.config_arr = RawArray('i', (6, 8, -1))  # hold the camera configuration (gain, expo)
        self.camparam_arr = RawArray('i', (0, 0, 0))  # hold the camera configuration (gain, expo, invert_colors)
        # self.rawFrame = np.zeros((960, 1280, 3), np.uint8)

    def connect(self):
        self.con_stat = "CONNECTING"
        self.cam_reader = CamReader(self.event, self.sharr, self.config_arr, self.camparam_arr)
        self.con_stat = "CONNECTED"

    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    def get_image(self) -> Image:
        frame = np.frombuffer(self.sharr, dtype=np.uint8).reshape(960, 1280, 3)
        # time.sleep(0.75)
        return Image.fromarray(frame)

    def get_status(self) -> dict:
        """Possible values are
        'connection':
            'CONNECTED',
            'CONNECTING',
            'UNKNOWN',
            'NOT CONNECTED'
        """
        return {
            "connection": self.con_stat,
        }


class CamReader(Process):
#     override the constructor
    def __init__(self, event, sharr, config_arr, camparam_arr):
        # execute the base constructor
        Process.__init__(self)
        # initialize integer attribute
        self.event = event
        self.config_arr = config_arr
        self.camparam_arr = camparam_arr
        self.gain = self.config_arr[0]
        self.expo = self.config_arr[1]
        self.expoAbs = self.config_arr[2]
        self.sharr = sharr
        self.data = Value('i', 0)
        # self.cap = cv2.VideoCapture(0)
        # self.frame = np.zeros(img_shape, np.uint8)
        self.change_expo_thisloop = 0
        
        self.hCam = ueye.HIDS(0)             #0: first available camera;  1-254: The camera with the specified camera ID
        self.sInfo = ueye.SENSORINFO()
        self.cInfo = ueye.CAMINFO()
        self.pcImageMemory = ueye.c_mem_p()
        self.MemID = ueye.int()
        self.rectAOI = ueye.IS_RECT()
        self.pitch = ueye.INT()
        self.nBitsPerPixel = ueye.INT(24)    #24: bits per pixel for color mode; take 8 bits per pixel for monochrome
        self.channels = 3                    #3: channels for color mode(RGB); take 1 channel for monochrome
        self.m_nColorMode = ueye.INT()		# Y8/RGB16/RGB24/REG32
        self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
        self.width = ueye.INT()
        self.height = ueye.INT()
        print('cam reader initialized')
        
        
    def init_ui_camera(self):      
        nRet = ueye.is_InitCamera(self.hCam, None)
        if nRet != ueye.IS_SUCCESS:
            print("is_InitCamera ERROR")
            
        # Reads out the data hard-coded in the non-volatile camera memory and writes it to the data structure that cInfo points to
        nRet = ueye.is_GetCameraInfo(self.hCam, self.cInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetCameraInfo ERROR")
            
        
        # You can query additional information about the sensor type used in the camera
        nRet = ueye.is_GetSensorInfo(self.hCam, self.sInfo)
        if nRet != ueye.IS_SUCCESS:
            print("is_GetSensorInfo ERROR")

        nRet = ueye.is_ResetToDefault(self.hCam)
        if nRet != ueye.IS_SUCCESS:
            print("is_ResetToDefault ERROR")


        # Set display mode to DIB
        nRet = ueye.is_SetDisplayMode(self.hCam, ueye.IS_SET_DM_DIB)
                
        # Set the right color mode
        if int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_BAYER:
            # setup the color depth to the current windows setting
            ueye.is_GetColorDepth(self.hCam, self.nBitsPerPixel, self.m_nColorMode)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_BAYER: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_CBYCRY:
            # for color camera models use RGB32 mode
            self.m_nColorMode = ueye.IS_CM_BGRA8_PACKED
            self.nBitsPerPixel = ueye.INT(32)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_CBYCRY: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        elif int.from_bytes(self.sInfo.nColorMode.value, byteorder='big') == ueye.IS_COLORMODE_MONOCHROME:
            # for color camera models use RGB32 mode
            self.m_nColorMode = self.ueye.IS_CM_MONO8
            self.nBitsPerPixel = self.ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("IS_COLORMODE_MONOCHROME: ", )
            print("\tm_nColorMode: \t\t", self.m_nColorMode)
            print("\tnBitsPerPixel: \t\t", self.nBitsPerPixel)
            print("\tbytes_per_pixel: \t\t", self.bytes_per_pixel)
            print()

        else:
            # for monochrome camera models use Y8 mode
            self.m_nColorMode = ueye.IS_CM_MONO8
            self.nBitsPerPixel = ueye.INT(8)
            self.bytes_per_pixel = int(self.nBitsPerPixel / 8)
            print("else")
            
            
        # Can be used to set the size and position of an "area of interest"(AOI) within an image
        nRet = ueye.is_AOI(self.hCam, ueye.IS_AOI_IMAGE_GET_AOI, self.rectAOI, ueye.sizeof(self.rectAOI))
        if nRet != ueye.IS_SUCCESS:
            print("is_AOI ERROR")

        self.width = self.rectAOI.s32Width
        self.height = self.rectAOI.s32Height

        # Allocates an image memory for an image having its dimensions defined by width and height and its color depth defined by nBitsPerPixel
        nRet = ueye.is_AllocImageMem(self.hCam, self.width, self.height, self.nBitsPerPixel, self.pcImageMemory, self.MemID)
        if nRet != ueye.IS_SUCCESS:
            print("is_AllocImageMem ERROR")
        else:
            # Makes the specified image memory the active memory
            nRet = ueye.is_SetImageMem(self.hCam, self.pcImageMemory, self.MemID)
            if nRet != ueye.IS_SUCCESS:
                print("is_SetImageMem ERROR")
            else:
                # Set the desired color mode
                nRet = ueye.is_SetColorMode(self.hCam, self.m_nColorMode)


        # Activates the camera's live video mode (free run mode)
        nRet = ueye.is_CaptureVideo(self.hCam, ueye.IS_DONT_WAIT)
        if nRet != ueye.IS_SUCCESS:
            print("is_CaptureVideo ERROR")

        # Enables the queue mode for existing image memory sequences
        nRet = ueye.is_InquireImageMem(self.hCam, self.pcImageMemory, self.MemID, self.width, self.height, self.nBitsPerPixel, self.pitch)
        if nRet != ueye.IS_SUCCESS:
            print("is_InquireImageMem ERROR")
        else:
            print("New camera loaded")
            
            
        # FPS fix
        time.sleep(2)
        
        print('Attemtping FPS change')
        
        
        fps = ueye.double()
        ret = ueye.is_GetFramesPerSecond(self.hCam, fps)
        print('Current FPS', ret, fps)
        time.sleep(0.5)
        
        number = ueye.UINT()
        ret = ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_SET, ueye.UINT(90), ueye.sizeof(number))
        print('PxCLK set:',ret, number)
        ret = ueye.is_PixelClock(self.hCam, ueye.IS_PIXELCLOCK_CMD_GET, number, ueye.sizeof(number))
        print('PxCLK current:',ret, number)
        time.sleep(0.5)
        
        fps = ueye.double(15)
        fps_actual = ueye.double()
        ret = ueye.is_SetFrameRate (self.hCam, fps, fps_actual)
        print('Changing FPS', ret, fps, fps_actual)
        time.sleep(1)
        
        fps = ueye.double()
        ret = ueye.is_GetFramesPerSecond(self.hCam, fps)
        print('Current FPS', ret, fps)
        time.sleep(0.5)
        


    # override the run function
    def run(self):
        self.init_ui_camera()
        while(True):
            try:
                # read frame from new camera
                array = ueye.get_data(self.pcImageMemory, self.width, self.height, self.nBitsPerPixel, self.pitch, copy=False)
                # bytes_per_pixel = int(nBitsPerPixel / 8)
                # ...reshape it in an numpy array...
                frame = np.reshape(array, (self.height.value, self.width.value, self.bytes_per_pixel))
                # ...resize the image by a half (frame should be 1280x960 now)
                frame = cv2.resize(frame,(0,0),fx=0.5, fy=0.5)
                
                # correctly resize frame
                final_img = np.zeros((960, 1280, 4), dtype = np.ubyte)
                final_img[:, :, :] = frame
                
                # correctly resize frame
                # w, h = frame.shape[1], frame.shape[0]
                # frame = cv2.resize(frame, (int(w*(448/800)), 448))
                # final_img = np.zeros((448, 800, 4), dtype = np.ubyte)
                # pos_x = int((800-frame.shape[1])/2)
                # final_img[:, pos_x:pos_x + frame.shape[1], :] = frame
                if self.camparam_arr[2] == 1:
                    final_img[:,:,2] = final_img[:,:,0]
            except:
                # final_img = generate_random_image()
                # final_img = generate_random_image(shape = (1280, 960))
                pass
                
            # copy frame to shared array (sharr)
            self.sharr_np = np.frombuffer(self.sharr, dtype=np.uint8).reshape(*[960, 1280, 3])
            np.copyto(self.sharr_np, final_img[:,:,:3])
            
            
            # check if config requires change
            # if self.gain != self.config_arr[0]:
                # self.change_gain(self.config_arr[0])
            # if self.expo != self.config_arr[1]:
                # self.change_expo(self.config_arr[1])
            
            # time.sleep(0.010)
            # self.pGamma = ueye.double()
            # self.pGamma.value = 128
            # # print(sys.getsizeof(self.pGamma))
            # # ret = ueye.is_Gamma(self.hCam, ueye.IS_GAMMA_CMD_GET, self.pGamma, 8)
            # ret = ueye.is_SetAutoParameter(self.hCam, ueye.IS_SET_AUTO_REFERENCE, self.pGamma.value, 0)
            # print(f'ret: {ret}')
            # d = {
            #     ueye.IS_INVALID_PARAMETER: 'IS_INVALID_PARAMETER',
            #     ueye.IS_NO_SUCCESS: 'IS_NO_SUCCESS',
            #     ueye.IS_NOT_SUPPORTED: 'IS_NOT_SUPPORTED',
            #     ueye.IS_SUCCESS: 'IS_SUCCESS',
            # }
            # print(f'pGama is {self.pGamma.value}, ret: {ret}, means: {d[ret]}')


            
            # change expo
            if self.config_arr[1] != 0:
                time.sleep(0.01)
                self.pParam = ueye.double()
                ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, self.pParam, 8)
                self.pParam.value = self.pParam.value + self.config_arr[1]
                time.sleep(0.01)
                ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, self.pParam, 8)
                time.sleep(0.01)
                ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, self.pParam, 8)
                self.camparam_arr[1] = int(self.pParam.value)
                print('cur exp read:', self.pParam.value)
                self.config_arr[1] = 0
                    
            if self.config_arr[2] != 0:
                time.sleep(0.01)
                self.pGamma = ueye.double()
                ueye.is_Gamma(self.hCam, ueye.IS_GAMMA_CMD_GET, self.pGamma, 8)
                print(f'pGama is {self.pGamma.value}, adding {self.config_arr[2]}')
                self.pGamma.value += self.config_arr[2]
                print(f'pGama is {self.pGamma.value}')
                time.sleep(0.01)
                ueye.is_Gamma(self.hCam, ueye.IS_GAMMA_CMD_SET, self.pGamma, 8)
                time.sleep(0.01)                
                ueye.is_Gamma(self.hCam, ueye.IS_GAMMA_CMD_GET, self.pGamma, 8)
                self.camparam_arr[2] = int(self.pGamma.value)
                print('cur gamma read:', self.pGamma.value)
                self.config_arr[2] = 0
                
                
            # time.sleep(0.275)
            # self.pParam = ueye.double()
            # self.pParam.value = 30
            # ueye.is_Exposure(self.hCam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, self.pParam, 8)
            # time.sleep(0.275)

            if self.event.is_set():
                print('Cam reader: Call to exit camera received!')
                # Releases an image memory that was allocated using is_AllocImageMem() and removes it from the driver management
                ueye.is_FreeImageMem(self.hCam, self.pcImageMemory, self.MemID)

                # Disables the hCam camera handle and releases the data structures and memory areas taken up by the uEye camera
                ueye.is_ExitCamera(self.hCam)
                break
            
    
    def change_gain(self, gain):
        parameter = cv2.CAP_PROP_GAIN
        self.cap.set(parameter, gain)
        print('new camera gain %s'%str(gain))
        
    
    def change_expo(self, expo):
        print('change expo is', self.change_expo_thisloop)
        self.change_expo_thisloop = -1
        print('change expo is', self.change_expo_thisloop)
