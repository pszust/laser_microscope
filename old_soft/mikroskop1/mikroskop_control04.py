import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
from tkinter import *
from PIL import Image, ImageTk, ImageFont, ImageDraw
import cv2
from PIL import Image, ImageTk
import os
import sys, ftd2xx as ftd
import numpy as np
import serial
import sys
import glob
import time

width, height = 800, 600
cap = cv2.VideoCapture(1)
padd = 20
frame_x = 0
frame_y = 0
elli_angle = 0
elli_relative_zero = 149.5  # this is absolute angle at which relative angle is 90


# constants
inc_x = int(1024/5)
inc_y = int(768/5)  
rgb_weights = [0.2989, 0.5870, 0.1140]
calib_dots_dim = 4
ckernel = np.asarray([
    [0,1,1,1,0],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [0,1,1,1,0]
]).astype(np.uint8)

RAW = 0
MAP = 1
DYN = 2


# counters 
projector_calib_c = calib_dots_dim**2+1


def list_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


def ft_read(d, nbytes):
    s = d.read(nbytes)
    return s


def connect_to_elliptec(com_i, timeout = 3):
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port='COM%d'%com_i
    ser.timeout = timeout
    
    try:
        ser.open()    
        ser.write(b'0in')
        resp = ser.read(32)
        if resp == b'0IN0E114002352021150101680002300': 
            print('Connected to elliptec on COM%d'%com_i)
            return ser
        else:
            print('COM%d is not elliptec')
            return None
    except:
        print('Device on COM%d is not available'%com_i)
        return None


def angle_to_ellocommand(value):
    value = int(value*398)
    if value < 0: 
        value_hex = str(hex(((abs(value) ^ 0xffffffff) + 1) & 0xffffffff))
    else:
        value_hex = str(hex(value))


    value_hex = value_hex[value_hex.find('x')+1:].zfill(8)
    value_hex = value_hex.replace('a', 'A')
    value_hex = value_hex.replace('b', 'B')
    value_hex = value_hex.replace('c', 'C')
    value_hex = value_hex.replace('d', 'D')
    value_hex = value_hex.replace('e', 'E')
    value_hex = value_hex.replace('f', 'F')

    return bytes('0ma%s'%value_hex.zfill(8), 'ascii')


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
    
    
def calib_get_postion(comp, sigma = 151):
    # check if it is not outside (no dot)
    if comp.max() < 25:
        return -1
    
    # blur and normalize
    comp = cv2.GaussianBlur(comp, (sigma, sigma), cv2.BORDER_DEFAULT)
    comp = (comp-comp.min())
    comp = comp/comp.max()*255
    
    # threshold, find countor, get center
    tresh = (comp.max()-comp.min())/2
    ret, thresh1 = cv2.threshold(np.uint8(comp), tresh, 1, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    if M["m00"] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    else:
        cx = 0
        cy = 0
    
    return [cx, cy]


def num_to_coords(num, size = 10, dim = (1024, 768)):    
    nx = int(num%size)
    ny = int((num-num%size)/size)
    
    inc_x = int(dim[0]/size)
    inc_y = int(dim[1]/size)
    
    x = int(inc_x/2) + nx*inc_x
    y = int(inc_y/2) + ny*inc_y
    
#     print('x = %d, y = %d'%(x, y))
    return x, y


def get_homography_matrix():
    baseline = np.dot(np.load('calibration/baseline.npy')[...,:3], rgb_weights)
    # images = []
    # for i in range(0, calib_dots_dim**2):
        # temp = np.load('calibration/num%d.npy'%i)
        # images.append(np.dot(temp[...,:3], rgb_weights))
        
    coords_prj = []
    coords_cam = []

    for i in range(0, calib_dots_dim**2):
        img = np.load('calibration/num%d.npy'%i)
        img = np.dot(img[...,:3], rgb_weights)
        c = calib_get_postion(img-baseline)
        if c != -1:
            coords_cam.append(c)
            crds = num_to_coords(i, size = calib_dots_dim)
            coords_prj.append([crds[0], crds[1]])
            
    coords_prj_arr = np.array(coords_prj)
    coords_cam_arr = np.array(coords_cam)
    
    h, status = cv2.findHomography(coords_cam_arr, coords_prj_arr)
    im_out = cv2.warpPerspective(baseline, h, (1024, 768))
    
    fig = plt.figure(figsize = (12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.set_title('camera image')
    ax1.imshow(baseline, cmap = 'gray')
    for i, crd in enumerate(coords_cam):
        ax1.text(crd[0], crd[1], str(i))
        
    ax2.set_title('projector array')
    ax2.imshow(im_out, cmap = 'gray')
    for i, crd in enumerate(coords_prj):
        ax2.text(crd[0], crd[1], str(i), c = 'white')
        
    fig.savefig('calibration/calib_result.png', dpi = 400)
    
    return h


def angle_abs_to_rel(ang_abs):
    rel = elli_relative_zero + 90 - ang_abs
    if rel > 360: rel -= 360
    if rel < 0: rel += 360
    return rel


def angle_rel_to_abs(ang_rel):
    absa = elli_relative_zero - ang_rel + 90
    if absa > 360: absa -= 360
    if absa < 0: absa += 360
    return absa
    

    
class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master
        
        # loadable internal variables
        self.elli_angle = 151  # this was previously called abs_ang
        self.pulseOn = 1
        self.pulseOff = 1
        
        # non-loadable variables (are modified inside app)
        self.mode_draw = 0
        self.erosion = 0
        self.erosion_counter = 0
        self.erosion_delay = 1
        self.hold_proj = 0   
        self.camera_overlay = 0
        self.brush_size = 50
        self.always_overlay = 0
        self.camera_image_type = RAW
        self.camera_map_counter = 0
        self.frame_map_switch = 8
        self.polrot = 10
        
        
        self.elliptec = None
        self.video_writer = None
        self.projector_window = None
        self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
        # self.projector_arr = cv2.circle(self.projector_arr, (500, 300), 50, (255, 255, 255), -1)
        
        # this is grayscale image that overlays the camera and it is projected through homography to projector screen
        self.camera_draw = np.zeros((448, 800, 3), np.uint8)
        self.camera_draw = cv2.circle(self.camera_draw, (500, 300), 50, (255, 255, 255), -1)
        
        # matrix for homography (is set during calibration)
        self.homomatrix = np.zeros((3,3))
        
        self.image_baseline = np.zeros((600, 800, 3), np.uint8)
        
        # logo for fun/testing
        self.logo = cv2.imread('unilogo.png')
        self.projector_arr = self.logo
        
        # this is mouse xy relative to camera canvas
        self.mouse_x = 0
        self.mouse_y = 0

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()
        self.main_loop()
        self.elli_refresh('COM1')
    
    
    #Creation of init_window
    def init_window(self):
        col = 0
        row = 0
        self.save_image = False

        # changing the title of our master widget      
        self.master.title("Mikroskop control")
        
        # menu
        menu = Menu(root)
        self.master.config(menu=menu)
        
        fileMenu = Menu(menu, tearoff=False)
        fileMenu.add_command(label="Run projector calibration", command=self.start_calib)
        fileMenu.add_command(label="Load projector calibration", command=self.load_calib)
        fileMenu.add_command(label="Reload variables", command=self.reload_variables)
        fileMenu.add_command(label="Exit", command=self.exit)
        menu.add_cascade(label="File", menu=fileMenu)
        
        # keybindings
        self.master.bind('[', self.brush_dec)
        self.master.bind(']', self.brush_inc)
        
        
        #  -- FIRST COLUMN --
        # main image        
        # self.lmain = Label(root)
        # self.lmain.grid(row=row, column=col, padx = padd)
        self.canvas = Canvas(width=800, height=600, bg='black')
        self.canvas.grid(row=row, column=col, padx = padd)
        # self.canvas.bind("<Double-Button-1>", self.image_move)  # for future interactions        
        self.canvas.bind("<ButtonPress-1>", self.mouse_motionB1)
        # self.canvas.bind("<ButtonRelease-1>", self.cam_btn_release)
        self.canvas.bind('<B1-Motion>', self.mouse_motionB1)
        self.canvas.bind('<Motion>', self.mouse_motion)
        row += 1
        
        # IMAGE CONTROLS FRAME
        current_frame = Frame(root)
        current_frame.grid(row=row, column=col, padx = padd)
        row += 1
        self.label = Label(current_frame, text = 'Display controls: ')
        self.label.pack(side =  LEFT)
        
        source_var = StringVar(self.master)
        source_var.set(0) # default value
        self.source_menu = OptionMenu(current_frame, source_var, *list_cameras(), command = self.mod_source)
        self.source_menu.pack(side = LEFT)
        
        self.btnHoldProj = Button(current_frame, text = 'Hold', command = self.btn_hold_proj)
        self.btnHoldProj.pack(side = LEFT)
        self.btnAlwaysOverlay = Button(current_frame, text = 'Always overlay', command = self.btn_always_overlay)
        self.btnAlwaysOverlay.pack(side = LEFT)
        
        # DRAWING CONTROLS FRAME
        current_frame = Frame(root)
        current_frame.grid(row=row, column=col, padx = padd)
        row += 1
        self.label = Label(current_frame, text = 'Camera draw controls: ')
        self.label.pack(side =  LEFT)
        
        self.btnDraw = Button(current_frame, text = 'Draw', command = self.btn_draw)
        self.btnDraw.pack(side = LEFT)      
        self.btnPoint = Button(current_frame, text = 'Point', command = self.btn_erosion, state = DISABLED)
        self.btnPoint.pack(side = LEFT)             
        self.btnErosion = Button(current_frame, text = 'Erosion', command = self.btn_erosion)
        self.btnErosion.pack(side = LEFT) 
        
        eros_var = StringVar(self.master)
        eros_var.set(1) # default value
        self.mnErospeed = OptionMenu(current_frame, eros_var, *[1, 2, 3, 4, 5, 8, 10, 15, 20], command = self.mn_set_erospeed)
        self.mnErospeed.pack(side = LEFT)
        
        self.btnClear = Button(current_frame, text = 'Clear', command = self.btn_clear_camdraw)
        self.btnClear.pack(side = LEFT)
        
        # SAVING CONTROLS FRAME
        current_frame = Frame(root)
        current_frame.grid(row=row, column=col, padx = padd)
        row += 1
        self.label = Label(current_frame, text = 'Image type controls: ')
        self.label.pack(side =  LEFT)
        
        self.btnIcntRaw = Button(current_frame, text = 'Raw', command = self.btn_icnt_raw)
        self.btnIcntRaw.pack(side = LEFT)        
        self.btnIcntRaw.config(relief=SUNKEN)
        self.btnIcntDyn = Button(current_frame, text = 'Dynamic', command = self.btn_icnt_dyn)
        self.btnIcntDyn.pack(side = LEFT)
        self.btnIcntMap = Button(current_frame, text = 'Map', command = self.btn_icnt_map)
        self.btnIcntMap.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Blink speed: ')
        self.label.pack(side =  LEFT)
        fms_var = StringVar(self.master)
        fms_var.set(8) # default value
        self.mnFMS = OptionMenu(current_frame, fms_var, *[3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 50], command = self.mn_set_fms)
        self.mnFMS.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Rotation: ')
        self.label.pack(side =  LEFT)
        polrot_var = StringVar(self.master)
        polrot_var.set(10) # default value
        self.mnPolrot = OptionMenu(current_frame, polrot_var, *[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15], command = self.mn_set_polrot)
        self.mnPolrot.pack(side = LEFT)
        

        # SAVING CONTROLS FRAME
        current_frame = Frame(root)
        current_frame.grid(row=row, column=col, padx = padd)
        row += 1
        
        self.label_save = Label(current_frame, text = 'Sample name: ')
        self.label_save.pack(side =  LEFT)
        
        self.path = StringVar()
        self.nameEntered = Entry(current_frame, width = 15, textvariable = self.path)
        self.nameEntered.pack(side = LEFT)
        
        self.buttonSave = Button(current_frame, text = 'Save', command = self.save_img)
        self.buttonSave.pack(side = LEFT)
        
        self.buttonRecord = Button(current_frame, text = 'Record video', command = self.record_video)
        self.buttonRecord.pack(side = LEFT)
        
        
        # image type controls
        # type_frame = Frame(root)
        # type_frame.grid(row=row, column=col, padx = padd)
        # row += 1
        
        
        
        
        
        #  -- SECOND COLUMN --
        row = 0
        col += 1
        
        # elliptec control frame
        ell_control_frame = Frame(root)
        ell_control_frame.grid(row=row, column=col, padx = padd)
        row += 1
        
        # elliptec position frame
        ell_pos_frame = Frame(ell_control_frame)
        ell_pos_frame.pack(fill = Y)        
        self.label_elli_pos_abs = Label(ell_pos_frame, text = 'Abs pos = %2.2f'%0)
        self.label_elli_pos_abs.config(font=('Segoe UI', 16))
        self.label_elli_pos_abs.pack(side =  LEFT)
        
        ell_pos_frame = Frame(ell_control_frame)
        ell_pos_frame.pack(fill = Y) 
        self.label_elli_pos_rel = Label(ell_pos_frame, text = 'Rel pos = %2.2f'%0)
        self.label_elli_pos_rel.config(font=('Segoe UI', 16))
        self.label_elli_pos_rel.pack(side =  LEFT)
        
        
        # elliptec connection frame
        ell_con_frame = Frame(ell_control_frame)
        ell_con_frame.pack(fill = Y)
        
        ell_com_var = StringVar(self.master)
        ell_com_var.set('COM3') # default value
        self.ell_com_menu = OptionMenu(ell_con_frame, ell_com_var, *serial_ports(), command = self.elli_refresh)
        self.ell_com_menu.pack(side = LEFT)
        
        self.label_ell_status = Label(ell_con_frame, text = 'Elliptec status: ')
        self.label_ell_status.pack(side =  LEFT)
        
        self.label_ell_status2 = Label(ell_con_frame, text = 'unknown', bg='gray')
        self.label_ell_status2.pack(side =  LEFT)       
        
        
        ell_rot_frame = Frame(ell_control_frame)
        ell_rot_frame.pack(fill = Y)        
        self.label_ell1 = Label(ell_rot_frame, text = 'Rotate absolute: ')
        self.label_ell1.pack(side =  LEFT)        
        self.ell_var = StringVar()
        self.ell_var.set('0.0')
        self.nameEntered = Entry(ell_rot_frame, width = 15, textvariable = self.ell_var)
        self.nameEntered.pack(side = LEFT, fill = X)        
        self.buttonRotate = Button(ell_rot_frame, text = 'Rotate', command = self.rotate_elli_abs)
        self.buttonRotate.pack(side = LEFT)
        
        ell_rot_frame = Frame(ell_control_frame)
        ell_rot_frame.pack(fill = Y)        
        self.label_ell1 = Label(ell_rot_frame, text = 'Rotate relative: ')
        self.label_ell1.pack(side =  LEFT)        
        self.ell_var2 = StringVar()
        self.ell_var2.set('90.0')
        self.nameEntered2 = Entry(ell_rot_frame, width = 15, textvariable = self.ell_var2)
        self.nameEntered2.pack(side = LEFT, fill = X)        
        self.buttonRotate2 = Button(ell_rot_frame, text = 'Rotate', command = self.rotate_elli_rel)
        self.buttonRotate2.pack(side = LEFT)
        
        ell_rot_frame = Frame(ell_control_frame)
        ell_rot_frame.pack(fill = Y) 
        self.buttonRotateStepM1 = Button(ell_rot_frame, text = '-10', command = lambda: self.rotate_elli_step(-10))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(ell_rot_frame, text = '-5', command = lambda: self.rotate_elli_step(-5))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(ell_rot_frame, text = '-1', command = lambda: self.rotate_elli_step(-1))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(ell_rot_frame, text = '-0.25', command = lambda: self.rotate_elli_step(-0.25))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(ell_rot_frame, text = '+0.25', command = lambda: self.rotate_elli_step(0.25))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(ell_rot_frame, text = '+1', command = lambda: self.rotate_elli_step(1))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(ell_rot_frame, text = '+5', command = lambda: self.rotate_elli_step(5))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(ell_rot_frame, text = '+10', command = lambda: self.rotate_elli_step(10))
        self.buttonRotateStepP1.pack(side = LEFT)
        
        
        #  -- THIRD COLUMN --
        row = 0
        col += 1
        
        # projector control frame
        proj_frame = Frame(root)
        proj_frame.grid(row=row, column=col, padx = padd)
        row += 1
        
        proj_frame1 = Frame(proj_frame)
        proj_frame1.grid(row=0, column=0, padx = padd)
        
        self.init_proj_win_btn = Button(proj_frame1, text = 'Init projector window', command = self.initiate_projector_window)
        self.init_proj_win_btn.pack(side = LEFT)
        
        self.act_proj_win_btn = Button(proj_frame1, text = 'Activate projector window', command = self.activate_projector_window)
        self.act_proj_win_btn.pack(side = LEFT)
        
        self.act_proj_win_btn = Button(proj_frame1, text = 'Close projector window', command = self.close_projector_window)
        self.act_proj_win_btn.pack(side = LEFT)
        
        
        proj_frame2 = Frame(proj_frame)
        proj_frame2.grid(row=1, column=0, padx = padd)
        
        self.proj_mirror_canvas = Canvas(proj_frame2, width=256, height=192, bg='black')
        self.proj_mirror_canvas.pack(side = LEFT)
        
        
        
    def main_loop(self):
        global projector_calib_c       
        self.show_frame()
        
        if self.projector_window != None:
            # refresh the image on actual projector screen
            self.refresh_projector_image()
            
            # display scaled copy of image in the main window
            resized = cv2.resize(self.projector_arr, (256, 192), interpolation = cv2.INTER_AREA)
            if self.hold_proj == 1:
                resized = cv2.putText(resized, 'PROJECTOR BLANKED', (12, 24), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, cv2.LINE_AA)
            
            img = Image.fromarray(resized)
            self.imgproj_scaled = ImageTk.PhotoImage(image=img)
            self.proj_mirror_canvas.create_image(0, 0, image=self.imgproj_scaled, anchor=NW)
            
            
            # camera draw erosion
            if self.erosion == 1 and self.hold_proj != 1:
                if self.camera_draw.max() != 0:
                    if self.erosion_counter == self.erosion_delay:
                        self.camera_draw = cv2.erode(self.camera_draw, ckernel)
                        self.erosion_counter = 0
                    self.erosion_counter += 1
                
        
        
        if projector_calib_c < calib_dots_dim**2:
            self.proj_calib()
            
            # if projector_calib_c == 0:
                # self.homomatrix = get_homography_matrix()
        
        
        self.master.after(15, self.main_loop)

    def reload_variables(self):
        f = open('external_variables.txt', 'r')
        cnt = f.read()
        f.close()

        cnt_dict = {}
        cnt = cnt.split('\n')
        for c in cnt:
            key, value = c.split(' = ')
            cnt_dict[key] = value
            
        self.elli_angle = cnt_dict['elli_angle']
        self.pulseOff = cnt_dict['pulseOff']
        self.pulseOn = cnt_dict['pulseOn']


    def mn_set_polrot(self, value):
        self.polrot = int(value)
        self.camera_map_counter = 0
    

    def mn_set_fms(self, value):
        self.frame_map_switch = int(value)
        self.camera_map_counter = 0


    def btn_icnt_raw(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.btnIcntRaw.config(relief=SUNKEN)
        self.camera_image_type = RAW


    def btn_icnt_dyn(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.btnIcntDyn.config(relief=SUNKEN)
        self.camera_image_type = DYN


    def btn_icnt_map(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.btnIcntMap.config(relief=SUNKEN)
        self.camera_image_type = MAP   
    
    
    def brush_dec(self, event):
        if self.brush_size > 0:
            self.brush_size -= 5
  
  
    def brush_inc(self, event):
        if self.brush_size < 140:
            self.brush_size += 5
    
    
    def mn_set_erospeed(self, value):
        self.erosion_delay = int(value)
        self.erosion_counter = 0
    
 
    def btn_draw(self):
        to_set = 0
        if self.mode_draw == 1:
            to_set = 0
            self.btnDraw.config(relief=RAISED)
        if self.mode_draw == 0:
            to_set = 1            
            self.btnDraw.config(relief=SUNKEN)
        self.mode_draw = to_set
        
        
    def btn_hold_proj(self):
        to_set = 0
        if self.hold_proj == 1:
            to_set = 0
            self.btnHoldProj.config(relief=RAISED)
            self.camera_overlay = 0
        if self.hold_proj == 0:
            to_set = 1            
            self.btnHoldProj.config(relief=SUNKEN)
            self.camera_overlay = 0.3
        self.hold_proj = to_set
        
        
    def btn_erosion(self):
        to_set = 0
        if self.erosion == 1:
            to_set = 0
            self.btnErosion.config(relief=RAISED)
        if self.erosion == 0:
            to_set = 1            
            self.btnErosion.config(relief=SUNKEN)
        self.erosion = to_set

               
    def btn_always_overlay(self):
        to_set = 0
        if self.always_overlay == 1:
            to_set = 0
            self.btnAlwaysOverlay.config(relief=RAISED)
        if self.always_overlay == 0:
            to_set = 1            
            self.btnAlwaysOverlay.config(relief=SUNKEN)
        self.always_overlay = to_set
        
        
    def btn_clear_camdraw(self):
        self.camera_draw = np.zeros((448, 800, 3), np.uint8)

 
    def mouse_motionB1(self, event):
        x = event.x
        y = event.y - 76  # to account for bigger canvas than camera image
        
        if self.mode_draw == 1:
            self.draw_oncamera(x, y)
            
        # sets the mouse coords for drawing brush circle
        self.mouse_x = event.x
        self.mouse_y = event.y


    def mouse_motion(self, event):
        '''
        this only sets the mouse coordinates
        (only when the button is not pressed)
        '''
        self.mouse_x = event.x
        self.mouse_y = event.y
        
    
    def draw_oncamera(self, x, y):
        # print('x = %d, y = %d'%(event.x, event.y))
        # x = event.x
        # y = event.y - 76  # to account for bigger canvas than camera image
        
        # self.camera_draw = np.zeros((448, 800, 3), np.uint8)
        self.camera_draw = cv2.circle(self.camera_draw, (x, y), self.brush_size, (255, 255, 255), -1)
              
    
    def debug_saveimg(self):
        i = int(np.random.random()*100)
        
        ret, frame = cap.read()
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        np.save(self.camera_draw, 'temp/camera_draw_%d'%i)
        np.save(self.projector_arr, 'temp/projector_arr_%d'%i)
        np.save(self.cv2image, 'temp/camera_frame_%d'%i)
  
        
    def load_calib(self):
        self.homomatrix = get_homography_matrix()
        print('Calibration done!')
        print('homography matrix:')
        print(self.homomatrix)


    def start_calib(self):
        global projector_calib_c
        projector_calib_c = -1
        self.frame_counter = 0


    def proj_calib(self):
        # here there are two counters:
        # projector_calib_c - counts where the dots are places
        # self.frame_counter - counts frames passed between putting the dot on projector screen and saving the camera image
        
        global projector_calib_c, inc_x, inc_y        
        
        # at the beginning, make the projector screen black
        if projector_calib_c == -1 and self.frame_counter == 0:
            self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
        
        # if it is done with baseline (projector_calib_c > -1), at frame_counter == 0 start putting dots on projector array
        if projector_calib_c > -1 and self.frame_counter == 0:
            self.proj_x, self.proj_y = num_to_coords(projector_calib_c, size = calib_dots_dim)
        
            self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
            self.projector_arr = cv2.circle(self.projector_arr, (self.proj_x, self.proj_y), 18, (255, 255, 255), -1)
        
        # when N frames passed after last projector array modification (to give time for camera to update image) read and save the camera image 
        if self.frame_counter == 5:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            if projector_calib_c > -1:  # meaining its done with the baseline                
                np.save('calibration/num%d'%projector_calib_c, frame)
            else:
                np.save('calibration/baseline', frame)
        
            projector_calib_c += 1
            self.frame_counter = 0
        else:
            self.frame_counter += 1
            

    def proj_calib_old(self):
        global projector_calib_c, inc_x, inc_y        
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        
        # at the beginning, make the projector screen black
        if projector_calib_c == calib_dots_dim**2+1:
            self.frame_counter = 5            
            self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
            
        # after one cycle of blackness, establish baseline image for further comparison (just before any lasers)    
        if projector_calib_c == calib_dots_dim**2:
            # ret, frame = cap.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.image_baseline = frame.copy()
            np.save('calibration/baseline', self.image_baseline)
        
        # put white dot at specified coordinates and save the camera image
        if self.frame_counter == 5:
            if projector_calib_c < calib_dots_dim**2+1:
                num = calib_dots_dim**2 - projector_calib_c
                self.proj_x, self.proj_y = num_to_coords(num, size = calib_dots_dim)
            
                self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
                self.projector_arr = cv2.circle(self.projector_arr, (self.proj_x, self.proj_y), 18, (255, 255, 255), -1)
                np.save('calibration/num%d'%num, frame)
        
            projector_calib_c -= 1
            self.frame_counter = 0
            
            
        self.frame_counter += 1
        
        
    def record_video(self):
        global frame_x, frame_y
        # start recording new video
        if self.video_writer == None:
            # get the correct filename
            fname = self.path.get()
            if fname == '': fname = 'unnamed'
            all_files = os.listdir(os.getcwd() + '/saved_images/')
            saved = 0
            num = 0
            while saved == 0:
                if fname + '_%s.avi'%str(num).zfill(2) in all_files:
                    num += 1
                else:
                    out_path = 'saved_images/' + fname + '_%s.avi'%str(num).zfill(2)
                    saved =1
            
            self.video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_x, frame_y))
            self.buttonRecord.configure(text = 'Stop recording')
            
        # if it is already recording, stop it
        else:
            self.video_writer.release()
            self.video_writer = None
            self.buttonRecord.configure(text = 'Record video')
    
    
    def elli_refresh(self, value):
        self.elliptec = connect_to_elliptec(int(value[3]))
    
        if self.elliptec == None:        
            self.label_ell_status2.config(text = 'not connected', bg='red')
        else:
            self.label_ell_status2.config(text = 'connected', bg='lime')
    
    
    def rotate_elli(self, ang_rel):
        '''
        rotate elliptec to selected angle (relative angle)
        and set all necessary display options
        '''        
        # set elliptec
        ang_abs = angle_rel_to_abs(ang_rel)
        command = angle_to_ellocommand(ang_abs)
        self.elliptec.write(command)
        
        # write to textbox variables
        self.ell_var.set(ang_abs)
        self.ell_var2.set(ang_rel)
        
        # write to text
        self.label_elli_pos_abs.configure(text = 'Abs pos = %2.2f'%ang_abs)
        self.label_elli_pos_rel.configure(text = 'Rel pos = %2.2f'%ang_rel)
    

    def rotate_elli_abs(self):
        # get angles
        ang_abs = float(self.ell_var.get())
        ang_rel = angle_abs_to_rel(ang_abs)
        
        # call rotate function and defocuf from text field
        self.rotate_elli(ang_rel)
        self.buttonRotate.focus_set()
        
    
    def rotate_elli_rel(self):
        # get angles
        ang_rel = float(self.ell_var2.get())
        
        # call rotate function and defocuf from text field
        self.rotate_elli(ang_rel)
        self.buttonRotate2.focus_set()
        
        
    def rotate_elli_step(self, value):
        # get angle
        ang_rel = float(self.ell_var2.get())
        
        # change angle and write
        ang_rel += value
        ang_abs = angle_rel_to_abs(ang_rel)
        command = angle_to_ellocommand(ang_abs)
        self.elliptec.write(command)
        
        # recaluclate relative angle after change
        ang_rel = angle_abs_to_rel(ang_abs)
        
        # write to textbox variables
        self.ell_var.set(ang_abs)
        self.ell_var2.set(ang_rel)
        
        # write to text
        self.label_elli_pos_abs.configure(text = 'Abs pos = %2.2f'%ang_abs)
        self.label_elli_pos_rel.configure(text = 'Rel pos = %2.2f'%ang_rel)


    def image_move(self, event):
        print('x=%d, y=%d'%(event.x, event.y))
    
        
    def save_img(self):
        self.save_image = True

      
    def show_frame(self):
        global frame_x, frame_y, cap
        
        # get image from camera
        ret, frame = cap.read()
        if ret == False:
            frame = np.zeros((448, 800, 3), np.uint8)        
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #A
        
        # frame_x, frame_y - do usuniecia, ale sa jeszcze w video recorder
        frame_x = frame.shape[1]
        frame_y = frame.shape[0]
        
        # now, cv2image is what will be displayed
        # here are possible modifications
        if self.camera_image_type != RAW:
            if self.camera_map_counter < self.frame_map_switch: rel_ang = 90 - self.polrot
            if self.camera_map_counter >= self.frame_map_switch: rel_ang = 90 + self.polrot
            
            self.rotate_elli(rel_ang)
            # abs_ang = angle_rel_to_abs(rel_ang)
            # self.ell_var.set(str(abs_ang))
            # self.ell_var2.set(str(rel_ang))
            
            # command = angle_to_ellocommand(abs_ang)
            # self.elliptec.write(command)
            # self.label_elli_pos_abs.configure(text = 'Abs pos = %2.2f'%abs_ang)
            # self.label_elli_pos_rel.configure(text = 'Rel pos = %2.2f'%(rel_ang))
            
            self.camera_map_counter += 1
            if self.camera_map_counter == self.frame_map_switch*2: self.camera_map_counter = 0
        
        
        # overlay camera_draw as red channel with selected alpha if hold is on
        if self.camera_overlay > 0 or self.always_overlay > 0:
            zer = np.zeros((self.camera_draw.shape[0], self.camera_draw.shape[1])).astype(np.uint8)
            red_overlay = cv2.merge((self.camera_draw[:, :, 0], zer, zer))
            cv2image = cv2.addWeighted(cv2image, 1, red_overlay, 0.3, 0.0)

        # draw brush circle
        if self.mode_draw == 1:
            cv2image = cv2.circle(cv2image, (self.mouse_x, self.mouse_y-76), self.brush_size, (220, 80, 80), 2)
        
        # cv2image = cv2.flip(cv2image, 0)
        img = Image.fromarray(cv2image)
        self.imgtk = ImageTk.PhotoImage(image=img)
        # self.lmain.imgtk = imgtk
        # self.lmain.configure(image=imgtk)
        # self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=self.imgtk, anchor=CENTER)
        
        # record video
        if self.video_writer != None:
            self.video_writer.write(cv2image)
        
        if self.save_image == True:
            fname = self.path.get()
            if fname == '': fname = 'unnamed'
            all_files = os.listdir(os.getcwd() + '/saved_images/')
            saved = 0
            num = 0
            while saved == 0:
                if fname + '_%s.png'%str(num).zfill(2) in all_files:
                    num += 1
                else:
                    img.save('saved_images/' + fname + '_%s.png'%str(num).zfill(2))
                    print(fname + '_%s.png saved!'%str(num).zfill(2))
                    saved =1
                    
            self.save_image = False   


    def mod_source(self, value):
        global cap
        cap.release()
        cap = cv2.VideoCapture(int(value))
        print(value)
        
        
    def exit(self):
        global cap
        cap.release()
        root.quit()
        
       
    def initiate_projector_window(self):
        if self.projector_window == None:                
            # self.projector_window = ProjectorWindow(root)
            # self.app = ProjectorWindow(self.projector_window)
            self.projector_window = Toplevel(root)
            self.projector_window.title("Projector window - move to projector screen")
            self.projector_window.geometry("400x400")
            
    
    def close_projector_window(self):
        if self.projector_window != None:
            self.projector_window.destroy()
            self.projector_window = None
    
        
    def activate_projector_window(self):
        # initialize full screen mode
        self.projector_window.overrideredirect(True)
        self.projector_window.state("zoomed")
        # self.projector_window.activate()
        
        self.canvas_proj = Canvas(self.projector_window, width=1024, height=768, bg='black')
        self.canvas_proj.pack(side = LEFT)
        
        
    def refresh_projector_image(self):
        try:
            if self.mode_draw == 1:  # if drawing is on use camera draw and transform it into projector array
                im_out = cv2.warpPerspective(self.camera_draw, self.homomatrix, (1024, 768))
                self.projector_arr = im_out
            else:  # for now, just put an eagle there (maybe it should be black screen?)
                if projector_calib_c >= calib_dots_dim**2:  # if it is not calibrating
                    self.projector_arr = self.logo
            
            if self.hold_proj == 0:
                img = Image.fromarray(cv2.flip(self.projector_arr, 1))
            else:
                img = Image.fromarray(np.zeros((768, 1024, 3), np.uint8))                
            self.proj_imgtk = ImageTk.PhotoImage(image=img)
            self.canvas_proj.create_image(512, 384, image=self.proj_imgtk, anchor=CENTER)
        except:
            pass
        
        
# root window created. Here, that would be the only window, but
# you can later have windows within windows.
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
root.mainloop()
        
        
        
        
        
        
        
        

# save_image = False

# root = Tk()
# root.bind('<Escape>', lambda e: exit())

# source_frame = Frame(root)
# source_frame.grid(row=0, column=0, padx = padd)

# sv = StringVar()
# source_entry = Entry(source_frame, textvariable=sv)
# source_entry.pack()

# buttonSource = Button(source_frame, text = 'Change source', command = mod_source)
# buttonSource.pack()


# lmain = Label(root)
# lmain.grid(row=1, column=0, padx = padd)

# buttonSave = Button(root, text = 'Save', command = save_img)
# buttonSave.grid(row=2, column=0, padx = padd)

# path = StringVar()
# nameEntered = Entry(root, width = 15, textvariable = path)
# nameEntered.grid(row=3, column=0, padx = padd)

# show_frame()
# root.mainloop()