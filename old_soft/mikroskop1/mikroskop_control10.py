import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
from tkinter import *
from PIL import Image, ImageTk, ImageFont, ImageDraw
import os
import sys, ftd2xx as ftd
import numpy as np
import serial
import sys
import glob
import time
from threading import Thread
import datetime
from scipy.ndimage.filters import gaussian_filter, maximum_filter

width, height = 800, 600
bar_height = 100
# cap = cv2.VideoCapture(1)
padd = 20
frame_x = 0
frame_y = 0
elli_angle = 0
# self.elli_angle = 146.25 # 62.5  # this is absolute angle at which relative angle is 90
data_dir = 'd:/Katalog 1/Science/Projekt lasery/mikroskop/mikroskop_data/'


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

scale_um = {'X5': 500, 'X10': 200, 'X20': 100, 'X50': 50, 'X100': 20}
scale_px = {'X5': 200, 'X10': 160, 'X20': 160, 'X50': 200, 'X100': 160}

RAW = 0
MAP = 1
DYN = 2


# counters 
projector_calib_c = calib_dots_dim**2+1
    
    
def blur_crop(arr, sigma = 7, crop = 40):
    arr = gaussian_filter(arr, sigma=sigma)
    test = cv2.resize(arr, (arr.shape[1]+crop, arr.shape[0]+crop), interpolation = cv2.INTER_AREA)
    new = np.zeros(arr.shape, float)
    half_crop = int(crop/2)
    new = test[half_crop:-half_crop, half_crop:-half_crop]
    return new


def minimize_bisection(function, arguments = [], rng = [0, 1], jumps = 10):
    jump = abs(rng[1]-rng[0])/2  # current jump
    cp = (rng[0]+rng[1])/2  # current position
#     cr = function(cp)  # current result of the function
    for i in range(0, jumps):
        # calculate results on the left and right
        args = [cp-jump] + arguments
        rl = function(*args)
        args = [cp+jump] + arguments
        rr = function(*args)
#         print('jump = %d, cp = %f, rl=%f,rr=%f'%(i, cp, rl, rr))
        if rr < rl:
            cp += jump
        else:
            cp -= jump
        jump /= 2
    return cp


def estimate_gradient(img, sig = 300):
    blur = blur_crop(img, sigma = sig)
    return blur.max()/blur.min()


def check_factor(factor, image, grad):
    img_c = image.copy() - grad*factor + grad.mean()*factor
    return estimate_gradient(img_c)


def degradient(img, resize = 10, sig = 300, jumps = 8):
    sy = int(img.shape[0]/resize)
    sx = int(img.shape[1]/resize)
    resized = cv2.resize(img, (sx, sy), interpolation = cv2.INTER_AREA)
    grad = blur_crop(resized, sigma=int(sig/resize), crop = int(40/resize))
    
    factor = minimize_bisection(check_factor, arguments = [resized.copy(), grad], rng = [0, 4], jumps = jumps)
    # recalculate gradient using full unresized image
    grad = blur_crop(img, sigma=sig)
    return img.copy() - grad*factor + grad.mean()*factor
    

def advanced_map(img_minus, img_plus, sg = 4, degrad_size = 10, maxi = 2):
    # convert to grayscale
    arrPg = cv2.cvtColor(img_plus, cv2.COLOR_BGR2GRAY).astype(float)/255
    arrMg = cv2.cvtColor(img_minus, cv2.COLOR_BGR2GRAY).astype(float)/255
    
    # gauss if necessary
    if (sg>0):
        arrPg = gaussian_filter(arrPg, sigma=sg)
        arrMg = gaussian_filter(arrMg, sigma=sg)
    
    # degradient if necessary
    if (degrad_size>0):
        arrPg = degradient(arrPg, resize = degrad_size)
        arrMg = degradient(arrMg, resize = degrad_size)
        
    # maxi filter if necessary (good for eliminating birefringence)
    if maxi>0:
        arrPg = -maximum_filter(-arrPg, maxi)
        arrMg = -maximum_filter(-arrMg, maxi)
        
    # mapka
    mapka = arrPg-arrMg
    mapka = mapka-mapka.mean()
    
    # get domain threshold
    min1, max1 = get_minmax(get_distribution(mapka))
    ranger = max([abs(min1), max1])*0.3
    # np.save('temp.npy', mapka)
    
    return mapka, ranger
    
    
def get_distribution(mapka):
    flat_dif = mapka.copy()
    flat_dif = flat_dif.flatten()
    flat_dif.sort()
    return flat_dif


def get_minmax(flat_dif, percent = 0.01):   
    count = flat_dif.shape[0]

    minval = flat_dif[int(count*percent)]
    maxval = flat_dif[-int(count*percent)]

    return minval, maxval


def get_amap_parameter(string, name):
    string = string[string.find(name)+1+len(name):]
    string = string[:string.find(',')]
    return string   


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
        ax1.text(crd[0], crd[1], str(i), c = 'red')
        
    ax2.set_title('projector array')
    ax2.imshow(im_out, cmap = 'gray')
    for i, crd in enumerate(coords_prj):
        ax2.text(crd[0], crd[1], str(i), c = 'white')
        
    fig.savefig('calibration/calib_result.png', dpi = 400)
    
    return h


def angle_abs_to_rel(ang_abs, elli_angle):
    rel = elli_angle + 90 - ang_abs
    if rel > 360: rel -= 360
    if rel < 0: rel += 360
    return rel


def angle_rel_to_abs(ang_rel, elli_angle):
    absa = elli_angle - ang_rel + 90
    if absa > 360: absa -= 360
    if absa < 0: absa += 360
    return absa
    

def add_border(img, border, color = (0, 0, 0)):
    # initialize new, bigger image
    new_img = np.zeros((768+2*border, 1024+2*border, 3), np.uint8)
    new_img[:,:] = color
    
    # paste old img in the middle and resize to match original size
    new_img[border:768+border, border:1024+border] = img
    new_img = cv2.resize(new_img, (1024, 768))
    
    return new_img


def from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb  


def connect_to_grbl(com_i, timeout = 3):
    ser = serial.Serial()
    ser.baudrate = 115200
    ser.port='COM%d'%com_i
    ser.timeout = timeout
    
    try:
        ser.open()    
        resp = ser.read(32)
        if resp == b"\r\nGrbl 0.9j ['$' for help]\r\n": 
            print('Connected to grbl on COM%d'%com_i)
            return ser
        else:
            print('COM%d is not grbl')
            return None
    except:
        print('Device on COM%d is not available'%com_i)
        return None
    
    
def send_to_grbl(ser, command, maxlen = 64):
    ser.write(command)
    full_response = b''
    resp = b''
    i = 0
    while (resp != b'ok\r\n') or (i > maxlen):
        resp = ser.readline()
        full_response += resp
        i += 1
        
    return full_response
    ser.write(command)
    full_response = b''
    resp = b''
    while (resp != b'ok\r\n'):
        resp = ser.readline()
        full_response += resp
        
    return full_response


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


def cmap_array(arr, cmap, rng = 1, middle = 0):
     return cmap((arr/rng+1-middle/rng)/2)[:,:,:3]


def domain_eater_out(point, size, eat_dir, eat_stage):
    # draw circle on empty array
    img = np.zeros((768, 1024), np.uint8)
    img = cv2.circle(img, point, size, (255), -1)
    eat_dir -= 90
    eat = (np.sin(eat_dir*np.pi/180), np.cos(eat_dir*np.pi/180))
    
    # draw eater
#     eat_stage = 1-eat_stage
    ex = int(-2*size*eat_stage*eat[0] + point[0])
    ey = int(-2*size*eat_stage*eat[1] + point[1])
    eater = cv2.circle(np.zeros((768, 1024), np.uint8), (ex, ey), size, (255), -1)
#     img = img - (img&(~eater))
    img = img&eater
    
    return cv2.merge((img, img, img))


def domain_eater_in(point, size, eat_dir, eat_stage):
    # draw circle on empty array
    img = np.zeros((768, 1024), np.uint8)
    img = cv2.circle(img, point, size, (255), -1)
    eat_dir -= 90
    eat = (np.sin(eat_dir*np.pi/180), np.cos(eat_dir*np.pi/180))
    # draw eater moved at full size in opposite of eat direction - 
    eat_stage = 1-eat_stage
    ex = int((2*size*eat_stage)*eat[0] + point[0])
    ey = int((2*size*eat_stage)*eat[1] + point[1])
    eater = cv2.circle(np.zeros((768, 1024), np.uint8), (ex, ey), size, (255), -1)
#     img = img - (img&(~eater))
    img = img-eater
    
    return cv2.merge((img, img, img))


def connect_to_thermal(com_i, timeout = 3):
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port='COM%d'%com_i
    ser.timeout = 3

    try:
        ser.open()    
        time.sleep(4)
        command = b'WHO 1\n'
        ser.write(command)
        resp = ser.readline()
        formatted = resp.decode('utf-8')
        if formatted == 'thermal\r\n': 
            print('Connected to thermal on COM%d'%com_i)
            return ser
        else:
            print('COM%d is not thermal')
            return None
    except:
        print('Device on COM%d is not available'%com_i)
        return None

    
def thermal_get_temps(ser):
    command = b'GET 1\n'
    ser.write(command)
    resp = ser.readline()
    formatted = resp.decode('utf-8')
    if len(formatted) > 0:
        try:
            c_str = formatted.split(',')[0]
            current = float(c_str[c_str.find('=')+1:])
            s_str = formatted.split(',')[1]
            t_set = float(s_str[s_str.find('=')+1:])
            return current, t_set
        except:
            return -1, -1
    else:
        return -1, -1


def thermal_set_temps(ser, temperature):    
    command = 'SET %d'%temperature
    command = bytes(command, 'utf-8')
    ser.write(command)


class CamReader(Thread):
    '''runs on separate thread and read image from camera'''
    def __init__(self, master):
        super().__init__()
        self.master = master

        self.cap = cv2.VideoCapture(1)
        self.frame = np.zeros((448, 800, 3), np.uint8)
        
        fontpath = 'OpenSans-Regular.ttf'
        self.font11 = ImageFont.truetype(fontpath, 34)
        self.frame_is_new = True
        
        
    def read_frame(self):
        ret, self.frame = self.cap.read()
        if ret == False:
            self.frame = generate_random_image()
        self.frame_is_new = True            
        self.master.after(100, self.read_frame)
        
        
    def get_frame(self):
        if self.frame_is_new == True:
            self.frame_is_new = False
            return self.frame, True
        else:
            return self.frame, False
            
        
        
    def get_frame_shape(self):
        return self.frame.shape

    
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
        self.border = 90  # border added to projector image to counter the fact, that camera does not see full projected laser image
        self.loop_delay_target = 30  # it will try to match this delay (todo: actully implement this)
        
        # non-loadable variables (are modified inside app)
        self.interaction_mode = 'none'
        self.erosion = 0
        self.erosion_counter = 0
        self.erosion_delay = 1
        self.hold_proj = 0   
        self.camera_overlay = 0
        self.brush_size = 50
        self.always_overlay = 0
        self.camera_image_type = 'RAW'
        self.camera_map_counter = 0
        self.frame_map_switch = 20
        self.polrot = 10
        self.show_parameters = 1  # determines whether to add information on camera image before display (scale, temp and stuff)
        self.current_obiektyw = 'X10'
        self.loop_cnt = 0  # loop counter, for displayng loop informations
        self.loop_time = 0
        self.time_prev = 0
        self.save_data = False  # if True saves raw data during show_frame
        self.rec_data = False  # if True saves raw frame with specified freqency
        self.rec_data_cnt = 0  # used to count the save freq during show_frame
        self.prev_grayframe = np.zeros((448, 800), float)  # this stores camera frame as grayscale for differential polarization map
        self.diff_cmap = plt.get_cmap('PiYG')
        self.calib_grad = np.load('calibration_gradient.npy')
        self.calib_factor = 1.2
        self.map_rng = 0.01
        self.map_gauss = 0
        self.cv2image_const = np.zeros((448, 800, 3), np.uint8)  # this is for 'MAP'
        self.pt_eat_dir = 0  # for point laser: eat directin
        # self.pt_eat_speed = 3  # for point laser: how much eat_factor increases every mainloop
        # self.pt_size = 100  # how big is circle made by point
        self.pt_eat_stage = 0  # current eat stage
        self.pt_point = (400, 500)
        self.camera_point_draw = np.zeros((448, 800, 3), np.uint8)  # this is space where points are drawn
        
        
        self.cam_reader = CamReader(self.master)
        self.cam_reader.read_frame()
        self.elliptec = None
        self.video_writer = None
        self.projector_window = None
        self.projector_arr = np.zeros((768, 1024, 3), np.uint8)
        self.grbl = None
        self.grblX = 0.0
        self.grblY = 0.0
        self.grblZ = 0.0
        self.B1_was_pressed = False
        self.thrm = None
        self.should_set_temp = False  # when true, refresh_temperature function (running on separate thread) will set the temperature
        self.t_cur = -1  # current temperature (set by refresh_temperature)
        
        # self.projector_arr = cv2.circle(self.projector_arr, (500, 300), 50, (255, 255, 255), -1)
       
        fontpath = 'OpenSans-Regular.ttf'
        self.font11 = ImageFont.truetype(fontpath, 11)
        
        # this is grayscale image that overlays the camera and it is projected through homography to projector screen
        self.camera_draw = np.zeros((448, 800, 3), np.uint8)
        self.camera_draw = cv2.circle(self.camera_draw, (500, 300), 50, (255, 255, 255), -1)
        
        # matrix for homography (is set during calibration)
        self.homomatrix = np.zeros((3,3))        
        self.image_baseline = np.zeros((600, 800, 3), np.uint8)
        
        # logo for fun/testing
        self.logo = cv2.imread('idle_patterns/unilogo.png')
        self.rownosc = cv2.imread('idle_patterns/rownosc.png')
        self.default_image = self.logo
        self.projector_arr = self.default_image
        
        # this is mouse xy relative to camera canvas
        self.mouse_x = 0
        self.mouse_y = 0

        # with that, we want to then run init_window, which doesn't yet exist
        self.init_window()
        self.main_loop()
        self.elli_refresh('COM1')
        self.read_extvars()
        self.thr_refresh_temp()
    
    
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
        fileMenu.add_command(label="Reload variables", command=self.read_extvars)
        
        idleMenu = Menu(root, tearoff=False)
        fileMenu.add_cascade(label="Idle pattern", menu=idleMenu)
        idleMenu.add_command(label = 'Logo', command = lambda: self.change_default_image(self.logo))
        idleMenu.add_command(label = 'Rectangles', command = lambda: self.change_default_image(self.rownosc))
        
        objectiveMenu = Menu(root, tearoff=False)
        # fileMenu.add_cascade(label="Change objective", menu=objectiveMenu)
        objectiveMenu.add_command(label = 'X5', command = lambda: self.change_objective('X5'))
        objectiveMenu.add_command(label = 'X10', command = lambda: self.change_objective('X10'))
        objectiveMenu.add_command(label = 'X20', command = lambda: self.change_objective('X20'))
        objectiveMenu.add_command(label = 'X50', command = lambda: self.change_objective('X50'))
        objectiveMenu.add_command(label = 'X100', command = lambda: self.change_objective('X100'))
        
        fileMenu.add_command(label="Exit", command=self.exit)
        menu.add_cascade(label="File", menu=fileMenu)
        menu.add_cascade(label="Change objective", menu=objectiveMenu)
        
        # keybindings
        self.master.bind('[', self.brush_dec)
        self.master.bind(']', self.brush_inc)
        self.master.bind('<Escape>', self.esc_key_btn)
        self.master.bind('<Shift-Key-C>', self.key_shiftc)
        
        
        #  -- FIRST COLUMN --
        # main image        
        # self.lmain = Label(root)
        # self.lmain.grid(row=row, column=col, padx = padd)
        self.canvas = Canvas(width=800-4, height=600, bg='black')
        self.canvas.grid(row=row, column=col, padx = padd)
        # self.canvas.bind("<Double-Button-1>", self.image_move)  # for future interactions        
        self.canvas.bind("<ButtonPress-1>", self.mouse_motionB1)
        # self.canvas.bind("<ButtonRelease-1>", self.cam_btn_release)
        self.canvas.bind('<B1-Motion>', self.mouse_motionB1)
        self.canvas.bind('<Motion>', self.mouse_motion)
        self.canvas.bind('<ButtonRelease-1>', self.mouse_B1_release)
        row += 1
        
        # IMAGE CONTROLS FRAME
        current_frame = Frame(root)
        current_frame.grid(row=row, column=col, padx = padd)
        row += 1
        self.label = Label(current_frame, text = 'Display controls: ')
        self.label.pack(side =  LEFT)
        
        # source_var = StringVar(self.master)
        # source_var.set(0) # default value
        # self.source_menu = OptionMenu(current_frame, source_var, *list_cameras(), command = self.mod_source)
        # self.source_menu.pack(side = LEFT)
        
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
        self.btnPoint = Button(current_frame, text = 'Point', command = self.btn_point)
        self.btnPoint.pack(side = LEFT)             
        self.btnErosion = Button(current_frame, text = 'Erosion', command = self.btn_erosion)
        self.btnErosion.pack(side = LEFT) 
        
        eros_var = StringVar(self.master)
        eros_var.set(1) # default value
        self.mnErospeed = OptionMenu(current_frame, eros_var, *[1, 2, 3, 4, 5, 8, 10, 15, 20], command = self.mn_set_erospeed)
        self.mnErospeed.pack(side = LEFT)
        
        self.btnClear = Button(current_frame, text = 'Clear', command = self.btn_clear_camdraw)
        self.btnClear.pack(side = LEFT)
        
        # IMAGE TYPE FRAME
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
        fms_var.set(30) # default value
        self.mnFMS = OptionMenu(current_frame, fms_var, *[15, 20, 25, 30, 40, 50, 100, 200, 500], command = self.mn_set_fms)
        self.mnFMS.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Rotation: ')
        self.label.pack(side =  LEFT)
        polrot_var = StringVar(self.master)
        polrot_var.set(10) # default value
        self.mnPolrot = OptionMenu(current_frame, polrot_var, *[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15], command = self.mn_set_polrot)
        self.mnPolrot.pack(side = LEFT)
        
        
        # IMAGE TYPE FRAME 2 (ADVANCED MAPPING)
        current_frame = Frame(root)
        current_frame.grid(row=row, column=col, padx = padd)
        row += 1
        self.label = Label(current_frame, text = 'Advanced type controls: ')
        self.label.pack(side =  LEFT)
        
        self.btnAdvMap = Button(current_frame, text = 'Make here', command = self.btn_adv_map)
        self.btnAdvMap.pack(side = LEFT)
        
        self.advMapOptions = StringVar()
        self.advMapOptions.set('G=4, M=5, Df=10, Tf=0.3')
        self.nameEntered = Entry(current_frame, width = 20, textvariable = self.advMapOptions)
        self.nameEntered.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'G - gauss sigma, M - maxfilter sigma, Df - degradient resize factor, Tf - domian threshold factor')
        self.label.pack(side =  LEFT)
        

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
        
        self.buttonsaveData = Button(current_frame, text = 'Save frame', command = self.save_data_button)
        self.buttonsaveData.pack(side = LEFT)
        
        self.buttonrecData = Button(current_frame, text = 'Record frames', command = self.rec_data_button)
        self.buttonrecData.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Freq: ')
        self.label.pack(side =  LEFT)
        self.savedata_freq_var = StringVar(self.master)
        self.savedata_freq_var.set(10) # default value
        self.mnSDfreq = OptionMenu(current_frame, self.savedata_freq_var, *[1, 2, 5, 8, 10, 20, 50, 100])
        self.mnSDfreq.pack(side = LEFT)
        
        # LOOP TIME INFO
        current_frame = Frame(root)
        current_frame.grid(row=row, column=col, padx = padd)
        row += 1
        
        self.label_looptime = Label(current_frame, text = 'Loop time: 0.0000')
        self.label_looptime.pack(side =  LEFT)
        
        
        # TEMPORARY CONTROLS           
        current_frame = Frame(root)
        current_frame.grid(row=row, column=col, padx = padd)
        row += 1        
        
        self.label_elli_pos_abs = Label(current_frame, text = 'TEMPORARY CONTROLS')
        self.label_elli_pos_abs.config(font=('Segoe UI', 16))
        self.label_elli_pos_abs.pack(side =  LEFT)
                   
        current_frame = Frame(root)
        current_frame.grid(row=row, column=col, padx = padd)
        row += 1    

        self.label = Label(current_frame, text = 'Point action: ')
        self.label.pack(side =  LEFT)
        self.ptact_var = StringVar(self.master)
        self.ptact_var.set('eat in') # default value
        self.mnSDfreq = OptionMenu(current_frame, self.ptact_var, *['eat in', 'eat out'])
        self.mnSDfreq.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Eat speed: ')
        self.label.pack(side =  LEFT)
        self.eatspeed_var = StringVar(self.master)
        self.eatspeed_var.set(1) # default value
        self.mnSDfreq = OptionMenu(current_frame, self.eatspeed_var, *[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 4, 5, 10])
        self.mnSDfreq.pack(side = LEFT)
        
        self.label = Label(current_frame, text = 'Eat square: ')
        self.label.pack(side =  LEFT)
        self.eatsqr_var = StringVar(self.master)
        self.eatsqr_var.set(0) # default value
        self.mnSDfreq = OptionMenu(current_frame, self.eatsqr_var, *[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        self.mnSDfreq.pack(side = LEFT)       
             
        
        #  -- SECOND COLUMN --
        row = 0
        col = 1
        
        # control_frame (main frame that holds multiple control frames for elliptec, g-code and others)
        control_frame = Frame(root)
        control_frame.grid(row=row, column=col, padx = padd)
        row += 1
        ctrl_pad = 4
        
        # ELLIPTEC SUBFRAME
        eli_frame = Frame(control_frame)
        eli_frame.grid(row=0, column=0, padx = ctrl_pad, pady = ctrl_pad)
        self.create_elliptec_frame(eli_frame)
        
        # GRBL SUBFRAME
        grbl_frame = Frame(control_frame)
        grbl_frame.grid(row=1, column=0, padx = ctrl_pad, pady = ctrl_pad)
        self.create_grbl_frame(grbl_frame)
        
        # THERMAL SUBFRAME
        thermal_frame = Frame(control_frame)
        thermal_frame.grid(row=2, column=0, padx = ctrl_pad, pady = ctrl_pad)
        self.create_thermal_frame(thermal_frame)
        
        
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
        
        
    def create_elliptec_frame(self, eli_frame):       
        # elliptec frame name
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.label_elli_pos_abs = Label(cur_frame, text = 'ELLIPTEC CONTROL')
        self.label_elli_pos_abs.config(font=('Segoe UI', 16))
        self.label_elli_pos_abs.pack(side =  LEFT)
        
        # elliptec position frames
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.label_elli_pos_abs = Label(cur_frame, text = 'Abs pos = %2.2f'%0)
        self.label_elli_pos_abs.config(font=('Segoe UI', 13))
        self.label_elli_pos_abs.pack(side =  LEFT)
        
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y) 
        self.label_elli_pos_rel = Label(cur_frame, text = 'Rel pos = %2.2f'%0)
        self.label_elli_pos_rel.config(font=('Segoe UI', 13))
        self.label_elli_pos_rel.pack(side =  LEFT)
        
        
        # elliptec connection frame
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)
        
        ell_com_var = StringVar(self.master)
        ell_com_var.set('COM3') # default value
        self.ell_com_menu = OptionMenu(cur_frame, ell_com_var, *serial_ports(), command = self.elli_refresh)
        self.ell_com_menu.pack(side = LEFT)
        
        self.label_ell_status = Label(cur_frame, text = 'Elliptec status: ')
        self.label_ell_status.pack(side =  LEFT)
        
        self.label_ell_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_ell_status2.pack(side =  LEFT)       
        

        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        self.label_ell1 = Label(cur_frame, text = 'Rotate relative: ')
        self.label_ell1.pack(side =  LEFT)        
        self.ell_var2 = StringVar()
        self.ell_var2.set('90.0')
        self.nameEntered2 = Entry(cur_frame, width = 15, textvariable = self.ell_var2)
        self.nameEntered2.pack(side = LEFT, fill = X)        
        self.buttonRotate2 = Button(cur_frame, text = 'Rotate', command = self.rotate_elli_rel)
        self.buttonRotate2.pack(side = LEFT)        
        self.button = Button(cur_frame, text = 'Set 90 here', command = self.elli_set_zero)
        self.button.pack(side = LEFT)
        
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y) 
        self.buttonRotateStepM1 = Button(cur_frame, text = '-10', command = lambda: self.rotate_elli_step(-10))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-5', command = lambda: self.rotate_elli_step(-5))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-1', command = lambda: self.rotate_elli_step(-1))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-0.25', command = lambda: self.rotate_elli_step(-0.25))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+0.25', command = lambda: self.rotate_elli_step(0.25))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+1', command = lambda: self.rotate_elli_step(1))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+5', command = lambda: self.rotate_elli_step(5))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+10', command = lambda: self.rotate_elli_step(10))
        self.buttonRotateStepP1.pack(side = LEFT)
    

    def create_grbl_frame(self, frame):
        # grbl frame name
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab = Label(cur_frame, text = 'GRBL CONTROL')
        self.lab.config(font=('Segoe UI', 16))
        self.lab.pack(side =  LEFT)
        
        # grbl position frames
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab_grblpos = Label(cur_frame, text = 'X = %2.4f, Y = %2.4f'%(0,0))
        self.lab_grblpos.config(font=('Segoe UI', 13))
        self.lab_grblpos.pack(side =  LEFT)
        
        
        # grbl set/move connection frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        # set frame
        set_frame = Frame(cur_frame)
        set_frame.grid(row=0, column=0)
        
        sze_frame = Frame(set_frame)
        sze_frame.grid(row=0, column=0)     
        lab = Label(sze_frame, text = 'Step size:')
        lab.config(font=('Segoe UI', 9))
        lab.pack(side =  LEFT)   
        self.grbl_size = StringVar()
        self.grbl_size.set('0.2')
        self.grbl_size_entry = Entry(sze_frame, width = 15, textvariable = self.grbl_size)
        self.grbl_size_entry.pack(side = LEFT, fill = X)
        
        spd_frame = Frame(set_frame)
        spd_frame.grid(row=1, column=0)         
        lab = Label(spd_frame, text = 'Step speed:')
        lab.config(font=('Segoe UI', 9))
        lab.pack(side =  LEFT)    
        self.grbl_speed = StringVar()
        self.grbl_speed.set('10.0')
        self.grbl_speed_entry = Entry(spd_frame, width = 15, textvariable = self.grbl_speed)
        self.grbl_speed_entry.pack(side = LEFT, fill = X)
        
        btn_frame = Frame(set_frame)
        btn_frame.grid(row=2, column=0)
        btn = Button(btn_frame, text = 'Set params')
        btn.pack(side = LEFT)
        btn = Button(btn_frame, text = 'Reset zero', command = lambda: self.grbl_command('G10 P0 L20 X0 Y0 Z0\n'))
        btn.pack(side = LEFT)
        
        # move frame
        mov_frame = Frame(cur_frame)
        mov_frame.grid(row=0, column=1)
        
        bframe = Frame(mov_frame)
        bframe.grid(row=0, column=0)
        btn = Button(bframe, text = '\\')
        btn.pack(side = LEFT)
        btn = Button(bframe, text = 'U', command = lambda: self.grbl_command('G21 G91 G1 Y%s F%s\n'%(self.grbl_size.get(), self.grbl_speed.get())))
        btn.pack(side = LEFT)
        btn = Button(bframe, text = '/')
        btn.pack(side = LEFT)
        
        bframe = Frame(mov_frame)
        bframe.grid(row=1, column=0)
        btn = Button(bframe, text = 'L', command = lambda: self.grbl_command('G21 G91 G1 X-%s F%s\n'%(self.grbl_size.get(), self.grbl_speed.get())))
        btn.pack(side = LEFT)
        btn = Button(bframe, text = '.')
        btn.pack(side = LEFT)
        btn = Button(bframe, text = 'R', command = lambda: self.grbl_command('G21 G91 G1 X%s F%s\n'%(self.grbl_size.get(), self.grbl_speed.get())))
        btn.pack(side = LEFT)
        
        bframe = Frame(mov_frame)
        bframe.grid(row=2, column=0)
        btn = Button(bframe, text = '/')
        btn.pack(side = LEFT)
        btn = Button(bframe, text = 'D', command = lambda: self.grbl_command('G21 G91 G1 Y-%s F%s\n'%(self.grbl_size.get(), self.grbl_speed.get())))
        btn.pack(side = LEFT)
        btn = Button(bframe, text = '\\')
        btn.pack(side = LEFT)
        
        # connection frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        grbl_com_var = StringVar(self.master)
        grbl_com_var.set('COM3') # default value
        self.grbl_com_menu = OptionMenu(cur_frame, grbl_com_var, *serial_ports(), command = self.grbl_refresh)
        self.grbl_com_menu.pack(side = LEFT)
        
        self.label_grbl_status = Label(cur_frame, text = 'GRBL status: ')
        self.label_grbl_status.pack(side =  LEFT)
        
        self.label_grbl_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_grbl_status2.pack(side =  LEFT) 


    def create_thermal_frame(self, frame):
        # grbl frame name
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab = Label(cur_frame, text = 'THERMAL CONTROL')
        self.lab.config(font=('Segoe UI', 16))
        self.lab.pack(side =  LEFT)
        
        # grbl position frames
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)        
        self.lab_tempcur = Label(cur_frame, text = 'T = %3.2f °C'%0)
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
        self.set_temp_btn = Button(cur_frame, text = 'Set', command = self.should_set_temp_action)
        self.set_temp_btn.pack(side = LEFT)
        
        # connection frame
        cur_frame = Frame(frame)
        cur_frame.pack(fill = Y)
        
        grbl_com_var = StringVar(self.master)
        grbl_com_var.set('COM3') # default value
        self.grbl_com_menu = OptionMenu(cur_frame, grbl_com_var, *serial_ports(), command = self.thrm_refresh)
        self.grbl_com_menu.pack(side = LEFT)
        
        self.label_thrm_status = Label(cur_frame, text = 'THERMAL status: ')
        self.label_thrm_status.pack(side =  LEFT)
        
        self.label_thrm_status2 = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_thrm_status2.pack(side =  LEFT) 
        
    
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
                    
            # draw point 
            if self.interaction_mode == 'point' and self.hold_proj != 1:
                eater = self.ptact_var.get()
                if self.pt_eat_stage < 1:
                    self.pt_eat_stage += (float(self.eatspeed_var.get())/100)*(1+self.pt_eat_stage)**int(self.eatsqr_var.get())
                    if eater == 'eat in':
                        pointed_draw = domain_eater_in(self.pt_point, self.brush_size, self.pt_eat_dir, self.pt_eat_stage)
                    if eater == 'eat out':
                        pointed_draw = domain_eater_out(self.pt_point, self.brush_size, self.pt_eat_dir, self.pt_eat_stage)
                if self.pt_eat_stage >= 1:
                    pointed_draw = np.zeros((768, 1024, 3), np.uint8)
                self.camera_point_draw = pointed_draw
                    
        
        if projector_calib_c < calib_dots_dim**2:
            self.proj_calib()
            
            # if projector_calib_c == 0:
                # self.homomatrix = get_homography_matrix()
        
        
        # timing stuff
        self.loop_cnt += 1
        time_now = time.time()
        self.loop_time += time_now-self.time_prev
        self.time_prev = time_now
        
        if self.loop_cnt >= 10:
            self.loop_cnt = 0            
            self.label_looptime.config(text = 'Main loop time: %f ms (%f Hz), ldp = %d'%(self.loop_time/10*1000, 1/(self.loop_time/10), self.loop_delay_target))
            if self.loop_time/10 < 0.1:
                self.label_looptime.config(fg='green')
            else:
                self.label_looptime.config(fg='red')
            
            self.loop_time = 0
        
        self.master.after(self.loop_delay_target, self.main_loop)
        
        
    def grbl_get_position(self):
        if self.grbl != None:
            resp = send_to_grbl(self.grbl, b'?\n')
            self.grblX, self.grblY, self.grblZ = [float(f) for f in resp[resp.find('WPos:')+5:resp.find('>\\r\\nok')].split(',')]
            self.lab_grblpos.config(text = 'X = %2.4f, Y = %2.4f'%(self.grblX, self.grblY))
            
            
    def grbl_command(self, command):
        print(command)
        if self.grbl != None:
            resp = send_to_grbl(self.grbl, bytes(command, encoding = 'utf-8'))
            resp = str(send_to_grbl(self.grbl, b'?\n'))
            self.grblX, self.grblY, self.grblZ = [float(f) for f in resp[resp.find('WPos:')+5:resp.find('>\\r\\nok')].split(',')]
            self.lab_grblpos.config(text = 'X = %2.4f, Y = %2.4f'%(self.grblX, self.grblY))
           
    
    def change_objective(self, obj):
        self.current_obiektyw = obj
     
        
    def read_extvars(self):
        f = open('external_variables.txt', 'r')
        cnt = f.read().split('\n')
        f.close()

        extvars = {}
        for i in range(0, len(cnt)):
            cnt[i] = cnt[i].replace(' ', '')
            comment = cnt[i].find('#')
            if comment >= 0:
                cnt[i] = cnt[i][:comment]

            if len(cnt[i]) > 0:
                name, val = cnt[i].split('=')
                extvars[name] = val
                
        # set the variables
        # self.pulseOff = extvars['pulseOff']
        # self.pulseOn = extvars['pulseOn']
        self.border = int(extvars['border'])
        self.elli_angle = float(extvars['elli_angle'])
        self.loop_delay_target = int(extvars['loop_delay_target'])
        self.calib_factor = float(extvars['calib_factor'])
        self.map_rng = float(extvars['map_rng'])
        self.map_gauss = int(extvars['map_gauss'])
        
    
    def change_default_image(self, img):
        self.default_image = img
    

    def mn_set_polrot(self, value):
        '''this sets the polarization rotation angle for autorotation'''
        self.polrot = int(value)
        self.camera_map_counter = 0
    

    def mn_set_fms(self, value):
        '''this sets the polarization rotation frequency for autorotation'''
        self.frame_map_switch = int(value)
        self.camera_map_counter = 0
    

    def btn_icnt_raw(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.btnIcntRaw.config(relief=SUNKEN)
        self.camera_image_type = 'RAW'


    def btn_icnt_dyn(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.btnIcntDyn.config(relief=SUNKEN)
        self.camera_image_type = 'DYN'


    def btn_icnt_map(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.btnIcntMap.config(relief=SUNKEN)
        self.camera_image_type = 'MAP'
    
    
    def btn_adv_map(self):
        self.btnIcntDyn.config(relief=RAISED)
        self.btnIcntRaw.config(relief=RAISED)
        self.btnIcntMap.config(relief=RAISED)
        self.camera_image_type = 'AMAP'
        self.camera_map_counter = 0
    
    
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
        to_set = 'none'
        self.btnDraw.config(relief=RAISED)
        self.btnPoint.config(relief=RAISED)
        if self.interaction_mode == 'draw':
            to_set = 'none'
            self.btnDraw.config(relief=RAISED)
        if self.interaction_mode != 'draw':
            to_set = 'draw'            
            self.btnDraw.config(relief=SUNKEN)
        self.interaction_mode = to_set
        
        
    def btn_point(self):
        to_set = 'none'
        self.btnDraw.config(relief=RAISED)
        self.btnPoint.config(relief=RAISED)
        if self.interaction_mode == 'point':
            to_set = 'none'
            self.btnPoint.config(relief=RAISED)
        if self.interaction_mode != 'point':
            to_set = 'point'            
            self.btnPoint.config(relief=SUNKEN)
        self.interaction_mode = to_set
        
        
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
        self.pt_eat_stage = 1  # also clears point patterns
    
    
    def key_shiftc(self, event):
        self.btn_clear_camdraw()
    
 
    def mouse_motionB1(self, event):
        x = event.x
        y = event.y - 76 + 50  # to account for bigger canvas than camera image
        
        if self.interaction_mode == 'draw':
            self.draw_oncamera(x, y)
            
        if self.interaction_mode == 'point':
            if self.B1_was_pressed == False:
                self.B1_pressed_at = (x, y)
                self.B1_was_pressed = True
            
        # sets the mouse coords for drawing brush circle
        self.mouse_x = event.x
        self.mouse_y = event.y


    def mouse_motion(self, event):
        '''
        (only when the button is not pressed)
        '''
        self.mouse_x = event.x
        self.mouse_y = event.y
    
    
    def mouse_B1_release(self, event):
        if self.interaction_mode == 'point' and self.B1_was_pressed == True:
            self.create_point(self.B1_pressed_at[0], self.B1_pressed_at[1])
            x = event.x - self.B1_pressed_at[0]
            y = -(event.y - self.B1_pressed_at[1])
            self.pt_eat_dir = np.arctan(y/(x+0.00001))*180/np.pi
            if self.pt_eat_dir < 0: self.pt_eat_dir += 180
            if y < 0: self.pt_eat_dir += 180
            
            print('x = %d, y = %d, angle = %d'%(int(x), int(y), int(self.pt_eat_dir)))
            
            
        self.B1_was_pressed = False    
    
 
    def create_point(self, x, y):
        self.pt_point = (x, y)
        self.pt_eat_stage = 0
    
    
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
        if self.frame_counter == 10:
            # ret, frame = cap.read()
            frame = self.cam_reader.frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            if projector_calib_c > -1:  # meaining its done with the baseline                
                np.save('calibration/num%d'%projector_calib_c, frame)
                print('saved calibration/num%d'%projector_calib_c)
            else:
                np.save('calibration/baseline', frame)
                print('saved baseline')
        
            projector_calib_c += 1
            self.frame_counter = 0
        else:
            self.frame_counter += 1
         
        
    def record_video(self):
        # start recording new video
        if self.video_writer == None:
            # get the correct filename
            fname = self.path.get()
            if fname == '': fname = 'unnamed'
            all_files = os.listdir(data_dir + '/saved_images/')
            saved = 0
            num = 0
            while saved == 0:
                if fname + '_%s.avi'%str(num).zfill(2) in all_files:
                    num += 1
                else:
                    out_path = data_dir + 'saved_images/' + fname + '_%s.avi'%str(num).zfill(2)
                    saved =1
                    
            frame_y, frame_x = self.cam_reader.get_frame_shape()[:2]
            self.video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_x, frame_y + bar_height))
            # self.buttonRecord.configure(text = 'Stop recording')
            self.buttonRecord.configure(relief = SUNKEN, bg=from_rgb((245, 150, 150)))
            
        # if it is already recording, stop it
        else:
            self.video_writer.release()
            self.video_writer = None
            # self.buttonRecord.configure(text = 'Record video')
            self.buttonRecord.configure(relief = RAISED, bg=from_rgb((240, 240, 237)))
    
    
    def rec_data_button(self):
        if self.rec_data == False:
            self.rec_data = True
            self.buttonrecData.configure(relief = SUNKEN, bg=from_rgb((245, 150, 150)))
        else:
            self.rec_data = False
            self.buttonrecData.configure(relief = RAISED, bg=from_rgb((240, 240, 237)))
            
        
    def save_data_button(self):
        self.save_data = True
    
    
    def elli_refresh(self, value):
        self.elliptec = connect_to_elliptec(int(value[3]))
    
        if self.elliptec == None:        
            self.label_ell_status2.config(text = 'not connected', bg='red')
        else:
            self.label_ell_status2.config(text = 'connected', bg='lime')
    
    
    def grbl_refresh(self, value):
        self.grbl = connect_to_grbl(int(value[3]))
    
        if self.grbl == None:        
            self.label_grbl_status2.config(text = 'not connected', bg='red')
        else:
            self.label_grbl_status2.config(text = 'connected', bg='lime')
    
    
    def thrm_refresh(self, value):
        self.thrm = connect_to_thermal(int(value[3]))
    
        if self.thrm == None:        
            self.label_thrm_status2.config(text = 'not connected', bg='red')
        else:
            self.label_thrm_status2.config(text = 'connected', bg='lime')
    
    
    def thr_refresh_temp(self):
        # Call work function
        self.t1 = Thread(target=self.refresh_temperature)
        self.t1.start()
        
    
    def should_set_temp_action(self):
        self.should_set_temp = True
        self.esc_key_btn(1)
    
    
    def refresh_temperature(self):
        '''this function handles all communication with thermal'''
        if self.thrm != None:
            # read the temperature
            self.t_cur, t_set = thermal_get_temps(self.thrm)
            if self.t_cur != -1:
                self.lab_tempcur.config(text = 'T = %3.2f °C'%self.t_cur)
            else:
                print('Thermal was silent')
                
            # if necessary, set new target temperature
            if self.should_set_temp == True:
                temp = int(self.set_temp_var.get())
                thermal_set_temps(self.thrm, temp)
                self.should_set_temp = False
                print('Setting temperature to %d'%temp)
                
        self.master.after(1000, self.thr_refresh_temp)
                       
    
    def rotate_elli(self, ang_rel):
        '''
        rotate elliptec to selected angle (relative angle)
        and set all necessary display options
        '''        
        # set elliptec
        ang_abs = angle_rel_to_abs(ang_rel, self.elli_angle)
        command = angle_to_ellocommand(ang_abs)
        self.elliptec.write(command)
        
        # write to textbox variables
        # self.ell_var.set(ang_abs)
        self.ell_var2.set(ang_rel)
        
        # write to text
        self.label_elli_pos_abs.configure(text = 'Abs pos = %2.2f'%ang_abs)
        self.label_elli_pos_rel.configure(text = 'Rel pos = %2.2f'%ang_rel)
            
    
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
        ang_abs = angle_rel_to_abs(ang_rel, self.elli_angle)
        command = angle_to_ellocommand(ang_abs)
        self.elliptec.write(command)
        
        # recaluclate relative angle after change
        ang_rel = angle_abs_to_rel(ang_abs, self.elli_angle)
        
        # write to textbox variables
        # self.ell_var.set(ang_abs)
        self.ell_var2.set(ang_rel)
        
        # write to text
        self.label_elli_pos_abs.configure(text = 'Abs pos = %2.2f'%ang_abs)
        self.label_elli_pos_rel.configure(text = 'Rel pos = %2.2f'%ang_rel)


    def elli_set_zero(self):
        ang_rel = float(self.ell_var2.get())
        ang_abs = angle_rel_to_abs(ang_rel, self.elli_angle)
        self.elli_angle = ang_abs

    
    def esc_key_btn(self, value):        
        self.buttonRotate2.focus_set()
    

    def image_move(self, event):
        print('x=%d, y=%d'%(event.x, event.y))
    
        
    def save_img(self):
        self.save_image = True

      
    def show_frame(self):
        frame, frame_is_new = self.cam_reader.get_frame()
            
        # depending on camera_image_type, cv2image will be selected
        # no transformations
        if self.camera_image_type == 'RAW':
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #A
        
        # dynamic just shows two opposite polarized images
        if self.camera_image_type == 'DYN':
            self.cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # also no direct image transfromarmations 
            self.camera_map_counter += 1
            if self.camera_map_counter == self.frame_map_switch: 
                rel_ang = 90 - self.polrot 
                self.rotate_elli(rel_ang)            
            if self.camera_map_counter == 2*self.frame_map_switch: 
                self.camera_map_counter = 0
                rel_ang = 90 + self.polrot        
                self.rotate_elli(rel_ang)  
            
        # map shows difference between two polarizations (at the end of each cycle (from 0 to FMS and from FMS to 2*FMS) it calculates the map and switches the polarizer)
        if self.camera_image_type == 'MAP':
            self.camera_map_counter += 1
            if self.camera_map_counter == self.frame_map_switch: 
                rel_ang = 90 - self.polrot
                self.rotate_elli(rel_ang)
                if self.map_gauss > 0:
                    norm_cur = gaussian_filter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)/255, sigma = self.map_gauss)
                    norm_prev = gaussian_filter(self.prev_grayframe.astype(float)/255, sigma = self.map_gauss)
                else:                
                    norm_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)/255
                    norm_prev = self.prev_grayframe.astype(float)/255
                self.cv2image = cmap_array(norm_cur - norm_prev - self.calib_factor*self.calib_grad, self.diff_cmap, rng = self.map_rng)
                self.cv2image = (255*self.cv2image).astype(np.uint8)
                self.cv2image_const = self.cv2image.copy()
                self.prev_grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.camera_map_counter == 2*self.frame_map_switch: 
                self.camera_map_counter = 0
                rel_ang = 90 + self.polrot        
                self.rotate_elli(rel_ang)
                if self.map_gauss > 0:
                    norm_cur = gaussian_filter(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)/255, sigma = self.map_gauss)
                    norm_prev = gaussian_filter(self.prev_grayframe.astype(float)/255, sigma = self.map_gauss)
                else:                
                    norm_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)/255
                    norm_prev = self.prev_grayframe.astype(float)/255
                self.cv2image = cmap_array(norm_prev - norm_cur - self.calib_factor*self.calib_grad, self.diff_cmap, rng = self.map_rng)
                self.cv2image = (255*self.cv2image).astype(np.uint8)
                self.cv2image_const = self.cv2image.copy()
                self.prev_grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            if (self.camera_map_counter != 2*self.frame_map_switch) and (self.camera_map_counter != self.frame_map_switch):
                self.cv2image = self.cv2image_const.copy()
        
        if self.camera_image_type == 'AMAP':
            # sequence of operations is as follows:
            # first rotate polarizer to plus position
            if self.camera_map_counter == 0:
                rel_ang = 90 + self.polrot        
                self.rotate_elli(rel_ang)
            
            # wait few frames and save current frame to memory then rotate the polarizer to minus position
            if self.camera_map_counter == 15:
                self.amap_frameP = frame.copy()
                rel_ang = 90 - self.polrot        
                self.rotate_elli(rel_ang)
                
                # save minus and plus arrays for debugging
                fname = self.path.get()
                dt_string = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
                fname = dt_string + ' ' + fname + ' (p_plus)'
                path_name = data_dir + 'raw_mikro_data/' + fname
                np.save(path_name, self.amap_frameP)
                
            # save the minus frame
            if self.camera_map_counter == 30:
                self.amap_frameM = frame.copy()
                # save minus and plus arrays for debugging
                fname = self.path.get()
                dt_string = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
                fname = dt_string + ' ' + fname + ' (p_minus)'
                path_name = data_dir + 'raw_mikro_data/' + fname
                np.save(path_name, self.amap_frameM)
            
            # calculate the map based on parameters read from text entry
            if self.camera_map_counter == 31:
                params = self.advMapOptions.get()
                params = params.replace(' ', '')
                # text = 'G - gauss sigma, M - maxfilter sigma, Df - degradient resize factor, Tf - domian threshold factor'
                sg = int(get_amap_parameter(params, 'G'))
                mx = int(get_amap_parameter(params, 'M'))
                dg = int(get_amap_parameter(params, 'Df'))
                dt = float(get_amap_parameter(params, 'Tf'))
                
                self.advanced_mapka, self.amap_ranger = advanced_map(self.amap_frameM, self.amap_frameP, sg = sg, degrad_size = dg, maxi = mx)
                
                
            
            
            if self.camera_map_counter < 32:
                self.camera_map_counter += 1
            else:
                self.cv2image = cmap_array(self.advanced_mapka, self.diff_cmap, rng = self.amap_ranger)
                self.cv2image = (255*self.cv2image).astype(np.uint8)
        
        
        # from now, self.cv2image is what will be displayed        
        # overlay camera_draw as red channel with selected alpha if hold is on
        if self.camera_overlay > 0 or self.always_overlay > 0:
            zer = np.zeros((self.camera_draw.shape[0], self.camera_draw.shape[1])).astype(np.uint8)
            red_overlay = cv2.merge((zer, zer, self.camera_draw[:, :, 0]))           
            self.cv2image = cv2.addWeighted(self.cv2image, 1, red_overlay, 0.3, 0.0)

        # draw brush circle
        if self.interaction_mode == 'draw':
            self.cv2image = cv2.circle(self.cv2image, (self.mouse_x, self.mouse_y-76+50), self.brush_size, (220, 80, 80), 2)
        
        if self.interaction_mode == 'point':
            if self.B1_was_pressed == True:  # means mouse was already pressed, draw circle and arrow at place where it was pressed
                # draw stationary circle
                self.cv2image = cv2.circle(self.cv2image, (self.B1_pressed_at[0], self.B1_pressed_at[1]), self.brush_size, (220, 80, 80), 1)
                self.cv2image = cv2.circle(self.cv2image, (self.B1_pressed_at[0], self.B1_pressed_at[1]), 5, (220, 80, 80), -1)
                
                # draw nice arrow indicating eat direction
                dx = self.mouse_x - self.B1_pressed_at[0]+0.00001
                dy = self.mouse_y -76+50 - self.B1_pressed_at[1]
                alfa = np.arctan(dy/dx)
                if (dx<0): alfa += np.pi
                dx = int(self.brush_size*np.cos(alfa))
                dy = int(self.brush_size*np.sin(alfa))
                self.cv2image = cv2.arrowedLine(self.cv2image, self.B1_pressed_at, (self.B1_pressed_at[0]+dx, self.B1_pressed_at[1]+dy), (255, 255, 255), 1)
            else:  # mouse is not pressed, draw circe at current location        
                self.cv2image = cv2.circle(self.cv2image, (self.mouse_x, self.mouse_y-76+50), self.brush_size, (220, 80, 80), 1)
                self.cv2image = cv2.circle(self.cv2image, (self.mouse_x, self.mouse_y-76+50), 5, (220, 80, 80), -1)
  
           
            
        # adds infobar (if not already present, for example when camera_image_type == 'MAP', cv2image does not reset every loop)
        if self.cv2image.shape[0] == 448:
            self.cv2image = self.add_infobar(self.cv2image)
        
        
        # cv2image = cv2.flip(cv2image, 0)
        img = Image.fromarray(self.cv2image)
        self.imgtk = ImageTk.PhotoImage(image=img)
        # self.lmain.imgtk = imgtk
        # self.lmain.configure(image=imgtk)
        # self.canvas.delete("all")
        self.canvas.create_image(400, 300, image=self.imgtk, anchor=CENTER)
        
        # record video
        if self.video_writer != None:
            self.video_writer.write(cv2.cvtColor(self.cv2image, cv2.COLOR_BGR2RGB))
        
        # save image
        if self.save_image == True:
            fname = self.path.get()
            if fname == '': fname = 'unnamed'
            all_files = os.listdir(data_dir + '/saved_images/')
            saved = 0
            num = 0
            while saved == 0:
                if fname + '_%s.png'%str(num).zfill(2) in all_files:
                    num += 1
                else:
                    img.save(data_dir + 'saved_images/' + fname + '_%s.png'%str(num).zfill(2))
                    print(fname + '_%s.png saved!'%str(num).zfill(2))
                    saved =1
                    
            self.save_image = False
            
            
        # save raw frame data
        if self.save_data == True and frame_is_new == True:
            fname = self.path.get()
            dt_string = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
            fname = dt_string + ' ' + fname
            path_name = data_dir + 'raw_mikro_data/' + fname
            np.save(path_name, frame)
            self.save_data = False
           
        # record raw frame (basically save frame but with specified freqency)
        if self.rec_data == True:
            if self.rec_data_cnt >= int(self.savedata_freq_var.get()):
                self.rec_data_cnt = 0
                fname = self.path.get()
                dt_string = datetime.datetime.now().strftime("%d-%m-%Y %H.%M.%S")
                fname = dt_string + ' ' + fname + ' (auto)'
                path_name = data_dir + 'raw_mikro_data/' + fname
                np.save(path_name, frame)
            self.rec_data_cnt += 1


    def add_infobar(self, cv2image):
        # params
        # scalebar_x = 600
        scalebar_y = 10
        scb_h = 20
        
        # create empty panel (with black line at top)
        parameters_panel = np.zeros((bar_height, 800, 3), np.uint8)
        parameters_panel[:,:] = (220, 220, 220)
        parameters_panel[:3, :] = (40, 40, 40)
        
        # add scale bar        
        scalebar_x = 800-scale_px[self.current_obiektyw]-20
        parameters_panel = cv2.rectangle(parameters_panel, (scalebar_x, scalebar_y), (scalebar_x+scale_px[self.current_obiektyw], scalebar_y+scb_h), (40,40,60), -1)
        text = '%s um'%scale_um[self.current_obiektyw]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # add scale text
        textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
        textX = int((scalebar_x+scale_px[self.current_obiektyw]/2) - (textsize[0] / 2))
        textY = int(scalebar_y+scb_h/2 + (textsize[1] / 2))        
        parameters_panel = cv2.putText(parameters_panel, text, (textX, textY), font , 0.5, (240, 240, 250), 1, cv2.LINE_AA)

        # add text (using pillow)
        text = ''
        if self.t_cur > 0:
            text = 'temperature = %2.2f °C\n'%self.t_cur
        else:
            text = 'temperature = sensor error\n'
        text += 'polarizator = %2.2f\n'%float(self.ell_var2.get())
        text += 'objective = %s\n'%self.current_obiektyw
        img_pil = Image.fromarray(parameters_panel)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 10), text, font = self.font11, fill = (10, 10, 10))
        parameters_panel = np.array(img_pil)
        
        
        # merge with camera image (cv2image)
        new_img = np.zeros((100+448, 800, 3), np.uint8)
        new_img[:448, :] = cv2image
        new_img[448:, :] = parameters_panel
        return new_img


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
        
        self.canvas_proj = Canvas(self.projector_window, width=1024, height=768, bg='black', highlightthickness=0, relief='ridge')
        self.canvas_proj.pack(side = LEFT)
    
        
    def refresh_projector_image(self):
        try:
            if self.interaction_mode == 'draw':  # if drawing is on use camera draw and transform it into projector array
                im_out = cv2.warpPerspective(self.camera_draw, self.homomatrix, (1024, 768))
                self.projector_arr = im_out
            if self.interaction_mode == 'point':
                im_out = cv2.warpPerspective(self.camera_point_draw, self.homomatrix, (1024, 768))
                self.projector_arr = im_out
            if self.interaction_mode == 'none':  # for now, just put an eagle there (maybe it should be black screen?)
                if projector_calib_c >= calib_dots_dim**2:  # if it is not calibrating
                    self.projector_arr = self.default_image
            
            if self.hold_proj == 0:
                # img = Image.fromarray(cv2.flip(add_border(self.projector_arr, self.border), 1))
                self.projector_arr_border = add_border(self.projector_arr, self.border)
                img = Image.fromarray(cv2.flip(self.projector_arr_border, 1))
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