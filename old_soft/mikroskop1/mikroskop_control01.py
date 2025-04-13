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

width, height = 800, 600
cap = cv2.VideoCapture(1)
padd = 20
frame_x = 0
frame_y = 0
elli_angle = 0
elli_relative_zero = -32.5
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
    

class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master
        
        self.elliptec = None
        self.video_writer = None

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()
        self.show_frame()
        self.elli_refresh('COM1')
    
    
    #Creation of init_window
    def init_window(self):
        col = 0
        row = 0
        self.save_image = False

        # changing the title of our master widget      
        self.master.title("Mikroskop control")
        
        
        # menu
        menu = Menu(self.master)
        self.master.config(menu=menu)
        
        fileMenu = Menu(menu, tearoff=False)
        fileMenu.add_command(label="Open XRD")
        fileMenu.add_command(label="Exit")
        menu.add_cascade(label="File")
        
        #  -- FIRST COLUMN --
        # main image        
        # self.lmain = Label(root)
        # self.lmain.grid(row=row, column=col, padx = padd)
        self.canvas = Canvas(width=800, height=700, bg='black')
        self.canvas.grid(row=row, column=col, padx = padd)
        self.canvas.bind("<Double-Button-1>", self.image_move)
        row += 1
        
        # image controls
        im_control_frame = Frame(root)
        im_control_frame.grid(row=row, column=col, padx = padd)
        row += 1
        
        source_var = StringVar(self.master)
        source_var.set(0) # default value
        self.source_menu = OptionMenu(im_control_frame, source_var, *list_cameras(), command = self.mod_source)
        self.source_menu.pack(side = LEFT)

        # save controls
        save_frame = Frame(root)
        save_frame.grid(row=row, column=col, padx = padd)
        row += 1
        
        self.label_save = Label(save_frame, text = 'Sample name: ')
        self.label_save.pack(side =  LEFT)
        
        self.path = StringVar()
        self.nameEntered = Entry(save_frame, width = 15, textvariable = self.path)
        self.nameEntered.pack(side = LEFT)
        
        self.buttonSave = Button(save_frame, text = 'Save', command = self.save_img)
        self.buttonSave.pack(side = LEFT)
        
        self.buttonRecord = Button(save_frame, text = 'Record video', command = self.record_video)
        self.buttonRecord.pack(side = LEFT)
        
        
        #  -- SECOND COLUMN --
        row = 0
        col += 1
        
        # elliptec control frame
        ell_control_frame = Frame(root)
        ell_control_frame.grid(row=row, column=col, padx = padd)
        
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
        self.buttonRotate = Button(ell_rot_frame, text = 'Rotate', command = self.rotate_elli)
        self.buttonRotate.pack(side = LEFT)
        
        ell_rot_frame = Frame(ell_control_frame)
        ell_rot_frame.pack(fill = Y)        
        self.label_ell1 = Label(ell_rot_frame, text = 'Rotate relative: ')
        self.label_ell1.pack(side =  LEFT)        
        self.ell_var2 = StringVar()
        self.ell_var2.set('0.0')
        self.nameEntered2 = Entry(ell_rot_frame, width = 15, textvariable = self.ell_var2)
        self.nameEntered2.pack(side = LEFT, fill = X)        
        self.buttonRotate = Button(ell_rot_frame, text = 'Rotate', command = self.rotate_elli_rel)
        self.buttonRotate.pack(side = LEFT)
        
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
        
    
    def rotate_elli(self):
        global elli_angle, elli_relative_zero
        value = float(self.ell_var.get())
        command = angle_to_ellocommand(value)
        self.elliptec.write(command)
        elli_angle = value
        self.label_elli_pos_abs.configure(text = 'Abs pos = %2.2f'%elli_angle)
        self.label_elli_pos_rel.configure(text = 'Rel pos = %2.2f'%(elli_angle-elli_relative_zero+90))        
        
    
    def rotate_elli_rel(self):
        global elli_angle, elli_relative_zero
        value = float(self.ell_var2.get())+elli_relative_zero-90
        command = angle_to_ellocommand(value)
        self.elliptec.write(command)
        elli_angle = value
        self.label_elli_pos_abs.configure(text = 'Abs pos = %2.2f'%elli_angle)
        self.label_elli_pos_rel.configure(text = 'Rel pos = %2.2f'%(elli_angle-elli_relative_zero+90))
        
        
    def rotate_elli_step(self, value):
        global elli_angle, elli_relative_zero
        elli_angle += value
        
        command = angle_to_ellocommand(elli_angle)
        self.elliptec.write(command)
        
        self.label_elli_pos_abs.configure(text = 'Abs pos = %2.2f'%elli_angle)
        self.label_elli_pos_rel.configure(text = 'Rel pos = %2.2f'%(elli_angle-elli_relative_zero+90))


    def image_move(self, event):
        print('x=%d, y=%d'%(event.x, event.y))
    
        
    def save_img(self):
        self.save_image = True

      
    def show_frame(self):
        global frame_x, frame_y, cap
        ret, frame = cap.read()
        if ret == False:
            frame = np.zeros((448, 800, 3), np.uint8)
        
        # frame = cv2.flip(frame, 1)
        frame_x = frame.shape[1]
        frame_y = frame.shape[0]
        
        width = frame.shape[1]
        height = frame.shape[0]
        start_point = (int(width/2), int(height/2) -20)
        end_point = (int(width/2), int(height/2) +20)
        frame = cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
        
        start_point = (int(width/2)-20, int(height/2))
        end_point = (int(width/2)+20, int(height/2))
        frame = cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
        
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        # cv2image = cv2.flip(cv2image, 0)
        img = Image.fromarray(cv2image)
        self.imgtk = ImageTk.PhotoImage(image=img)
        # self.lmain.imgtk = imgtk
        # self.lmain.configure(image=imgtk)
        # self.canvas.delete("all")
        self.canvas.create_image(400, 350, image=self.imgtk, anchor=CENTER)
        
        # record video
        if self.video_writer != None:
            self.video_writer.write(frame)
        
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
        # self.lmain.after(10, self.show_frame)
        self.master.after(50, self.show_frame)


    def mod_source(self, value):
        global cap
        cap.release()
        cap = cv2.VideoCapture(int(value))
        print(value)
        
        
    def exit(self):
        global cap
        cap.release()
        root.quit()
        
       
    
        
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