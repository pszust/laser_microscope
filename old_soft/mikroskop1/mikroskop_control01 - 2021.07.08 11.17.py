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

width, height = 800, 600
cap = cv2.VideoCapture(0)
padd = 20
frame_x = 0
frame_y = 0

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


def connect_to_elliptec(com_i):
    ser = serial.Serial()
    ser.baudrate = 9600
    ser.port='COM%d'%com_i
    ser.timeout = 6
    
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
    value = value*398
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
        self.elli_refresh()
    
    
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
        ell_control_frame1 = Frame(root)
        ell_control_frame1.grid(row=row, column=col, padx = padd)
        row += 1
        
        self.label_ell_status = Label(ell_control_frame1, text = 'Elliptec status: ')
        self.label_ell_status.pack(side =  LEFT)
        
        self.label_ell_status2 = Label(ell_control_frame1, text = 'connected', bg='gray')
        self.label_ell_status2.pack(side =  LEFT)
        
        self.button_refresh_ell = Button(ell_control_frame1, text = 'Refresh', command = self.elli_refresh)
        self.button_refresh_ell.pack(side = LEFT)
        
        ell_control_frame = Frame(root)
        ell_control_frame.grid(row=row, column=col, padx = padd)
        row += 1
        
        self.label_ell1 = Label(ell_control_frame, text = 'Elliptec: ')
        self.label_ell1.pack(side =  LEFT)
        
        self.ell_var = IntVar()
        self.ell_var.set(0)
        self.nameEntered = Entry(ell_control_frame, width = 15, textvariable = self.ell_var)
        self.nameEntered.pack(side = LEFT)
        
        self.buttonSave = Button(ell_control_frame, text = 'Rotate', command = self.rotate_elli)
        self.buttonSave.pack(side = LEFT)
        
        
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
    
    
    def elli_refresh(self):
        self.label_ell_status2.config(text = 'not connected', bg='red')
        resp = ''
        try:
            self.elliptec.close()
        except:
            print('Elliptec')
            
        try:
            self.elliptec = ftd.open(0)
            print(self.elliptec.getDeviceInfo())
            self.elliptec.setBaudRate(9600)
            
            # check connection
            self.elliptec.write(b'0in')
            resp = str(ft_read(self.elliptec, 32))
        except:
            print('Elliptec not connected')
            
        if resp.find('0IN0') > 0:        
            self.label_ell_status2.config(text = 'connected', bg='green')
        else:
            self.label_ell_status2.config(text = 'not connected', bg='red')
        
    
    def rotate_elli(self):
        value = self.ell_var.get()*398
        value_hex = str(hex(value)) 
        # it does not understand small letters
        value_hex = value_hex[value_hex.find('x')+1:].zfill(8)
        value_hex = value_hex.replace('a', 'A')
        value_hex = value_hex.replace('b', 'B')
        value_hex = value_hex.replace('c', 'C')
        value_hex = value_hex.replace('d', 'D')
        value_hex = value_hex.replace('e', 'E')
        value_hex = value_hex.replace('f', 'F')
        command = bytes('0ma%s'%value_hex.zfill(8), 'ascii')
        self.elliptec.write(command)
        
    
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