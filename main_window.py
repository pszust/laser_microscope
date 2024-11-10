import tkinter as tk
import cv2
from tkinter import (
    Frame, Label, Button, Entry, Text, StringVar,
    Canvas, Menu, filedialog, Scrollbar, END, LEFT, RIGHT, BOTH, Y, X, W, E, N, S, NW
)
from rigol_control import RigolController
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
import time
import pyvisa
import serial
import sys


padd = 2
group_name_font = ('Segoe UI', 16)
subsystem_name_font = ('Segoe UI', 14, 'bold')
laser_on_color = '#772eff'
laser_off_color = '#5d615c'

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


class MainWindow(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master

        self.rigol_controller = RigolController()

        self.elliptec_angle_var = StringVar()

        self.projector_window = None

        # self.pack(fill=BOTH, expand=True)
        self.create_widgets()

    def create_widgets(self):
        # Create the menu bar
        self.create_menu()

        # COLUMN 0
        column_frame = Frame(self.master)
        column_frame.grid(row=0, column=0, padx = padd, sticky=N+S+E+W)

        frame = Frame(column_frame)
        frame.grid(row=0, column=0, padx = padd, sticky=N+S+E+W)
        frame.grid_columnconfigure(0, weight=1)  # Make the frame expand horizontally
        self.create_camera_frame(frame)

        frame = Frame(column_frame)
        frame.grid(row=1, column=0, padx = padd, sticky=N+S+E+W)
        frame.grid_columnconfigure(0, weight=1)  # Make the frame expand horizontally
        self.create_console_frame(frame)

        # COLUMN 1
        column_frame = Frame(self.master)
        column_frame.grid(row=0, column=1, padx = padd, sticky=N+S+E+W)
        
        frame = Frame(column_frame)
        frame.pack(fill = Y, padx = padd)
        self.create_projector_frame(frame)

        frame = Frame(column_frame)
        frame.pack(fill = Y, padx = padd)
        self.create_rigol_frame(frame)

        frame = Frame(column_frame)
        frame.pack(fill = Y, padx = padd)
        self.create_elliptec_frame(frame)

    def create_menu(self):
        self.menu = Menu(self.master)
        self.master.config(menu=self.menu)
        self.file_menu = Menu(self.menu, tearoff=False)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Load Image", command=self.load_image)
        self.file_menu.add_command(label="Exit", command=self.master.quit)

    def create_camera_frame(self, parent):

        self.canvas = Canvas(parent, width=800, height=600, bg='black')
        self.canvas.grid(row=0, column=0, sticky=W+E)

        # Placeholder image in canvas
        self.display_placeholder_image()

    def create_console_frame(self, parent):
        # console view
        frame = Frame(parent)
        frame.grid(row=0, column=0, sticky=W+E)
        self.console = Text(frame, wrap='word', height=15)
        self.console.pack(side=LEFT, fill=X, expand=True)

        scroll_bar = Scrollbar(frame, command=self.console.yview)
        scroll_bar.pack(side=RIGHT, fill=Y)
        self.console.configure(yscrollcommand=scroll_bar.set)

        # Input field for console
        frame = Frame(parent)
        frame.grid(row=1, column=0, sticky=W+E)
        self.console_input_label = Label(frame, text="Input Command:")
        self.console_input_label.pack(side=LEFT, fill=BOTH, expand=True)

        self.console_input = Entry(frame, width=50)
        self.console_input.pack(side=LEFT, fill=BOTH)
        self.console_input.bind('<Return>', self.process_console_input)

    def create_rigol_frame(self, parent):
        # Rigol laser control frame
        cur_frame = Frame(parent)
        cur_frame.pack(fill=Y)
        
        Label(cur_frame, text='LASER CONTROL', font=subsystem_name_font).pack(side=LEFT)
        
        self.lab_laser = Label(cur_frame, text='DUTY = 0.0%, CH1:LASER IS OFF', fg=laser_off_color)
        self.lab_laser.pack(side=LEFT)
        
        # Duty cycle controls
        cur_frame = Frame(parent)
        cur_frame.pack(fill=Y)
        
        Label(cur_frame, text='Set duty cycle CH1:').pack(side=LEFT)
        self.laserduty_var = StringVar(value='1')
        Entry(cur_frame, width=15, textvariable=self.laserduty_var).pack(side=LEFT, fill=X)

        Button(cur_frame, text='Set', command=self.set_laserduty).pack(side=LEFT)
        Button(cur_frame, text='CH1:laser', command=self.toggle_laser).pack(side=LEFT)

        # Rigol connection
        cur_frame = Frame(parent)
        cur_frame.pack(fill=Y)
        Button(cur_frame, text='Connect to Rigol', command=self.connect_rigol).pack(side=LEFT)
        self.label_rigol_status = Label(cur_frame, text='RIGOL status: unknown', bg='gray')
        self.label_rigol_status.pack(side=LEFT)


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

    def create_elliptec_frame(self, eli_frame):       
        # elliptec frame name
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        lab = Label(cur_frame, text = 'ELL14 TOP CONTROL')
        lab.config(font=('Segoe UI', 14, 'bold'))
        lab.pack(side =  LEFT)
        
        # elliptec position frames
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        lab = Label(cur_frame, text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(0, 0, 0))
        lab.config(font=('Segoe UI', 13))
        lab.pack(side =  LEFT)
                
        # elliptec connection frame
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)
        
        ell_com_var = StringVar(self.master)
        ell_com_var.set('COM3') # default value
        ell_com_menu = tk.OptionMenu(cur_frame, ell_com_var, *serial_ports(), command = self.elli_refresh)
        ell_com_menu.pack(side = LEFT)
        
        Label(cur_frame, text = 'Elliptec status: ').pack(side =  LEFT)        
        self.label_ell_status = Label(cur_frame, text = 'unknown', bg='gray')
        self.label_ell_status.pack(side =  LEFT)       
        
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y)        
        Label(cur_frame, text = 'Rotate relative: ').pack(side =  LEFT)

        # self.ell_var2_top.set(str(angle_abs_to_rel(0, self.ell_offset_top)))
        self.elliptec_angle_var.set("0")
        self.nameEntered2 = Entry(cur_frame, width = 15, textvariable = self.elliptec_angle_var)
        self.nameEntered2.pack(side = LEFT, fill = X)        
        self.buttonRotate2 = Button(cur_frame, text = 'Rotate', command = self.rotate_elli_rel)
        self.buttonRotate2.pack(side = LEFT)        
        self.button = Button(cur_frame, text = 'Set 90 here', command = self.elli_set_offset)
        self.button.pack(side = LEFT)
        
        cur_frame = Frame(eli_frame)
        cur_frame.pack(fill = Y) 
        self.buttonRotateStepM1 = Button(cur_frame, text = '-10', command = lambda: self.rotate_elli_step_top(-10))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-5', command = lambda: self.rotate_elli_step_top(-5))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-1', command = lambda: self.rotate_elli_step_top(-1))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-0.25', command = lambda: self.rotate_elli_step_top(-0.25))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '-0.05', command = lambda: self.rotate_elli_step_top(-0.05))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepM1 = Button(cur_frame, text = '+0.05', command = lambda: self.rotate_elli_step_top(0.05))
        self.buttonRotateStepM1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+0.25', command = lambda: self.rotate_elli_step_top(0.25))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+1', command = lambda: self.rotate_elli_step_top(1))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+5', command = lambda: self.rotate_elli_step_top(5))
        self.buttonRotateStepP1.pack(side = LEFT)
        self.buttonRotateStepP1 = Button(cur_frame, text = '+10', command = lambda: self.rotate_elli_step_top(10))
        self.buttonRotateStepP1.pack(side = LEFT)

    def load_image(self):
        # Placeholder function to load image
        filename = filedialog.askopenfilename(
            initialdir="/", title="Select Image",
            filetypes=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*"))
        )
        if filename:
            self.display_image(filename)
            self.log(f"Loaded image: {filename}")
    
    
    def elli_refresh(self, value):
        # value = ell_com_var2.get()
        self.elliptec = connect_to_elliptec(int(value[3]))
    
        if self.elliptec == None:        
            self.label_ell_status.config(text = 'not connected', bg='gray')
        else:
            self.label_ell_status.config(text = 'connected', bg='lime')

    def set_gain(self):
        # Placeholder function to set camera gain
        gain = self.gain_value.get()
        self.log(f"Camera gain set to {gain}.")

    def save_image(self):
        # Placeholder function to save image
        self.log("Save image clicked.")

    def display_placeholder_image(self):
        # Display a placeholder image in the canvas
        self.canvas.delete("all")
        placeholder = Image.new('RGB', (800, 600), color='grey')
        self.photo = ImageTk.PhotoImage(placeholder)
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def display_image(self, path):
        # Display the selected image in the canvas
        self.canvas.delete("all")
        image = Image.open(path)
        image = image.resize((800, 600), Image.ANTIALIAS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def process_console_input(self, event):
        # Process input from the console input field
        command = self.console_input.get()
        self.log(f"Command entered: {command}")
        self.console_input.delete(0, END)

    def log(self, message):
        # Log messages to the console output
        self.console.insert(END, f"{message}\n")
        self.console.see(END)

    def btn_laserduty(self):
        self.laserduty = float(self.laserduty_var.get())
        self.rigol_set_laserduty(self.laserduty)

    def set_laserduty(self):
        duty = self.laserduty_var.get()
        message = self.rigol_controller.set_laserduty(duty)
        self.log(message)
        self.lab_laser.config(text=f'DUTY = {duty}%, CH1:LASER IS {self.rigol_controller.laserstate}')

    def toggle_laser(self):
        message = self.rigol_controller.toggle_laser()
        state_color = laser_on_color if self.rigol_controller.laserstate == 'ON' else laser_off_color
        self.lab_laser.config(fg=state_color)
        self.log(message)
            
    def connect_rigol(self):
        if self.rigol_controller.connect():
            self.label_rigol_status.config(text='connected', bg='lime')
            self.log("Rigol connected!")
        else:
            self.label_rigol_status.config(text='not connected', bg='red')
            self.log("Rigol connection failed.")

    def rotate_elli(self, ang_rel):
        '''
        rotate elliptec to selected angle (relative angle)
        and set all necessary display options
        '''        
        # set elliptec
        ang_abs = angle_rel_to_abs(ang_rel, self.ell_offset_bottom)
        # ang_abs = ang_rel + self.ell_offset_bottom
        command = angle_to_ellocommand(ang_abs)
        self.elli_bottom_abs = ang_abs
        self.elliptec.write(command)
        
        # write to textbox variables
        # self.ell_var.set(ang_abs)
        self.ell_var2.set(ang_rel)
        
        # write to text
        self.label_ell_bottom.configure(text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(ang_rel, ang_abs, self.ell_offset_bottom))
        
        time.sleep(0.2)

    def rotate_elli_rel(self):
        # get angles
        ang_rel = float(self.ell_var2.get())
        
        # call rotate function and defocuf from text field
        self.rotate_elli(ang_rel)
        self.buttonRotate2.focus_set()
        
        
    def rotate_elli_step(self, value):
        if self.camera_image_type == 'MAP':
            self.ell_offset_bottom += value
        else:
            # get angle
            ang_rel = float(self.ell_var2.get())
            
            # change angle and write
            ang_rel = np.round(ang_rel+value, 5)
            ang_abs = angle_rel_to_abs(ang_rel, self.ell_offset_bottom)
            # ang_abs = ang_rel + self.ell_offset_bottom
            command = angle_to_ellocommand(ang_abs)
            self.elli_bottom_abs = ang_abs
            self.elliptec.write(command)
            
            # recaluclate relative angle after change
            # ang_rel = angle_abs_to_rel(ang_abs, self.ell_offset_bottom)
            
            # write to textbox variables
            # self.ell_var.set(ang_abs)
            self.ell_var2.set(ang_rel)
            
            # write to text
            self.label_ell_bottom.configure(text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(ang_rel, ang_abs, self.ell_offset_bottom))


    def rotate_elli_step_top(self, value):
        # get angle
        ang_rel = float(self.ell_var2_top.get())
        
        # change angle and write
        ang_rel = np.round(ang_rel+value, 5)
        ang_abs = angle_rel_to_abs(ang_rel, self.ell_offset_top)
        # ang_abs = ang_rel + self.ell_offset_bottom
        command = angle_to_ellocommand(ang_abs)
        self.elli_top_abs = ang_abs
        self.elliptec_top.write(command)
        
        # recaluclate relative angle after change
        # ang_rel = angle_abs_to_rel(ang_abs, self.ell_offset_bottom)
        
        # write to textbox variables
        # self.ell_var.set(ang_abs)
        self.ell_var2_top.set(ang_rel)
        
        # write to text
        self.label_ell_top.configure(text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(ang_rel, ang_abs, self.ell_offset_top))


    def elli_set_zero(self):  # obsolete?
        ang_rel = float(self.ell_var2.get())
        ang_abs = angle_rel_to_abs(ang_rel, self.ell_offset_bottom)
        self.ell_offset_bottom = ang_abs
        self.log('Setting self.ell_offset_bottom to %f'%self.ell_offset_bottom)
        self.log('ang_rel = %f'%ang_rel)
        
    def elli_top_set_offset(self):
        self.ell_offset_top = self.elli_top_abs
        self.log('Setting self.ell_offset_top to %f'%self.ell_offset_top)
        
        # write to text
        self.label_ell_top.configure(text = 'Rel=%2.2f, Abs=%2.2f, Off=%2.2f'%(self.ang_rel, self.ang_abs, self.ell_offset_top))

    def initiate_projector_window(self):
        if self.projector_window == None:                
            # self.projector_window = ProjectorWindow(root)
            # self.app = ProjectorWindow(self.projector_window)
            self.projector_window = tk.Toplevel(self.master)
            self.projector_window.title("Projector window - move to projector screen")
            self.projector_window.geometry("400x400")
            self.log('Opened projector window')
            
    
    def close_projector_window(self):
        if self.projector_window != None:
            self.projector_window.destroy()
            self.projector_window = None
            self.log('Closed projector window')

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
        self.canvas_proj.create_image(512, 384, image=self.proj_imgtk, anchor=tk.CENTER)