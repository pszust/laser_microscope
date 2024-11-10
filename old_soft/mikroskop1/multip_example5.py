import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFont, ImageDraw, ImageChops
import os
# import sys, ftd2xx as ftd
import numpy as np
import serial
import sys
import glob
import time
import datetime
from scipy.ndimage.filters import gaussian_filter, maximum_filter
import pyvisa
import random
import imutils
import queue, threading

padd = 1
    


class Window(Frame):
    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   
        

        #reference to the master widget, which is the tk window                 
        self.master = master
        
        
        self.message_queue = queue.Queue()
        self.message_event = '<<message>>'
        self.master.bind(self.message_event, self.process_message_queue)
                
        self.cap = cv2.VideoCapture(0)
        self.frame = np.zeros((448, 800, 3), np.uint8)
        
        self.init_window()
        self.main_loop()
        
        
    def init_window(self):
        self.counter = 0

        # changing the title of our master widget      
        self.master.title("Mikroskop control")
        
        
        current_frame = Frame(root)
        current_frame.pack(fill = Y, padx = padd)
        
        self.canvas = Canvas(current_frame, width=800-4, height=600, bg='black')
        self.canvas.pack(fill = Y, padx = padd)

        
        current_frame = Frame(root)
        current_frame.pack(fill = Y, padx = padd)
        
        self.label = Label(current_frame, text = 'jamnik')
        self.label.pack(side =  LEFT)

        
        
    def main_loop(self):
        ret, self.frame = self.cap.read()
        
        img = Image.fromarray(self.frame)
        self.imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(400, 300, image=self.imgtk, anchor=CENTER)
    
        self.master.after(100, self.main_loop)
        
        
    def send_message_to_ui(self, message):
        self.message_queue.put(message)
        self.master.event_generate(self.message_event, when='tail')

    def process_message_queue(self, event):
        while self.message_queue.empty() is False:
            message = self.message_queue.get(block=False)
            # process the message here
            # print(message)
            self.label.config(text = message)
            
      
class BackgroundThread:

    def __init__(self, tk_thread):
        self.tk_thread = tk_thread
        self.thread = threading.Thread(target=self.run_thread); self.thread.start()
        

    def run_thread(self):
        while(1):
            jamnik_rnd = int(100*random.random())
            
            time.sleep(1)         # just an example... this gives the Tk some time to start up
            tk_thread.send_message_to_ui('jamnik = %d'%jamnik_rnd)
    
        
        
        
if __name__ == '__main__':
    root = Tk()
    # root.geometry("1920x1080")

    w, h = 800, 800
    root.geometry("%dx%d+0+0" % (w, h))
    # root.iconbitmap('bicon02.ico')

    #creation of an instance
    tk_thread = Window(root)
    
    
    back_thread = BackgroundThread(tk_thread) 
    
    root.mainloop()
    