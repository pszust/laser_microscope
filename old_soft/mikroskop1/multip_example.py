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
from threading import Thread
import datetime
from scipy.ndimage.filters import gaussian_filter, maximum_filter
import pyvisa
import random
import imutils
import multiprocessing

padd = 1

textOut = 'jamnik'
counter = 0

def jamnik_outclass():
    global textOut
    global counter
    print('Jamnik outclass begin')
    
    time.sleep(1)
    counter += 1
    textOut = 'jamnik = %d'%counter
    
    print('Jamnik outclass done!, jamnik = %d'%counter)
    


class Window(Frame):
    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master
        self.init_window()
        self.main_loop()
        
        
    def init_window(self):
        self.counter = 0

        # changing the title of our master widget      
        self.master.title("Mikroskop control")
        
        
        current_frame = Frame(root)
        current_frame.pack(fill = Y, padx = padd)
        
        self.btn1 = Button(current_frame, text = 'Jamnik', command = jamnik_outclass)
        self.btn1.pack(side = LEFT)
        
        self.btn2 = Button(current_frame, text = 'JamnikM', command = self.multi_jamnik)
        self.btn2.pack(side = LEFT)
        
        
        self.label = Label(current_frame, text = textOut)
        self.label.pack(side =  LEFT)
        
        
    def jamnik_clicked(self):
        # self.counter += 1
        time.sleep(1)
        print('Jamnik!')
        # self.label.config(text = 'jamnik = %d'%self.counter)
        
    def multi_jamnik(self):
        p =  multiprocessing.Process(target= jamnik_outclass)
        p.start()
        
        
    def main_loop(self):
        self.label.config(text = textOut)
        self.master.after(100, self.main_loop)
    
        
        
        
if __name__ == '__main__':
    root = Tk()
    # root.geometry("1920x1080")

    w, h = 600, 400
    root.geometry("%dx%d+0+0" % (w, h))
    # root.iconbitmap('bicon02.ico')

    #creation of an instance
    app = Window(root)
    root.mainloop()