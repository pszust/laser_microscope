import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
from tkinter import *
from PIL import Image, ImageTk, ImageFont, ImageDraw
import cv2
from PIL import Image, ImageTk
import os
import sys, ftd2xx as ftd

width, height = 800, 600
cap = cv2.VideoCapture(0)
padd = 20
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


class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()
        self.show_frame()
    
    
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
        self.lmain = Label(root)
        self.lmain.grid(row=row, column=col, padx = padd)
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

        
        
        
    def save_img(self):
        self.save_image = True

      
    def show_frame(self):        
        _, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        
        width = frame.shape[1]
        height = frame.shape[0]
        start_point = (int(width/2), int(height/2) -20)
        end_point = (int(width/2), int(height/2) +20)
        frame = cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
        
        start_point = (int(width/2)-20, int(height/2))
        end_point = (int(width/2)+20, int(height/2))
        frame = cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
        
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        cv2image = cv2.flip(cv2image, 0)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        
        if self.save_image == True:
            fname = self.path.get()
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
        self.lmain.after(10, self.show_frame)


    def mod_source(self):
        global cap
        cap.release()
        cap = cv2.VideoCapture(int(self.source_menu.get()))
        print(self.sv.get())
        
        
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