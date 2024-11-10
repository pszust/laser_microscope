import cv2
import matplotlib as mpl
from matplotlib import pyplot as plt
from tkinter import *
from PIL import Image, ImageTk, ImageFont, ImageDraw
import cv2
from PIL import Image, ImageTk
import os

width, height = 800, 600
cap = cv2.VideoCapture(1)
padd = 20
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def save_img():
    global save_image
    save_image = True

  
def show_frame():
    global save_image, path
    
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
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    
    if save_image == True:
        fname = path.get()
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
                
        save_image = False   
    lmain.after(10, show_frame)


def mod_source():
    global cap
    cap.release()
    cap = cv2.VideoCapture(int(sv.get()))
    print(sv.get())
    
    
def exit():
    global cap
    cap.release()
    root.quit()
    

save_image = False

root = Tk()
root.bind('<Escape>', lambda e: exit())

source_frame = Frame(root)
source_frame.grid(row=0, column=0, padx = padd)

sv = StringVar()
source_entry = Entry(source_frame, textvariable=sv)
source_entry.pack()

buttonSource = Button(source_frame, text = 'Change source', command = mod_source)
buttonSource.pack()


lmain = Label(root)
lmain.grid(row=1, column=0, padx = padd)

buttonSave = Button(root, text = 'Save', command = save_img)
buttonSave.grid(row=2, column=0, padx = padd)

path = StringVar()
nameEntered = Entry(root, width = 15, textvariable = path)
nameEntered.grid(row=3, column=0, padx = padd)

show_frame()
root.mainloop()