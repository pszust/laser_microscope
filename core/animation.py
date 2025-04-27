import cv2
import numpy as np
import imutils
import os
from PIL import Image, ImageChops
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gui.main_window import MainWindow


class AnimationControl:
    CUSTOM_ANIMS_DIR = "animations/custom/"
    BASE_ANIMS_DIR = "animations/base/"


    def __init__(self, parent: "MainWindow"):
        self.master = parent
        self.anim_files = []

    def get_anim_files(self):
        self.retrieve_anim_files()
        return self.anim_files
    
    def retrieve_anim_files(self):
        self.anim_files = []
        for directory in (self.CUSTOM_ANIMS_DIR, self.BASE_ANIMS_DIR):
            anims = os.listdir(directory)
            self.anim_files.extend([os.path.join(directory, anim) for anim in anims])



class AnimationInterpreter:
    def __init__(self):
        self.anim_str = None
        self.variables = []

    def load_anim_str(self, path):
        with open(path, "r") as f:
            self.anim_str = f.read()

    def replace_variables(anim_str, variables):
        for n in range(0, len(variables)):
            var_name = 'var%d'%n
            anim_str = anim_str.replace(var_name, '%2.2f'%variables[n])
        return anim_str

    def draw_parametric_animation(self, posx, posy, rota, size, ctime):
        canvas = np.zeros((448, 800, 3), np.uint8)
        # print('PARANIM STR:', anim_str)
        
        # scale parameter is from textfile, size is from brush size 
        anim_str = self.replace_variables(self.anim_str, self.variables)
        for obj in anim_str.split('OBJECT')[1:]:
            splitted = obj.splitlines()

            line1 = splitted[0].split(' ')
            typ = line1[1]

            # init parameters
            cx = 0
            cy = 0
            scale = 1

            for line in splitted[1:]:
                line_splt = line.split(' ')
    #             line_splt = replace_variables(line_splt, variables)
                if eval(line_splt[0]) <= ctime:
                    if eval(line_splt[2]) >= ctime:
                        total_time = eval(line_splt[2])-eval(line_splt[0])
                        completness = (ctime-eval(line_splt[0]))/total_time
                        if line_splt[3] == 'MOVE':
                            change_x = eval(line_splt[7])-eval(line_splt[4])
                            change_y = eval(line_splt[8])-eval(line_splt[5])
                            cx = int(eval(line_splt[4]) + change_x*completness)
                            cy = int(eval(line_splt[5]) + change_y*completness)
                            cx = int(cx*size/100 + 800/2)
                            cy = int(cy*size/100 + 448/2)
                        if line_splt[3] == 'SCALE':
                            change_s = eval(line_splt[6])-eval(line_splt[4])
                            scale = eval(line_splt[4]) + change_s*completness

            if line1[1] == 'rectangle':
                stp = (int(cx-0.5*eval(line1[2])*scale*size/100), int(cy-0.5*eval(line1[3])*scale*size/100))
                enp = (int(cx+0.5*eval(line1[2])*scale*size/100), int(cy+0.5*eval(line1[3])*scale*size/100))
                clr = (eval(line1[4]), eval(line1[4]), eval(line1[4]))
                canvas = cv2.rectangle(canvas, stp, enp, clr, -1)
                
            if line1[1] == 'ellipse':
                axes = (int(eval(line1[2])*scale*size/100), int(eval(line1[3])*scale*size/100))
                clr = (int(line1[5]), int(eval(line1[5])), int(eval(line1[5])))
                canvas = cv2.ellipse(canvas, (cx, cy), axes, 0, 0, 360, clr, -1)

            # now rotate
            canvas_rot = imutils.rotate(canvas, rota)

            # now offset
            pil_img = Image.fromarray(canvas_rot)
            pil_img2 = ImageChops.offset(pil_img, int(posy-800/2), int(posx-448/2))
            canvas_done = np.array(pil_img2)

        return canvas_done