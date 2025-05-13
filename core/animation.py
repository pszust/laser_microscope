import cv2
import numpy as np
import imutils
import os
from PIL import Image, ImageChops
from utils.consts import CamConsts
from typing import TYPE_CHECKING
import time
import logging

if TYPE_CHECKING:
    from gui.main_window import MainWindow

logger = logging.getLogger(__name__)

CUSTOM_ANIMS_DIR = "animations/custom/"
BASE_ANIMS_DIR = "animations/base/"


class AnimationControl:

    def __init__(self, master: "MainWindow"):
        self.master = master
        self.anim_files = []
        self.interpreter = None
        logger.debug(f"Initialization done.")

    def get_anim_files(self):
        self.retrieve_anim_files()
        return self.anim_files

    def retrieve_anim_files(self):
        self.anim_files = []
        for directory in (CUSTOM_ANIMS_DIR, BASE_ANIMS_DIR):
            anims = os.listdir(directory)
            self.anim_files.extend([os.path.join(directory, anim) for anim in anims])

    def start_animation_gui_params(self, x, y, angle, size):
        anim_tab = self.master.anim_tab
        target = {
            "posx": x,
            "posy": y,
            "angle": angle,
            "size": size,
            "duration": int(anim_tab.var_duration.get()),
            "anim_path": anim_tab.var_anim_path.get(),
        }
        self.start_animation(target)

    def start_animation(self, target):
        self.interpreter = AnimationInterpreter(target)
        msg = ", ".join((f"{key}: {val}" for key, val in target.items()))
        logger.info(f"Started {msg}")

    def loop_event(self):
        if self.interpreter:
            drawn_anim = self.interpreter.draw_parametric_animation()
            if drawn_anim is not None:
                self.master.projector_control.update_animation_image(drawn_anim)
            else:
                self.master.projector_control.update_animation_image(None)
                self.interpreter = None
                logger.info(f"Done.")


class AnimationInterpreter:
    def __init__(self, target: dict):
        self.anim_str = self.load_anim_str(target["anim_path"])
        self.posx = target["posx"]
        self.posy = target["posy"]
        self.angle = target["angle"]
        self.size = target["size"]
        self.duration = target["duration"]
        self.start_time = time.time()

        # TODO: apply variables in the future
        # self.variables = []
        # self.anim_str = self.replace_variables(self.anim_str, self.variables)

    def load_anim_str(self, path: str) -> str:
        path2 = os.path.join(CUSTOM_ANIMS_DIR, os.path.basename(path))
        path3 = os.path.join(BASE_ANIMS_DIR, os.path.basename(path))
        if os.path.isfile(path):
            with open(path, "r") as f:
                return f.read()
        elif os.path.isfile(path2):
            with open(path2, "r") as f:
                return f.read()
        elif os.path.isfile(path3):
            with open(path3, "r") as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Animation file {path} not found!")

    def replace_variables(anim_str, variables):
        for n in range(0, len(variables)):
            var_name = "var%d" % n
            anim_str = anim_str.replace(var_name, "%2.2f" % variables[n])
        return anim_str

    def draw_parametric_animation(self) -> np.ndarray | None:
        canv_x, canv_y = CamConsts.SHAPE[:2]
        seconds_elapsed = time.time() - self.start_time
        ctime = seconds_elapsed / self.duration
        if ctime > 1:
            return None

        canvas = np.zeros((canv_y, canv_x, 3), np.uint8)

        # scale parameter is from text anim file, size is from brush size
        for obj in self.anim_str.split("OBJECT")[1:]:
            splitted = obj.splitlines()

            line1 = splitted[0].split(" ")
            typ = line1[1]

            cx = 0
            cy = 0
            scale = 1
            for line in splitted[1:]:
                line_splt = line.split(" ")
                #             line_splt = replace_variables(line_splt, variables)
                if eval(line_splt[0]) <= ctime:
                    if eval(line_splt[2]) >= ctime:
                        total_time = eval(line_splt[2]) - eval(line_splt[0])
                        completness = (ctime - eval(line_splt[0])) / total_time
                        if line_splt[3] == "MOVE":
                            change_x = eval(line_splt[7]) - eval(line_splt[4])
                            change_y = eval(line_splt[8]) - eval(line_splt[5])
                            cx = int(eval(line_splt[4]) + change_x * completness)
                            cy = int(eval(line_splt[5]) + change_y * completness)
                            cx = int(cx * self.size / 100 + canv_x / 2)
                            cy = int(cy * self.size / 100 + canv_y / 2)
                        if line_splt[3] == "SCALE":
                            change_s = eval(line_splt[6]) - eval(line_splt[4])
                            scale = eval(line_splt[4]) + change_s * completness

            if line1[1] == "rectangle":
                stp = (
                    int(cx - 0.5 * eval(line1[2]) * scale * self.size / 100),
                    int(cy - 0.5 * eval(line1[3]) * scale * self.size / 100),
                )
                enp = (
                    int(cx + 0.5 * eval(line1[2]) * scale * self.size / 100),
                    int(cy + 0.5 * eval(line1[3]) * scale * self.size / 100),
                )
                clr = (eval(line1[4]), eval(line1[4]), eval(line1[4]))
                canvas = cv2.rectangle(canvas, stp, enp, clr, -1)

            if line1[1] == "ellipse":
                axes = (
                    int(eval(line1[2]) * scale * self.size / 100),
                    int(eval(line1[3]) * scale * self.size / 100),
                )
                clr = (int(line1[5]), int(eval(line1[5])), int(eval(line1[5])))
                canvas = cv2.ellipse(canvas, (cx, cy), axes, 0, 0, 360, clr, -1)

            # now rotate
            canvas_rot = imutils.rotate(canvas, self.angle)

            # now offset
            pil_img = Image.fromarray(canvas_rot)
            pil_img2 = ImageChops.offset(pil_img, int(self.posy - canv_x / 2), int(self.posx - canv_y / 2))
            canvas_done = np.array(pil_img2)

        return canvas_done
