import os
import threading
import time
from tkinter import messagebox
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.external_executor import ExternalExecutor
from utils.command_handler import Command, ScriptParser, parse_command
from utils.consts import ErrorMsg
from utils.utils import thread_execute
import logging

if TYPE_CHECKING:
    from gui.main_window import MainWindow  # only used for type hints

logger = logging.getLogger(__name__)


class Automation:
    def __init__(self, parent: "MainWindow"):
        self.master = parent
        self.command_list = []
        self.variables = {}
        self.execution_position = [0]
        self.ext_executor = ExternalExecutor()
        self.command_map = {
            "laser_on": self.master.rigol_controller.laser_on,
            "laser_off": self.master.rigol_controller.laser_off,
            "set_laser_duty": self.master.rigol_controller.set_laserduty,
            "sleep": time.sleep,
            "start_animation": self.start_animation,
            "start_anim_gui_params": self.master.animation_control.start_animation_gui_params,
            "display_calibration_dot": self.master.projector_control.set_calibration_img,
            "save_calibration_img": self.save_calibration_img,
            "get_calibration": self.master.projector_control.get_calibration_matrix,
            "get_camera_image": self.get_camera_image,
            "exec_custom": self.execute_custom_func,
            "log_value": self.log_value,
            "display_alt_image": self.display_alt_image,
            "reset_alt_image": self.reset_alt_image,
        }

        self.unknown_no_of_args = (
            "exec_custom"
        )

        self.internal_commands_map = {
            "operator": self.use_operator,
        }

        self.running = False
        self.thread = None
        logger.debug(f"Initialization done.")

    def pass_command(self, command: str):
        parsed_command = parse_command(command)
        self.command_list.append(parsed_command)

    def execute(self):
        if not self.command_list:
            return 0

        # select correct block/command
        current = self.command_list
        for position in self.execution_position:
            if len(current) == position:  # means end of current block
                # exit the block and go to next position
                # means we are done with the whole script
                if len(self.execution_position) == 1:
                    self.execution_position = [0]
                    self.command_list = []
                else:
                    self.execution_position = self.execution_position[:-1]
                    self.execution_position[-1] += 1
                return 0
            else:
                current = current[position]

        if type(current) == list:  # means we have to move in to block
            self.execution_position.append(0)
        elif type(current) == Command:
            # execute command
            temp_msg = f"Cur cmd: {current.command}, args: {current.args}, pos: {self.execution_position}"
            # self.master.console_panel.log(temp_msg)
            self.master.after(0, lambda: self.master.console_panel.log(temp_msg))  # safer

            if current.command == "operator":
                self.use_operator(current)
            elif current.command == "if":
                result = self.use_check(current)
                if result is False:
                    # exit this if block
                    self.execution_position = self.execution_position[:-1]
            elif current.command == "loop":
                if current.args[0] > 0:
                    current.args[0] -= 1
                else:
                    # exit loop block
                    self.execution_position = self.execution_position[:-1]
            elif current.command == "restart_block":
                self.execution_position[-1] = -1  # this is because at the end there is += 1

            # execute script according to map
            if current.command in self.command_map:
                arguments = []
                for arg in current.args:
                    if type(arg) is str:
                        arg = self.variables.get(arg, arg)
                    arguments.append(arg)
                func = self.command_map[current.command]
                func(arguments) if current.command in self.unknown_no_of_args else func(*arguments)

            # go to next position
            self.execution_position[-1] += 1

    def use_operator(self, cmd: Command):
        l_value = cmd.args[0]
        operator = cmd.args[1]
        r_value = cmd.args[2]

        # check if r_value is not a command itself and execute it if true
        if r_value in self.command_map:
            nested_cmd = Command(r_value, [str(arg) for arg in cmd.args[3:]])
            arguments = []
            for arg in nested_cmd.args:
                if type(arg) is str:
                    arg = self.variables.get(arg, arg)
                arguments.append(arg)
            func = self.command_map[nested_cmd.command]
            r_value = func(arguments) if nested_cmd.command in self.unknown_no_of_args else func(*arguments)
            if r_value is None:
                logger.warning(f"Command {nested_cmd.get_format()} did not return any value!")

        if operator == "=":
            self.variables[l_value] = r_value
        elif operator == "+=":
            if l_value in self.variables:
                self.variables[l_value] += r_value
            else:
                raise (ValueError(ErrorMsg.err_var_missing.format(cmd.command, cmd.args, l_value)))
        elif operator == "-=":
            if l_value in self.variables:
                self.variables[l_value] -= r_value
            else:
                raise (ValueError(ErrorMsg.err_var_missing.format(cmd.command, cmd.args, l_value)))

    def use_check(self, cmd: Command):
        l_value = cmd.args[0]
        check = cmd.args[1]
        r_value = cmd.args[2]

        if type(l_value) == str:
            if l_value in self.variables:
                l_value = self.variables[l_value]
            else:
                raise (ValueError(ErrorMsg.err_var_missing.format(cmd.command, cmd.args, l_value)))

        if type(r_value) == str:
            if r_value in self.variables:
                r_value = self.variables[r_value]
            else:
                raise (ValueError(ErrorMsg.err_var_missing.format(cmd.command, cmd.args, r_value)))

        check_map = {
            "==": l_value == r_value,
            ">=": l_value >= r_value,
            "<=": l_value <= r_value,
            ">": l_value > r_value,
            "<": l_value < r_value,
        }
        if check in check_map:
            return check_map[check]
        else:
            raise (ValueError(f"{check} invalid"))

    def execute_script_file(self, path):
        scr = ScriptParser()
        script_lines = scr.load_script(path)
        scr.parse(script_lines)
        self.command_list = scr.commands

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        while self.running:
            self.execute()
            time.sleep(0.1)

    def start_animation(self, posx, posy, angle, size, duration, anim_name):
        target = {
            "posx": posx,
            "posy": posy,
            "angle": angle,
            "size": size,
            "duration": duration,
            "anim_path": anim_name,
        }
        self.master.animation_control.start_animation(target)

    def save_calibration_img(self, num: int):
        if num == 999:
            path = f"calibration/calibration_array_baseline.npy"
        else:
            path = f"calibration/calibration_array_{str(num).zfill(2)}.npy"
            
        self.master.camera_controller.save_as_array(path)
    
    def execute_custom_func(self, args):
        result = self.ext_executor.execute_custom_func(args)
        return result

    def log_value(self, value):
        logger.info(f"{value} logged")

    def get_camera_image(self) -> Image:
        return self.master.camera_controller.get_image()
    
    def display_alt_image(self, image):
        self.master.camera_panel.display_alt_image(image)
    
    def reset_alt_image(self):
        self.master.camera_panel.reset_alt_image()


