import inspect
import logging
import os
import re
import threading
import time
from tkinter import messagebox
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Tuple, Type

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.external_executor import ExternalExecutor
from utils.command_handler import Command, ScriptParser, parse_command
from utils.consts import ErrorMsg
from utils.utils import thread_execute

if TYPE_CHECKING:
    from gui.main_window import MainWindow  # only used for type hints

logger = logging.getLogger(__name__)

ENABLE_DEBUG_LOGGING = True
CATCH_EXCEPTION = False  # for normal operation set to True, for debugging False
CALLABLE_CONTROLS = [
    "camera_controller",
    "rigol_controller",
    "polar1_controller",
    "polar2_controller",
    "stage_controller",
    "labjack_controller",
    "flipper1_control",
    "flipper2_control",
    "heat_stage_control",
]


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
            "start_animation_target": self.master.animation_control.start_animation,
            "start_animation_man_params": self.start_animation_man_params,
            "start_anim_gui_params": self.master.animation_control.start_animation_gui_params,
            "display_calibration_dot": self.master.projector_control.set_calibration_img,
            "save_calibration_img": self.save_calibration_img,
            "get_calibration": self.master.projector_control.get_calibration_matrix,
            "get_camera_image": self.get_camera_image,
            "exec_custom": self.execute_custom_func,
            "log_value": self.log_value,
            "display_alt_image": self.display_alt_image,
            "reset_alt_image": self.reset_alt_image,
            "load_image_test1": self.load_image_test1,
            "load_image_test2": self.load_image_test2,
            "move_xy_absolute": self.move_xy_absolute,
            "move_xy_relative": self.move_xy_relative,
            "get_m30_state": self.get_m30_state,
            "flipper1_in": self.master.flipper1_control.flipper_in,
            "flipper1_out": self.master.flipper1_control.flipper_out,
        }

        self.unknown_no_of_args = "exec_custom"

        self.internal_commands_map = {
            "operator": self.use_operator,
        }

        self.running = False
        self.thread = None
        self.test_img_roll = 0  # used to load different test images
        self.scan_methods()
        logger.debug(f"Initialization done.")

    def scan_methods(self):
        out: dict[str, callable] = {}
        for control_name in CALLABLE_CONTROLS:
            obj = getattr(self.master, control_name)
            for name, member in inspect.getmembers(obj):
                if name.startswith("_"):
                    continue

                try:
                    bound = getattr(obj, name)
                except Exception:
                    continue

                if inspect.ismethod(bound) or inspect.isfunction(bound) or callable(bound):
                    key = f"{control_name}.{name}"
                    out[key] = bound

        for key, value in out.items():
            self.command_map[key] = value

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
            temp_msg = f"Exec cmd: {current.command}, args: {current.args}, pos: {self.execution_position}"
            if ENABLE_DEBUG_LOGGING:
                logger.debug(temp_msg)

            if current.command == "operator":
                self.use_operator(current)
            elif current.command == "if":
                result = self.use_check(current)
                if result is False:
                    # exit this if block
                    self.execution_position = self.execution_position[:-1]
            elif current.command == "loop":
                if isinstance(current.args[0], str):
                    current.args[0] = self.variables[current.args[0]]
                if current.args[0] > 0:
                    current.args[0] -= 1
                else:
                    # exit loop block
                    self.execution_position = self.execution_position[:-1]
            elif current.command == "break_block":
                for _ in range(current.args[0]):
                    self.execution_position = self.execution_position[:-1]
            elif current.command == "new_block":
                pass
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

        if type(r_value) == str and r_value in self.variables:
            r_value = self.variables[r_value]

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

        # for strings check if they are variables otherwise treat them as strings
        if type(l_value) == str:
            if l_value in self.variables:
                l_value = self.variables[l_value]
        if type(r_value) == str:
            if r_value in self.variables:
                r_value = self.variables[r_value]

        check_map = {
            "==": l_value == r_value,
            "!=": l_value != r_value,
            ">=": l_value >= r_value,
            "<=": l_value <= r_value,
            ">": l_value > r_value,
            "<": l_value < r_value,
        }
        if check in check_map:
            return check_map[check]
        else:
            raise (ValueError(f"{check} invalid"))

    def execute_script_file(self, path: str, args: list | None = None):
        """
        path: name of the script
        args: list of values that will be used to replace '%arg01' - this actually work both ways as the
        replacement is done before any processing - can use floats for example to set specific parameter
        or use str that can be converted to variable if script has specific logic
        """
        scr = ScriptParser()
        script_lines = scr.load_script(path, args=args)
        if script_lines:
            scr.parse(script_lines)
            self.command_list = scr.commands
        else:
            logger.error(f"Script at {path} was not executed")

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def cancel_execution(self, info_only=False):
        self.command_list = []
        self.execution_position = [0]
        if info_only:
            logger.info("Script execution stopped")
        else:
            logger.warning("Execution of script stopped!")

    def _run(self):
        while self.running:
            if CATCH_EXCEPTION:
                try:
                    self.execute()
                except Exception as err:
                    logger.error(err)
                    self.cancel_execution()
            else:
                self.execute()
            time.sleep(0.1)

    def update_variables(self, new_variables: dict, optional_msg="") -> None:
        """
        Used to supplement in-script variables from GUI
        """
        if optional_msg:
            logger.info(optional_msg)
        else:
            logger.info("New variables loaded:")
        for name, value in new_variables.items():
            self.variables[name] = value
            if isinstance(value, np.ndarray):
                value = f"<array with shape {value.shape}>"
            logger.info(f"  {name} = {value}")

    def update_variables_from_text(self, text: str, optional_msg="") -> None:
        """
        Used to supplement in-script variables from GUI
        """
        new_variables = self.parse_variables(text)
        self.update_variables(new_variables, optional_msg=optional_msg)

    def save_calibration_img(self, num: int):
        if num == 999:
            path = f"calibration/calibration_array_baseline.npy"
        else:
            path = f"calibration/calibration_array_{str(num).zfill(2)}.npy"

        self.master.camera_controller.save_as_array(path)

    def execute_custom_func(self, args):
        result = self.ext_executor.execute_custom_func(args)
        return result

    def display_alt_image(self, image):
        self.master.camera_panel.display_alt_image(image)

    def reset_alt_image(self):
        self.master.camera_panel.reset_alt_image()

    def move_xy_absolute(self, pos):
        x, y = pos
        self.master.stage_controller.set_postion(x, y)

    def move_xy_relative(self, pos):
        x, y = pos
        self.master.stage_controller.move_rel(x, y)

    @staticmethod
    def parse_variables(text: str) -> dict:
        """
        Parse assignment-like text into a dictionary of {name: value}.

        Supports:
        - comments starting with #
        - ints, floats
        - quoted strings ("..." or '...')
        - ignores empty lines

        Example:
            aaa = 123
            bebebe=  0.213  # comment
            c_12AB ="test1"  # string parameter
            z=-1
        """
        variables = {}
        # Regex: name = value (captures until # or end of line)
        line_re = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)(?:\s*#.*)?$")

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = line_re.match(line)
            if not m:
                continue
            name, val_str = m.groups()
            val_str = val_str.strip()

            # Parse value
            if (val_str.startswith('"') and val_str.endswith('"')) or (
                val_str.startswith("'") and val_str.endswith("'")
            ):
                value = val_str[1:-1]
            else:
                try:
                    if re.match(r"^-?\d+$", val_str):
                        value = int(val_str)
                    elif re.match(r"^-?\d*\.\d+(e[+-]?\d+)?$", val_str, re.I):
                        value = float(val_str)
                    else:
                        value = val_str
                except Exception:
                    value = val_str
            variables[name] = value

        return variables

    # FUNCTIONS

    def get_m30_state(self):
        return self.master.stage_controller.get_status().get("state")

    def log_value(self, *args):
        processed_args = []
        for arg in args:
            processed_args.append(self.variables.get(arg, arg))
        text = ", ".join(str(arg) for arg in processed_args)
        logger.info(f"Logged: {text}")

    def get_camera_image(self) -> Image.Image:
        return self.master.camera_controller.get_image()

    def load_image_test1(self) -> Image.Image:
        path = f"test/utils/mminus-v{str((self.test_img_roll))}.png"
        path = os.path.abspath(os.path.join(os.getcwd(), path))
        return Image.open(path)

    def load_image_test2(self) -> Image.Image:
        path = f"test/utils/mplus-v{str((self.test_img_roll))}.png"
        path = os.path.abspath(os.path.join(os.getcwd(), path))
        self.test_img_roll += 1
        if self.test_img_roll == 8:
            self.test_img_roll = 0
        return Image.open(path)

    def start_animation_man_params(
        self, posx, posy, angle, size, duration, anim_name, variables: tuple = ()
    ):
        target = {
            "posx": posx,
            "posy": posy,
            "angle": angle,
            "size": size,
            "duration": duration,
            "anim_path": anim_name,
            "variables": variables,
        }
        self.master.animation_control.start_animation(target)
