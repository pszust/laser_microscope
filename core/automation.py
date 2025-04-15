import time
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from typing import TYPE_CHECKING
from utils.command_handler import ScriptParser, parse_command, Command
from utils.consts import ErrorMsg

if TYPE_CHECKING:
    from gui.main_window import MainWindow  # only used for type hints


class Automation:
    def __init__(self, parent: "MainWindow"):
        self.master = parent
        self.command_list = []
        self.variables = {}
        self.execution_position = [0]
        self.command_map = {
            "laser_on": self.master.rigol_controller.laser_on,
            "laser_off": self.master.rigol_controller.laser_off,
            "set_laser_duty": self.master.rigol_controller.set_laserduty,
        }

        self.internal_commands_map = {
            "operator": self.use_operator,
        }

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
                elif len(self.execution_position) > 1:
                    self.execution_position = self.execution_position[:-1]
                    self.execution_position[-1] += 1
                return 0
            else:
                current = current[position]

        if type(current) == list:  # means we have to move in to block
            self.execution_position.append(0)
        elif type(current) == Command:
            # execute command
            self.master.console_panel.log(f"Cur cmd: {current.command}, args: {current.args}, pos: {self.execution_position}")
            
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

            # go to next position
            self.execution_position[-1] += 1

    def use_operator(self, cmd: Command):
        l_value = cmd.args[0]
        operator = cmd.args[1]
        r_value = cmd.args[2]

        if operator == "=":
            self.variables[l_value] = r_value
        elif operator == "+=":
            if l_value in self.variables:
                self.variables[l_value] += r_value
            else:
                raise(ValueError(ErrorMsg.err_var_missing.format(cmd.command, cmd.args, l_value)))
        elif operator == "-=":
            if l_value in self.variables:
                self.variables[l_value] -= r_value
            else:
                raise(ValueError(ErrorMsg.err_var_missing.format(cmd.command, cmd.args, l_value)))

    def use_check(self, cmd: Command):
        l_value = cmd.args[0]
        check = cmd.args[1]
        r_value = cmd.args[2]

        if type(l_value) == str:
            if l_value in self.variables:
                l_value = self.variables[l_value]
            else:
                raise(ValueError(ErrorMsg.err_var_missing.format(cmd.command, cmd.args, l_value)))

        if type(r_value) == str:
            if r_value in self.variables:
                r_value = self.variables[r_value]
            else:
                raise(ValueError(ErrorMsg.err_var_missing.format(cmd.command, cmd.args, r_value)))
            
        check_map = {
            "==": l_value == r_value,
            ">=": l_value >= r_value,
            "<=": l_value <= r_value,
            ">": l_value > r_value,
            "<": l_value < r_value
        }
        if check in check_map:
            return check_map[check]
        else:
            raise(ValueError(f"{check} invalid"))
        
    def execute_script_file(self, path):
        scr = ScriptParser()
        script_lines = scr.load_script(path)
        scr.parse(script_lines)
        self.command_list = scr.commands

