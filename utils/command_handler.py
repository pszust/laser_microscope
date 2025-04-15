import os
import numpy as np


def parse_command(raw_command: str):
    """command looks like this:
        set_laser_power(15)
        turn_on_laser()"""
    
    command = raw_command.strip()
    command = command.replace(" ", "")
    for c in "(),":
        command = command.replace(c, " ")
    command = command.split(" ")
    command_parsed = Command(command[0], command[1:])
    return command_parsed


class Command:
    def __init__(self, command: str, args: list):
        self.command = command
        self.args = [self.handle_arg(arg) for arg in args if arg]

    @staticmethod
    def handle_arg(arg: str):
        if arg.isdigit():
            parsed_arg = int(arg)
        elif "." in arg and arg.replace(".", "").isdigit():
            parsed_arg = float(arg)
        else:
            parsed_arg = arg
        return parsed_arg