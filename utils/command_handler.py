import os

import numpy as np


def parse_command(raw_command: str):
    """command looks like this:
    set_laser_power(15)
    turn_on_laser()
    if(var_a > 12){"""

    command = raw_command.strip()
    command = command.replace(" ", "")
    for c in "(),{}":
        command = command.replace(c, " ")
    command = command.split("#")[0]
    command = command.split(" ")
    command_list = []
    operators = ["==", ">=", "<=", "+=", "-=", "=", ">", "<"]
    operators_assign = ["=", "+=", "-="]
    operators_check = ["==", ">=", "<=", "<", ">"]
    for cmd in command:
        # add operator if the line has any of the assign type operators
        if (
            sum((op in cmd for op in operators_assign)) > 0
            and sum((op in cmd for op in operators_check)) == 0
        ):
            command_list = ["operator"]

        # handle the cases where there are a>1 a+=1 etc.
        operators_1c = [op for op in operators if len(op) == 1]
        operators_2c = [op for op in operators if len(op) == 2]
        if sum((op in cmd for op in operators)) > 0:
            for op_1c in operators_1c:
                if op_1c in cmd and sum((op_2c in cmd for op_2c in operators_2c)) == 0:
                    split = cmd.split(op_1c)
                    command_list.extend([split[0], op_1c, split[1]])
            for op_2c in operators_2c:
                if op_2c in cmd:
                    split = cmd.split(op_2c)
                    command_list.extend([split[0], op_2c, split[1]])
        else:
            command_list.append(cmd)
    command_parsed = Command(command_list[0], command_list[1:])
    return command_parsed if command_parsed.command else None


class Command:
    def __init__(self, command: str, args: list):
        self.command = command
        self.args = [self.handle_arg(arg) for arg in args if arg]

    @staticmethod
    def handle_arg(arg: str):
        if arg.isdigit():
            parsed_arg = int(arg)
        elif arg[0] == "-" and arg[1:].isdigit():
            parsed_arg = -int(arg[1:])
        elif "." in arg:
            parsed_arg = float(arg)
        else:
            parsed_arg = arg
        return parsed_arg

    def __call__(self, *args, **kwds):
        print(f"Cmd: {self.command}")
        for arg in self.args:
            print(f"   {arg}: {type(arg)}")

    def get_format(self):
        return f"{self.command}({", ".join((str(a) for a in self.args))})"


class ScriptParser:
    def __init__(self):
        self.commands = []
        self.example_script_path = "custom_scripts/example_script.scrpt"

    def load_script(self, path: str, args: list | None = []) -> list[str]:
        args = [] if not args else args
        scr_lines = self.recurse_construct_script([], path, args=args)
        return scr_lines

    def recurse_construct_script(self, script_lines: list, path: str, args=[]) -> list[str]:
        with open(path, "r") as f:
            cnt = f.read()
            for n, arg in enumerate(args):
                argname = f"%arg{str(n+1).zfill(2)}"
                cnt = cnt.replace(argname, arg)
            lines = cnt.split("\n")
        for line in lines:
            if "SCRPT:" in line:
                arglist = line.strip(" ").split(" ")
                fname = arglist[1]
                args = arglist[2:]
                script_lines.append("new_block{")
                self.recurse_construct_script(script_lines, f"scripts/custom/{fname}", args=args)
                script_lines.append("}")
            else:
                script_lines.append(line)
        return script_lines

    def parse(self, script: list[str]) -> list:
        level = 0
        nested = []
        for line in script:
            current_command = self.commands

            for _ in range(level):
                current_command = current_command[-1]

            if "{" in line:
                level += 1
                current_command.append([])
                current_command = current_command[-1]
                if cmd_parsed := parse_command(line):
                    nested.append(cmd_parsed.command)
            elif "}" in line:
                if nested[-1] == "loop":
                    current_command.append(Command("restart_block", []))
                nested = nested[:-1]
                level -= 1

            parsed_command = parse_command(line)
            if parsed_command:
                current_command.append(parsed_command)

    def print_commands(self):
        self._recurse_print(self.commands)

    def _recurse_print(self, cmd: list | Command, block_intend: int = 0):
        if isinstance(cmd, Command):
            print("  " * block_intend, cmd.get_format())
        else:
            for c in cmd:
                self._recurse_print(c, block_intend + 1)
