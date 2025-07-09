import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import importlib
import pkgutil
import external_functions

class ExternalExecutor:
    def __init__(self):
        self.loaded_functions = {}

        # Dynamically import all modules in custom_functions package
        for _, module_name, _ in pkgutil.iter_modules(external_functions.__path__):
            full_module_name = f"external_functions.{module_name}"
            module = importlib.import_module(full_module_name)

            # Load all callable functions from the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if callable(attr) and not attr_name.startswith("_"):
                    self.loaded_functions[attr_name] = attr

    def execute_custom_func(self, args):
        func_name = args[0]
        arguments = args[1:]

        if func_name in self.loaded_functions:
            return self.loaded_functions[func_name](*arguments)
        else:
            raise ValueError(f"Function '{func_name}' not found.")