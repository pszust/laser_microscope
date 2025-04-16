import time
from tkinter import messagebox

import numpy as np


class StageController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.x_pos_set = 0.0
        self.y_pos_set = 0.0
        self.stage = False

    def connect(self):
        self.con_stat = "CONNECTING"
        time.sleep(2)
        self.con_stat = "CONNECTED"
        self.stage = True

    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    def move_absolute_xy(self, x, y):
        # here, command will be send
        dx = self.x_pos - x
        dy = self.y_pos - y
        time_exp = min(8, np.sqrt(dx**2 + dy**2))
        time.sleep(time_exp)
        self.x_pos = x
        self.y_pos = y

    def get_status(self) -> dict:
        """Possible values are
        'connection':
            'CONNECTED',
            'CONNECTING',
            'UNKNOWN',
            'NOT CONNECTED'
        'x_pos':
            float value
        'y_pos':
            float value
        """
        return {"connection": self.con_stat, "x_pos": self.x_pos, "y_pos": self.y_pos}
