# rigol_control.py
import time
from tkinter import messagebox


class PolarController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.port = None
        self.rotation = 0.0

    def connect(self, port):
        self.port = port
        self.con_stat = "CONNECTING"
        time.sleep(2)
        self.con_stat = "CONNECTED"
        self.rigol = True

    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    def rotate(self, angle: float) -> None:
        if self.con_stat == "CONNECTED":
            self.rotation = angle
            if self.rotation >= 360:
                self.rotation -= 360
            if self.rotation < 0:
                self.rotation += 360
            time.sleep(0.1)

    def get_status(self) -> dict:
        """Possible values are
        'connection':
            'CONNECTED',
            'CONNECTING',
            'UNKNOWN',
            'NOT CONNECTED'
        'port':
            'ON',
            'OFF'
        'rotation':
            float value between 0.0 and 360.0
        """
        return {"connection": self.con_stat, "port": self.port, "rotation": self.rotation}
