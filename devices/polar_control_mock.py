# rigol_control.py
import logging
import time
from tkinter import messagebox

logger = logging.getLogger(__name__)


class PolarController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.rotation = 0.0
        self.device = None
        logger.debug(f"Initialization done.")

    def connect(self):
        self.con_stat = "CONNECTING"
        time.sleep(2)
        self.con_stat = "CONNECTED"
        self.device = True

    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    def rotate_rel(self, angle: float) -> None:
        if self.con_stat == "CONNECTED":
            self.rotation += angle
            if self.rotation >= 360:
                self.rotation -= 360
            if self.rotation < 0:
                self.rotation += 360
            time.sleep(0.1)

    def rotate_abs(self, angle: float) -> None:
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
        'rotation':
            float value between 0.0 and 360.0
        """
        return {"connection": self.con_stat, "rotation": self.rotation}
