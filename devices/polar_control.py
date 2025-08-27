# rigol_control.py
import logging
import time
from tkinter import messagebox
from pylablib.devices import Thorlabs

logger = logging.getLogger(__name__)

DEG_TO_STEP = 0.732420/100000
DEVICE_ID = "55520124"


class PolarController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.rotation = 0.0
        self.device = None
        logger.debug(f"Initialization done.")

    def connect(self):
        self.con_stat = "CONNECTING"
        self.device = Thorlabs.KinesisMotor(DEVICE_ID)
        self.con_stat = "CONNECTED"

    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    def rotate_abs(self, angle: float) -> None:
        self.rotation = angle
        self._rotate()

    def rotate_rel(self, angle: float) -> None:
        self.rotation += angle
        self._rotate()

    def _rotate(self):
        if self.con_stat == "CONNECTED":
            if self.rotation >= 360:
                self.rotation -= 360
            if self.rotation < 0:
                self.rotation += 360
            step = self._angle_to_steps(self.rotation)
            self.device.move_to(step)
        else:
            logger.warn("Attempting to rotate disconnected device")

    def _update_postion(self) -> None:
        if self.device:
            step = self.device.get_position()
            self.rotation = self._step_to_angle(step)

    def _angle_to_steps(self, angle: float) -> int:
        return int(angle/DEG_TO_STEP)

    def _step_to_angle(self, step: int) -> float:
        return step * DEG_TO_STEP

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
        self._update_postion()
        return {"connection": self.con_stat, "rotation": self.rotation}
