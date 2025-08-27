# rigol_control.py
import logging
import time
from tkinter import messagebox

from utils.utils import thread_execute
from pylablib.devices import Thorlabs

logger = logging.getLogger(__name__)

DEVICE_ID = "37009479"


class FlipperController:
    def __init__(self, id):
        self.con_stat = "UNKNOWN"
        self.state = "???"
        self.device = None
        logger.debug(f"Initialization done.")

    def connect(self):
        self.con_stat = "CONNECTING"
        try:
            self.device = Thorlabs.kinesis.MFF(DEVICE_ID)
            time.sleep(0.2)
            self.con_stat = "CONNECTED"
        except:
            self.con_stat = "NOT CONNECTED"


    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    @thread_execute
    def flipper_in(self):
        if self.device:
            self.device.move_to_state(1)
            time.sleep(0.25)

    @thread_execute
    def flipper_out(self):
        if self.device:
            self.device.move_to_state(0)
            time.sleep(0.25)

    def _update_state(self):
        state_map = {0: "OUT", 1: "IN"}
        read_state = -1
        if self.device:
            read_state = self.device.get_state()
        self.state = state_map.get(read_state, "???")

    def get_status(self):
        self._update_state()
        return {"connection": self.con_stat, "state": self.state}
