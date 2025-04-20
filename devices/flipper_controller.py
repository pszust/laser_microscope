# rigol_control.py
import time
from tkinter import messagebox
from utils.utils import thread_execute

class FlipperController:
    def __init__(self, id):
        self.con_stat = "UNKNOWN"
        self.state = "OUT"
        self.id = id

    @thread_execute
    def connect(self):
        self.con_stat = "CONNECTING"
        time.sleep(2)
        self.con_stat = "CONNECTED"

    @thread_execute
    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    @thread_execute
    def flipper_in(self):
        if self.state == "OUT":
            time.sleep(0.5)
            self.state = "IN"

    @thread_execute
    def flipper_out(self):
        if self.state == "IN":
            time.sleep(0.5)
            self.state = "OUT"

    def get_status(self):
        return {
            "connection": self.con_stat,
            "state": self.state,
            "id": self.id
        }