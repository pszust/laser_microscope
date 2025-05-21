import time
from tkinter import messagebox

import pyvisa
# import logging

# logger = logging.getLogger(__name__)


class RigolController:
    def __init__(self):
        self.rigol = None
        self.con_stat = "UNKNOWN"
        self.laserstate = "OFF"
        self.laserduty = 0.0
        # logger.debug(f"Initialization done.")

    def connect(self):
        self.con_stat = "CONNECTING"
        if self.rigol is not None:
            messagebox.showinfo(title="Rigol", message="Rigol is already connected.")
            return True

        rm = pyvisa.ResourceManager()
        try:
            inst = rm.open_resource('USB0::0x1AB1::0x0643::DG8A220800267::INSTR')
            if inst.query("*IDN?")[:18] == 'Rigol Technologies':
                self.rigol = inst
                inst.write(':SOUR1:APPL:SQU 2000,5,2.5,0')
            self.con_stat = "CONNECTED"
            return True
        except Exception as e:
            messagebox.showerror(title="Rigol", message=f"Connection to Rigol failed! {e}")
            self.rigol = None
            return False

    def disconnect(self):
        if self.rigol:
            self.rigol.close()
            self.rigol = None
        self.con_stat = "NOT CONNECTED"

    def set_laserduty(self, value):
        """Sets the laser duty cycle and updates the device."""
        self.laserduty = float(value)
        if self.rigol:
            self.rigol.write(f":SOUR1:FUNC:SQU:DCYC {self.laserduty}")
            time.sleep(0.25)
            response = self.rigol.query(":SOUR1:FUNC:SQU:DCYC?")
            return f"Laser duty cycle set to {response.strip()}."
        return "Rigol device not connected."

    def toggle_laser(self):
        """Toggles the laser ON/OFF state."""
        if not self.rigol:
            messagebox.showwarning(title="Laser", message="Connect to Rigol first to control the laser.")
            return "Rigol not connected."

        # Toggle laser state
        if self.laserstate == "OFF":
            self.rigol.write(":OUTP1 ON")
            self.laserstate = "ON"
            return "Laser turned ON."
        else:
            self.rigol.write(":OUTP1 OFF")
            self.laserstate = "OFF"
            return "Laser turned OFF."

    def laser_on(self):
        if self.con_stat != "CONNECTED":
            return "Not connected."
        self.rigol.write(":OUTP1 ON")
        self.laserstate = "ON"

    def laser_off(self):
        if self.con_stat != "CONNECTED":
            return "Not connected."
        self.rigol.write(":OUTP1 OFF")
        self.laserstate = "OFF"

    def get_status(self):
        """Possible values are
        'connection':
            'CONNECTED',
            'CONNECTING',
            'UNKNOWN',
            'NOT CONNECTED'
        'laserstate':
            'ON',
            'OFF'
        'laserduty':
            float value between 0.0 and 1.0
        """
        return {
            "connection": self.con_stat,
            "laserstate": self.laserstate,
            "laserduty": self.laserduty,
        }
