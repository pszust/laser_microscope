# rigol_control.py
import time
from tkinter import messagebox

class RigolController:
    def __init__(self):
        self.con_stat = "UNKNOWN"
        self.laserstate = 'OFF'
        self.laserduty = 0.0
        self.rigol = None  # Simulate a Rigol device connection

    def connect(self):
        self.con_stat = "CONNECTING"
        time.sleep(2)  # Simulate delay
        self.con_stat = "CONNECTED"
        self.rigol = True  # Simulate successful connection

    def disconnect(self):
        time.sleep(0.5)
        self.con_stat = "NOT CONNECTED"

    def set_laserduty(self, value):
        """Sets the laser duty cycle and updates the device."""
        self.laserduty = float(value)
        time.sleep(0.6)

    def toggle_laser(self):
        """Toggles the laser ON/OFF state."""
        if not self.rigol:
            messagebox.showwarning(title='Laser', message='Connect to Rigol first to control the laser.')
            return "Rigol not connected."

        # Toggle laser state
        if self.laserstate == 'OFF':
            time.sleep(0.5)
            self.laserstate = 'ON'
        else:
            time.sleep(0.5)
            self.laserstate = 'OFF'

    def laser_on(self):
        if self.con_stat != "CONNECTED":
            return "Not connected."
        time.sleep(0.5)
        self.laserstate = 'ON'

    def laser_off(self):
        if self.con_stat != "CONNECTED":
            return "Not connected."
        time.sleep(0.5)
        self.laserstate = 'OFF'

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
            'connection': self.con_stat,
            'laserstate': self.laserstate,
            'laserduty': self.laserduty
        }
