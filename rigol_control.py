import time
import pyvisa
from tkinter import messagebox

class RigolController:
    def __init__(self):
        self.rigol = None
        self.laserduty = 0.0
        self.laserstate = 'OFF'

    def connect(self):
        if self.rigol is not None:
            messagebox.showinfo(title='Rigol', message='Rigol is already connected.')
            return True

        rm = pyvisa.ResourceManager()
        try:
            inst = rm.open_resource('USB0::0x1AB1::0x0643::DG8A224704187::INSTR')
            if inst.query("*IDN?").startswith('Rigol Technologies'):
                self.rigol = inst
                self.rigol.write(':SOUR1:APPL:SQU 1000,5,2.5,0')
                return True
        except Exception as e:
            messagebox.showerror(title='Rigol', message=f'Connection to Rigol failed! {e}')
            self.rigol = None
            return False

    def disconnect(self):
        if self.rigol:
            self.rigol.close()
            self.rigol = None

    def set_laserduty(self, value):
        """Sets the laser duty cycle and updates the device."""
        self.laserduty = float(value)
        if self.rigol:
            self.rigol.write(f':SOUR1:FUNC:SQU:DCYC {self.laserduty}')
            time.sleep(0.25)
            response = self.rigol.query(':SOUR1:FUNC:SQU:DCYC?')
            return f"Laser duty cycle set to {response.strip()}."
        return "Rigol device not connected."

    def toggle_laser(self):
        """Toggles the laser ON/OFF state."""
        if not self.rigol:
            messagebox.showwarning(title='Laser', message='Connect to Rigol first to control the laser.')
            return "Rigol not connected."

        # Toggle laser state
        if self.laserstate == 'OFF':
            self.rigol.write(':OUTP1 ON')
            self.laserstate = 'ON'
            return "Laser turned ON."
        else:
            self.rigol.write(':OUTP1 OFF')
            self.laserstate = 'OFF'
            return "Laser turned OFF."
