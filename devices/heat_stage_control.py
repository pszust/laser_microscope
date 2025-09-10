import logging
import time
from tkinter import messagebox

import serial

from utils.utils import thread_execute

logger = logging.getLogger(__name__)


class HeatController:
    def __init__(self):
        self.device = None
        self.con_stat = "UNKNOWN"
        self.port = 0
        self.cur_temp = -1.0
        self.set_temp = -1.0
        self.rate = -1.0
        logger.debug(f"Initialization done.")

    @thread_execute
    def connect(self):
        if self.device is not None:
            messagebox.showinfo(title="Heat stage", message="Heat stage is already connected.")
            return True
        else:
            port=f"COM{self.port}"
            self.con_stat = "CONNECTING"
            ser = serial.Serial()
            ser.baudrate = 115200
            ser.port=port
            ser.timeout = 6

            ser.parity = serial.PARITY_NONE
            ser.bytesize = serial.EIGHTBITS
            ser.stopbits = serial.STOPBITS_ONE
            
            try:
                ser.open()    
                ser.write(b'tact?\r')
                resp = ser.read_until(b'>')
                if str(resp).find('tact?') >= 0: 
                    logging.info(f"Connected to thortemp on {port}")
                    self.device = ser
                    self.con_stat = "CONNECTED"
                else:
                    logging.info(f"{port} is not thortemp")
                    logging.info('Response: %s'%resp)
                    return None
            except:
                logging.warning(f"Device on {port} is not available")
                return None

    @thread_execute
    def disconnect(self):
        if self.device:
            self.device = None
            self.con_stat = "NOT CONNECTED"

    def get_status(self):
        return {
            "connection": self.con_stat,
            "current_temp": self.cur_temp,
            "set_temp": self.set_temp,
            "rate": self.rate,
        }

    @thread_execute
    def set_temperature(self, value: float):
        if self.device != None:
            self.device.write(b'tset=%2.1f'%value)

    @thread_execute
    def set_rate(self, value):
        self.rate = float(value)

    def thortemp_check_enable(self):
        if self.device != None:
            self.device.write(b'stat?\r')
            resp = self.device.read_until(b'>')
            resp_str = str(resp)

            if resp_str.find('stat') >= 0:
                value = resp_str[resp_str.find('\\r')+2 : resp_str.find('>')-1]
                stat_byte = bin(int(value, 16))
                return int(stat_byte[-1])  # return first bit as it represents the state of ENABLE
        else:
            logging.warning('Not connected')
    
    
    def thortemp_switch_enable(self):
        if self.device != None:
            self.device.write(b'ens\r')
            
    def thortemp_set_ramp(self, ramp: float):
        if self.device != None:
            self.device.write(b'ramp=%d'%ramp)
            
    def thortemp_get_temp(self) -> float:
        if self.device != None:
            self.device.write(b'tact?\r')
            resp = self.device.read_until(b'>')
            
            resp_str = str(resp)

            if resp_str.find('tact') >= 0:
                try:
                    value = float(resp_str[resp_str.find('\\r')+2 : resp_str.find('C\\')-1])
                    return value
                except:
                    print('Some stupid error occured, but dont worry!')
                    print(resp_str)
                    return -1
        return -1
