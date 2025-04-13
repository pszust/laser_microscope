import tkinter as tk
from tkinter import Frame, Label, Button, Entry, StringVar, LEFT, X, Y
import threading
import utils.consts as consts

class RigolPanel:
    def __init__(self, parent, controller):
        self.controller = controller
        self.frame = Frame(parent)
        self.frame.pack(fill=Y)

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text="LASER CONTROL", font=consts.subsystem_name_font).pack(side=LEFT)

        # Laser status label
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        self.lbl_laser = Label(cur_frame, text="DUTY = 0.0%, CH1:LASER IS OFF", fg=consts.laser_off_color)
        self.lbl_laser.pack(side=LEFT)

        # Duty cycle controls
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text="Set duty cycle CH1:").pack(side=LEFT)

        self.laserduty_var = StringVar(value="1")
        Entry(cur_frame, width=15, textvariable=self.laserduty_var).pack(side=LEFT, fill=X)

        Button(cur_frame, text="Set", command=self.thread_set_laserduty).pack(side=LEFT)
        Button(cur_frame, text="CH1:laser", command=self.thread_toggle_laser).pack(side=LEFT)

        # Connection controls
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Button(cur_frame, text="Connect to Rigol", command=self.thread_connect).pack(side=LEFT)
        self.lbl_status = Label(cur_frame, text="RIGOL status: unknown", bg="gray")
        self.lbl_status.pack(side=LEFT)

    # Threaded button actions
    def thread_connect(self):
        threading.Thread(target=self.controller.connect, daemon=True).start()

    def thread_set_laserduty(self):
        val = self.laserduty_var.get()
        threading.Thread(target=self.controller.set_laserduty, args=(val,), daemon=True).start()

    def thread_toggle_laser(self):
        threading.Thread(target=self.controller.toggle_laser, daemon=True).start()

    # GUI update hook
    def update(self):
        status = self.controller.get_status()

        # Update connection label
        con_state = status.get("connection", "UNKNOWN")
        con_color = {
            "CONNECTED": "lime",
            "CONNECTING": "yellow",
            "NOT CONNECTED": "gray"
        }.get(con_state, "gray")
        self.lbl_status.config(text=f"RIGOL status: {con_state}", bg=con_color)

        # Update laser label
        duty = status.get("duty", status["laserduty"])
        laser_state = status.get("laser", status["laserstate"])
        fg_color = consts.laser_on_color if laser_state == "ON" else consts.laser_off_color
        self.lbl_laser.config(text=f"DUTY = {duty}%, CH1:LASER IS {laser_state}", fg=fg_color)
