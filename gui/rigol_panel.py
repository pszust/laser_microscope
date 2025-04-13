# gui/rigol_panel.py
import tkinter as tk
from tkinter import Frame, Label, Button, Entry, StringVar, LEFT, X, Y
import threading

laser_on_color = "#772eff"
laser_off_color = "#5d615c"
subsystem_name_font = ("Segoe UI", 14, "bold")

class RigolPanel:
    def __init__(self, parent, controller):
        self.controller = controller
        self.frame = Frame(parent)
        self.frame.pack(fill=Y)

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        Label(cur_frame, text="LASER CONTROL", font=subsystem_name_font).pack(side=LEFT)

        # Laser status label
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=Y)
        self.lab_laser = Label(cur_frame, text="DUTY = 0.0%, CH1:LASER IS OFF", fg=laser_off_color)
        self.lab_laser.pack(side=LEFT)

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
        self.label_status = Label(cur_frame, text="RIGOL status: unknown", bg="gray")
        self.label_status.pack(side=LEFT)

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
        self.label_status.config(text=f"RIGOL status: {con_state}", bg=con_color)

        # Update laser label
        duty = status.get("duty", status["laserduty"])
        laser_state = status.get("laser", status["laserstate"])
        fg_color = laser_on_color if laser_state == "ON" else laser_off_color
        self.lab_laser.config(text=f"DUTY = {duty}%, CH1:LASER IS {laser_state}", fg=fg_color)
