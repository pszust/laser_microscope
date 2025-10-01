import logging
import tkinter as tk
from tkinter import Button, Canvas, Entry, Frame, Label, StringVar

from PIL import Image, ImageTk

import utils.consts as consts
import utils.image_utils as imgut
from controls.areal_control import ArealControl
from core.automation import Automation
from utils.command_handler import Command, parse_command
from utils.utils import activate_btn, deactivate_btn, thread_execute

logger = logging.getLogger(__name__)

UPDATE_MINIMAP_DELAY = 5


class ArealTab:
    def __init__(self, parent, control: ArealControl):
        self.control = control
        self.active_mapping = False
        self.update_minimap_counter = 0
        self.minimap_size = self.control.MINIMAP_SIZE

        self.frame = Frame(parent)
        self.frame.pack(fill=tk.Y)

        # Title
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="AREAL TAB", font=consts.subsystem_name_font).pack(side=tk.LEFT)

        # mini-info
        cur_frame = Frame(self.frame)
        cur_frame.pack(fill=tk.Y)
        Label(cur_frame, text="Before using any mapping functions, you need to activate mapping").pack(
            side=tk.LEFT
        )

        # --- Minimap ---
        self.minimap_canvas = Canvas(
            self.frame,
            width=self.minimap_size,
            height=self.minimap_size,
            bg="#000000",
            highlightthickness=1,
            highlightbackground="#333333",
        )
        self.minimap_canvas.pack(pady=6)
        self.minimap_canvas.bind("<Double-Button-1>", self._on_minimap_double_click)

        # Keep a reference to the PhotoImage so it isn't GC'd
        self._minimap_img = None
        self.control.reset_minimap()
        self.update_minimap()

        # --- Controls: active mapping ---
        btn_frame = Frame(self.frame)
        btn_frame.pack(fill=tk.X)
        self.btn_active_mapping = Button(
            btn_frame, text="Active mapping", command=self.enable_mapping, bg="#f2ffd9"
        )
        self.btn_active_mapping.pack(side=tk.LEFT, padx=3, pady=3)
        Button(btn_frame, text="Reset minimap", command=self.control.reset_minimap, bg="#f2ffd9").pack(
            side=tk.LEFT, padx=3, pady=3
        )
        Button(btn_frame, text="Save minimap", command=self.control.save_minimap, bg="#f2ffd9").pack(
            side=tk.LEFT, padx=3, pady=3
        )

        # --- Controls: Standard mapping ---
        frame = Frame(self.frame)
        frame.pack(fill=tk.X)

        # Row 1: Start point
        row = Frame(self.frame)
        row.pack(fill=tk.X, pady=2)
        Label(row, text="Start point").pack(side=tk.LEFT, padx=(0, 6))

        Label(row, text="X:").pack(side=tk.LEFT)
        self.var_start_x = StringVar(value="0.0")
        Entry(row, textvariable=self.var_start_x, width=6).pack(side=tk.LEFT, padx=(0, 6))

        Label(row, text="Y:").pack(side=tk.LEFT)
        self.var_start_y = StringVar(value="0.0")
        Entry(row, textvariable=self.var_start_y, width=6).pack(side=tk.LEFT, padx=(0, 6))

        Button(row, text="Here", command=self.set_start_here, bg="#f2ffd9").pack(side=tk.LEFT, padx=3)

        # Row 2: End point
        row = Frame(self.frame)
        row.pack(fill=tk.X, pady=2)
        Label(row, text="End point").pack(side=tk.LEFT, padx=(0, 6))

        Label(row, text="X:").pack(side=tk.LEFT)
        self.var_end_x = StringVar(value="0.0")
        Entry(row, textvariable=self.var_end_x, width=6).pack(side=tk.LEFT, padx=(0, 6))

        Label(row, text="Y:").pack(side=tk.LEFT)
        self.var_end_y = StringVar(value="0.0")
        Entry(row, textvariable=self.var_end_y, width=6).pack(side=tk.LEFT, padx=(0, 6))

        Button(
            row, text="Here", command=self.set_end_here, bg="#f2ffd9"  # non-existent method (to implement)
        ).pack(side=tk.LEFT, padx=3)

        # Row 3: Make areal map
        row = Frame(self.frame)
        row.pack(fill=tk.X, pady=4)
        Button(
            row,
            text="Make areal map",
            command=self.make_areal_map,  # non-existent method (to implement)
            bg="#f2ffd9",
        ).pack(side=tk.LEFT, padx=3, pady=3)
        self.var_map_name = StringVar(value="areal_map")
        Entry(row, width=24, textvariable=self.var_map_name).pack(side=tk.LEFT)

        # --- Label ---
        frame = Frame(self.frame)
        frame.pack(fill=tk.X)
        self.lbl_obj = Label(frame, text="info")

    def update(self):
        if self.active_mapping and self.update_minimap_counter > UPDATE_MINIMAP_DELAY:
            self.control.update_minimap()
            self.update_minimap()
            self.update_minimap_counter = 0
        self.update_minimap_counter += 1

        # if these cause lag - rework back to on-click
        self.set_start_byentry()
        self.set_end_byentry()
        self.control.current_obj_for_mapping = self.control.master.camera_panel.get_objective()

        # update mapping params
        txt = (
            "Current areal mapping parameters:",
            f"Start position: {self.control.amap_start_mm} [mm]",
            f"End position: {self.control.amap_end_mm} [mm]",
            f"Selected objective: {self.control.current_obj_for_mapping}",
            "After changing the objective - reset the minimap",
        )
        self.lbl_obj.config(text="\n".join(txt))
        self.lbl_obj.pack(side=tk.LEFT)

    def update_minimap(self) -> None:
        # just drawing
        minimap = self.control.minimap
        overlay = self.control.overlay
        combine = imgut.add_image(minimap, overlay, (0, 0))
        self._minimap_img = ImageTk.PhotoImage(Image.fromarray(combine))
        self.minimap_canvas.delete("all")
        self.minimap_canvas.create_image(0, 0, image=self._minimap_img, anchor=tk.NW)

    def enable_mapping(self) -> None:
        if self.active_mapping:
            self.active_mapping = False
            deactivate_btn(self.btn_active_mapping)
        else:
            self.active_mapping = True
            activate_btn(self.btn_active_mapping)

    def set_start_here(self):
        stage_status = self.control.master.stage_controller.get_status()
        x, y = stage_status.get("x_pos"), stage_status.get("y_pos")
        self.control.amap_start_mm = (x, y)
        self.var_start_x.set(f"{x:.2f}")
        self.var_start_y.set(f"{y:.2f}")

    def set_end_here(self):
        stage_status = self.control.master.stage_controller.get_status()
        x, y = stage_status.get("x_pos"), stage_status.get("y_pos")
        self.control.amap_end_mm = (x, y)
        self.var_end_x.set(f"{x:.2f}")
        self.var_end_y.set(f"{y:.2f}")

    def set_start_byentry(self):
        try:
            x = float(self.var_start_x.get())
            y = float(self.var_start_y.get())
            self.control.amap_start_mm = (x, y)
        except:
            pass

    def set_end_byentry(self):
        try:
            x = float(self.var_end_x.get())
            y = float(self.var_end_y.get())
            self.control.amap_end_mm = (x, y)
        except:
            pass

    def make_areal_map(self):
        # controls
        if (
            self.control.amap_start_mm[0] >= self.control.amap_end_mm[0]
            or self.control.amap_start_mm[1] >= self.control.amap_end_mm[1]
        ):
            logger.warning("Cannot make map like this!")
            return
        if self.control.master.stage_controller.get_status().get("connection") != "CONNECTED":
            logger.warning("Cannot make areal map without XY-stage connected")
            return
        if not self.active_mapping:
            self.enable_mapping()

        map_name = self.var_map_name.get()
        self.control.make_areal_map(map_name)

    def _on_minimap_double_click(self, event):
        # this is experimental feature
        if not self.active_mapping:
            self.enable_mapping()
        x, y = int(event.x), int(event.y)
        self.control.move_on_minimap(x, y)
